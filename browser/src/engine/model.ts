/**
 * GPT-style decoder-only transformer.
 * Every layer hand-written — no abstractions hiding the architecture.
 */

import {
  Tensor,
  matmul,
  add,
  scale,
  softmax,
  relu,
  layerNorm,
  embedding,
  sliceRows,
  causalMask,
  transpose,
} from './tensor';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ModelConfig {
  vocabSize: number;
  blockSize: number;   // context window length
  dModel: number;      // embedding dimension
  nHeads: number;
  nLayers: number;
}

export interface BlockWeights {
  ln1Gamma: Tensor;
  ln1Beta: Tensor;
  Wq: Tensor;
  Bq: Tensor;
  Wk: Tensor;
  Bk: Tensor;
  Wv: Tensor;
  Bv: Tensor;
  Wo: Tensor;
  Bo: Tensor;
  ln2Gamma: Tensor;
  ln2Beta: Tensor;
  ff1W: Tensor;
  ff1B: Tensor;
  ff2W: Tensor;
  ff2B: Tensor;
}

export interface ModelWeights {
  config: ModelConfig;
  tokenEmb: Tensor;    // [vocabSize, dModel]
  posEmb: Tensor;      // [blockSize, dModel]
  blocks: BlockWeights[];
  lnFGamma: Tensor;    // final layer norm
  lnFBeta: Tensor;
  outputW: Tensor;     // [dModel, vocabSize]
  outputB: Tensor;     // [1, vocabSize]
}

/** Stores all intermediate activations needed for the backward pass + visualisation. */
export interface ForwardState {
  // Inputs
  tokens: number[];

  // Embeddings
  tokenEmbOut: Tensor;
  posEmbSlice: Tensor;
  embSum: Tensor;        // tokenEmb + posEmb

  // Per-block activations
  blockStates: BlockForwardState[];

  // Final
  lnFIn: Tensor;
  lnFOut: Tensor;
  lnFMean: Float32Array;
  lnFInvStd: Float32Array;
  logits: Tensor;

  // For visualisation
  attentionWeights: Tensor[][]; // [layer][head] each [seqLen, seqLen]
}

export interface BlockForwardState {
  // Pre-attention
  ln1In: Tensor;
  ln1Out: Tensor;
  ln1Mean: Float32Array;
  ln1InvStd: Float32Array;

  // Attention
  Q: Tensor;
  K: Tensor;
  V: Tensor;
  attnScores: Tensor[];    // per head [seqLen, seqLen] (after softmax)
  attnOut: Tensor;         // after concat + Wo projection
  headOutputs: Tensor[];   // per head [seqLen, headDim]

  // Residual after attention
  resid1: Tensor;

  // Pre-FFN
  ln2In: Tensor;
  ln2Out: Tensor;
  ln2Mean: Float32Array;
  ln2InvStd: Float32Array;

  // FFN
  ff1Out: Tensor;          // before ReLU
  ff1Relu: Tensor;         // after ReLU
  ff2Out: Tensor;

  // Residual after FFN
  resid2: Tensor;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export function initModel(config: ModelConfig): ModelWeights {
  const { vocabSize, blockSize, dModel, nHeads, nLayers } = config;
  const dFF = 4 * dModel;

  const blocks: BlockWeights[] = [];
  for (let i = 0; i < nLayers; i++) {
    blocks.push({
      ln1Gamma: Tensor.ones(1, dModel),
      ln1Beta: Tensor.zeros(1, dModel),
      Wq: Tensor.xavier(dModel, dModel),
      Bq: Tensor.zeros(1, dModel),
      Wk: Tensor.xavier(dModel, dModel),
      Bk: Tensor.zeros(1, dModel),
      Wv: Tensor.xavier(dModel, dModel),
      Bv: Tensor.zeros(1, dModel),
      Wo: Tensor.xavier(dModel, dModel),
      Bo: Tensor.zeros(1, dModel),
      ln2Gamma: Tensor.ones(1, dModel),
      ln2Beta: Tensor.zeros(1, dModel),
      ff1W: Tensor.xavier(dModel, dFF),
      ff1B: Tensor.zeros(1, dFF),
      ff2W: Tensor.xavier(dFF, dModel),
      ff2B: Tensor.zeros(1, dModel),
    });
  }

  return {
    config,
    tokenEmb: Tensor.randn(vocabSize, dModel, 0.02),
    posEmb: Tensor.randn(blockSize, dModel, 0.02),
    blocks,
    lnFGamma: Tensor.ones(1, dModel),
    lnFBeta: Tensor.zeros(1, dModel),
    outputW: Tensor.xavier(dModel, vocabSize),
    outputB: Tensor.zeros(1, vocabSize),
  };
}

/** Collect all parameter tensors for optimiser / gradient zeroing. */
export function allParams(model: ModelWeights): Tensor[] {
  const params: Tensor[] = [
    model.tokenEmb,
    model.posEmb,
    model.lnFGamma,
    model.lnFBeta,
    model.outputW,
    model.outputB,
  ];
  for (const b of model.blocks) {
    params.push(
      b.ln1Gamma, b.ln1Beta,
      b.Wq, b.Bq, b.Wk, b.Bk, b.Wv, b.Bv, b.Wo, b.Bo,
      b.ln2Gamma, b.ln2Beta,
      b.ff1W, b.ff1B, b.ff2W, b.ff2B,
    );
  }
  return params;
}

/** Count total parameters. */
export function paramCount(model: ModelWeights): number {
  return allParams(model).reduce((sum, t) => sum + t.length, 0);
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

// Pre-compute causal mask (reused across forward calls)
let cachedMask: Tensor | null = null;
let cachedMaskSize = 0;

function getMask(size: number): Tensor {
  if (cachedMask && cachedMaskSize >= size) {
    return cachedMaskSize === size ? cachedMask : sliceMask(cachedMask, size);
  }
  cachedMask = causalMask(size);
  cachedMaskSize = size;
  return cachedMask;
}

function sliceMask(mask: Tensor, size: number): Tensor {
  const out = new Tensor(size, size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      out.data[i * size + j] = mask.data[i * mask.cols + j];
    }
  }
  return out;
}

export function forward(tokens: number[], model: ModelWeights): ForwardState {
  const { dModel, nHeads } = model.config;
  const seqLen = tokens.length;
  const headDim = dModel / nHeads;
  const mask = getMask(seqLen);

  // 1. Token embedding + positional embedding
  const tokenEmbOut = embedding(tokens, model.tokenEmb);
  const posEmbSlice = sliceRows(model.posEmb, 0, seqLen);
  let x = add(tokenEmbOut, posEmbSlice);
  const embSum = x;

  // 2. Transformer blocks
  const blockStates: BlockForwardState[] = [];
  const allAttnWeights: Tensor[][] = [];

  for (let layer = 0; layer < model.blocks.length; layer++) {
    const b = model.blocks[layer];

    // Pre-norm 1
    const ln1In = x;
    const ln1 = layerNorm(x, b.ln1Gamma, b.ln1Beta);

    // Multi-head attention
    const Q = add(matmul(ln1.out, b.Wq), b.Bq);  // [seqLen, dModel]
    const K = add(matmul(ln1.out, b.Wk), b.Bk);
    const V = add(matmul(ln1.out, b.Wv), b.Bv);

    // Per-head attention
    const headOutputs: Tensor[] = [];
    const headAttnWeights: Tensor[] = [];

    for (let h = 0; h < nHeads; h++) {
      // Extract head slice: columns [h*headDim, (h+1)*headDim)
      const qH = extractHead(Q, h, headDim);
      const kH = extractHead(K, h, headDim);
      const vH = extractHead(V, h, headDim);

      // Scaled dot-product attention
      const scores = matmul(qH, transpose(kH)); // [seqLen, seqLen]
      const scaled = scale(scores, 1 / Math.sqrt(headDim));

      // Apply causal mask
      const masked = add(scaled, mask);

      // Softmax
      const attnW = softmax(masked);
      headAttnWeights.push(attnW);

      // Weighted sum of values
      const headOut = matmul(attnW, vH); // [seqLen, headDim]
      headOutputs.push(headOut);
    }

    allAttnWeights.push(headAttnWeights);

    // Concatenate heads → [seqLen, dModel]
    const concat = concatHeads(headOutputs, seqLen, dModel);

    // Output projection
    const attnOut = add(matmul(concat, b.Wo), b.Bo);

    // Residual connection
    const resid1 = add(x, attnOut);

    // Pre-norm 2
    const ln2In = resid1;
    const ln2 = layerNorm(resid1, b.ln2Gamma, b.ln2Beta);

    // Feedforward: linear → ReLU → linear
    const ff1Out = add(matmul(ln2.out, b.ff1W), b.ff1B);
    const ff1Relu = relu(ff1Out);
    const ff2Out = add(matmul(ff1Relu, b.ff2W), b.ff2B);

    // Residual connection
    const resid2 = add(resid1, ff2Out);

    x = resid2;

    blockStates.push({
      ln1In,
      ln1Out: ln1.out,
      ln1Mean: ln1.mean,
      ln1InvStd: ln1.invStd,
      Q, K, V,
      attnScores: headAttnWeights,
      attnOut,
      headOutputs,
      resid1,
      ln2In,
      ln2Out: ln2.out,
      ln2Mean: ln2.mean,
      ln2InvStd: ln2.invStd,
      ff1Out,
      ff1Relu,
      ff2Out,
      resid2,
    });
  }

  // 3. Final layer norm + output projection
  const lnFIn = x;
  const lnF = layerNorm(x, model.lnFGamma, model.lnFBeta);
  const logits = add(matmul(lnF.out, model.outputW), model.outputB);

  return {
    tokens,
    tokenEmbOut,
    posEmbSlice,
    embSum,
    blockStates,
    lnFIn,
    lnFOut: lnF.out,
    lnFMean: lnF.mean,
    lnFInvStd: lnF.invStd,
    logits,
    attentionWeights: allAttnWeights,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Extract columns for one attention head from a [seqLen, dModel] tensor. */
function extractHead(A: Tensor, headIdx: number, headDim: number): Tensor {
  const out = new Tensor(A.rows, headDim);
  const startCol = headIdx * headDim;
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < headDim; j++) {
      out.data[i * headDim + j] = A.data[i * A.cols + startCol + j];
    }
  }
  return out;
}

/** Concatenate per-head outputs [seqLen, headDim] → [seqLen, dModel]. */
function concatHeads(heads: Tensor[], seqLen: number, dModel: number): Tensor {
  const out = new Tensor(seqLen, dModel);
  const headDim = heads[0].cols;
  for (let h = 0; h < heads.length; h++) {
    const startCol = h * headDim;
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < headDim; j++) {
        out.data[i * dModel + startCol + j] = heads[h].data[i * headDim + j];
      }
    }
  }
  return out;
}
