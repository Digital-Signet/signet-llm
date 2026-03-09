/**
 * Backward pass, Adam optimiser, and training step.
 * Manual backpropagation — no autograd.
 */

import {
  Tensor,
  matmul,
  matmulBackward,
  add,
  transpose,
  scale,
  softmax,
  softmaxBackward,
  reluBackward,
  layerNormBackward,
  crossEntropyLoss,
  crossEntropyBackward,
  embeddingBackward,
} from './tensor';
import {
  ModelWeights,
  ForwardState,
  BlockForwardState,
  allParams,
  forward,
} from './model';

// ---------------------------------------------------------------------------
// Adam optimiser
// ---------------------------------------------------------------------------

export interface AdamState {
  m: Float32Array[];   // first moments, one per parameter
  v: Float32Array[];   // second moments
  t: number;           // timestep
}

export interface AdamConfig {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
}

export const ADAM_DEFAULTS: AdamConfig = {
  lr: 3e-4,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
};

export function initAdam(model: ModelWeights): AdamState {
  const params = allParams(model);
  return {
    m: params.map(p => new Float32Array(p.length)),
    v: params.map(p => new Float32Array(p.length)),
    t: 0,
  };
}

export function adamStep(
  model: ModelWeights,
  state: AdamState,
  config: AdamConfig
): void {
  state.t++;
  const params = allParams(model);
  const { lr, beta1, beta2, eps } = config;
  const bc1 = 1 - Math.pow(beta1, state.t);
  const bc2 = 1 - Math.pow(beta2, state.t);

  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    const m = state.m[i];
    const v = state.v[i];

    for (let j = 0; j < p.length; j++) {
      const g = p.grad[j];
      m[j] = beta1 * m[j] + (1 - beta1) * g;
      v[j] = beta2 * v[j] + (1 - beta2) * g * g;
      const mHat = m[j] / bc1;
      const vHat = v[j] / bc2;
      p.data[j] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

function backward(state: ForwardState, targets: number[], model: ModelWeights): void {
  const { dModel, nHeads } = model.config;
  const headDim = dModel / nHeads;
  const seqLen = state.tokens.length;

  // 1. Loss gradient → dLogits
  const dLogits = crossEntropyBackward(state.logits, targets);

  // 2. Output projection backward: logits = lnFOut @ outputW + outputB
  //    dLogits → dLnFOut, accumulate outputW.grad, outputB.grad
  const dOutputProj = new Tensor(dLogits.rows, dLogits.cols);
  dOutputProj.data.set(dLogits.data);

  // outputB grad (broadcast add backward — sum over rows)
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < model.outputB.cols; j++) {
      model.outputB.grad[j] += dOutputProj.data[i * dOutputProj.cols + j];
    }
  }

  // matmul backward: logits = lnFOut @ outputW
  const dLnFOut = new Tensor(seqLen, dModel);
  matmulBackwardInto(dOutputProj, state.lnFOut, model.outputW, dLnFOut);

  // 3. Final layer norm backward
  const dLnFIn = layerNormBackward(
    dLnFOut, state.lnFIn, model.lnFGamma, model.lnFBeta,
    state.lnFMean, state.lnFInvStd
  );

  // 4. Transformer blocks backward (reverse order)
  let dx = dLnFIn;

  for (let layer = model.blocks.length - 1; layer >= 0; layer--) {
    const b = model.blocks[layer];
    const bs = state.blockStates[layer];

    // -- Residual 2 backward: resid2 = resid1 + ff2Out
    // dx splits into dResid1 and dFF2Out
    const dResid1FromFF = new Tensor(seqLen, dModel);
    const dFF2Out = new Tensor(seqLen, dModel);
    for (let i = 0; i < dx.length; i++) {
      dResid1FromFF.data[i] = dx.data[i];
      dFF2Out.data[i] = dx.data[i];
    }

    // -- FFN backward: ff2Out = ff1Relu @ ff2W + ff2B
    // ff2B grad
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < dModel; j++) {
        b.ff2B.grad[j] += dFF2Out.data[i * dModel + j];
      }
    }

    const dFF1Relu = new Tensor(seqLen, b.ff1W.cols);
    matmulBackwardInto(dFF2Out, bs.ff1Relu, b.ff2W, dFF1Relu);

    // ReLU backward
    const dFF1Out = reluBackward(dFF1Relu, bs.ff1Out);

    // ff1B grad
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < b.ff1B.cols; j++) {
        b.ff1B.grad[j] += dFF1Out.data[i * b.ff1B.cols + j];
      }
    }

    const dLn2Out = new Tensor(seqLen, dModel);
    matmulBackwardInto(dFF1Out, bs.ln2Out, b.ff1W, dLn2Out);

    // LayerNorm 2 backward
    const dLn2In = layerNormBackward(
      dLn2Out, bs.ln2In, b.ln2Gamma, b.ln2Beta,
      bs.ln2Mean, bs.ln2InvStd
    );

    // Add residual from FFN path
    const dResid1 = new Tensor(seqLen, dModel);
    for (let i = 0; i < dx.length; i++) {
      dResid1.data[i] = dResid1FromFF.data[i] + dLn2In.data[i];
    }

    // -- Residual 1 backward: resid1 = x + attnOut
    const dXFromAttn = new Tensor(seqLen, dModel);
    const dAttnOut = new Tensor(seqLen, dModel);
    for (let i = 0; i < dResid1.length; i++) {
      dXFromAttn.data[i] = dResid1.data[i];
      dAttnOut.data[i] = dResid1.data[i];
    }

    // -- Attention output projection backward: attnOut = concat @ Wo + Bo
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < dModel; j++) {
        b.Bo.grad[j] += dAttnOut.data[i * dModel + j];
      }
    }

    // We need concat for the matmul backward
    const concat = concatHeadsFromState(bs.headOutputs, seqLen, dModel);
    const dConcat = new Tensor(seqLen, dModel);
    matmulBackwardInto(dAttnOut, concat, b.Wo, dConcat);

    // -- Per-head attention backward
    const dQ = Tensor.zeros(seqLen, dModel);
    const dK = Tensor.zeros(seqLen, dModel);
    const dV = Tensor.zeros(seqLen, dModel);

    for (let h = 0; h < nHeads; h++) {
      const startCol = h * headDim;

      // Extract dConcat for this head
      const dHeadOut = extractHeadGrad(dConcat, h, headDim);

      // attnWeights = softmax(scaled_scores)
      // headOut = attnWeights @ vH
      const vH = extractHead(bs.V, h, headDim);
      const attnW = bs.attnScores[h];

      // dAttnW = dHeadOut @ vH^T
      const dAttnW = matmul(dHeadOut, transpose(vH));

      // dVH = attnW^T @ dHeadOut
      const dVH = matmul(transpose(attnW), dHeadOut);

      // Softmax backward
      const dScaled = softmaxBackward(dAttnW, attnW);

      // Scale backward: scaled = scores / sqrt(headDim)
      const dScores = scaleInPlace(dScaled, 1 / Math.sqrt(headDim));

      // scores = qH @ kH^T → dQH = dScores @ kH, dKH = dScores^T @ qH
      const qH = extractHead(bs.Q, h, headDim);
      const kH = extractHead(bs.K, h, headDim);

      const dQH = matmul(dScores, kH);
      const dKH = matmul(transpose(dScores), qH);

      // Scatter back into dQ, dK, dV
      scatterHead(dQH, dQ, h, headDim);
      scatterHead(dKH, dK, h, headDim);
      scatterHead(dVH, dV, h, headDim);
    }

    // Q = ln1Out @ Wq + Bq → backward
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < dModel; j++) {
        b.Bq.grad[j] += dQ.data[i * dModel + j];
        b.Bk.grad[j] += dK.data[i * dModel + j];
        b.Bv.grad[j] += dV.data[i * dModel + j];
      }
    }

    const dLn1OutFromQ = new Tensor(seqLen, dModel);
    const dLn1OutFromK = new Tensor(seqLen, dModel);
    const dLn1OutFromV = new Tensor(seqLen, dModel);

    matmulBackwardInto(dQ, bs.ln1Out, b.Wq, dLn1OutFromQ);
    matmulBackwardInto(dK, bs.ln1Out, b.Wk, dLn1OutFromK);
    matmulBackwardInto(dV, bs.ln1Out, b.Wv, dLn1OutFromV);

    // Sum gradients flowing into ln1Out
    const dLn1Out = new Tensor(seqLen, dModel);
    for (let i = 0; i < dLn1Out.length; i++) {
      dLn1Out.data[i] = dLn1OutFromQ.data[i] + dLn1OutFromK.data[i] + dLn1OutFromV.data[i];
    }

    // LayerNorm 1 backward
    const dLn1In = layerNormBackward(
      dLn1Out, bs.ln1In, b.ln1Gamma, b.ln1Beta,
      bs.ln1Mean, bs.ln1InvStd
    );

    // Combine residual paths
    dx = new Tensor(seqLen, dModel);
    for (let i = 0; i < dx.length; i++) {
      dx.data[i] = dXFromAttn.data[i] + dLn1In.data[i];
    }
  }

  // 5. Embedding backward
  // dx flows into both tokenEmb and posEmb
  embeddingBackward(dx, state.tokens, model.tokenEmb);

  // posEmb gradient: scatter-add dx into position rows
  for (let i = 0; i < seqLen; i++) {
    const srcOffset = i * dModel;
    const dstOffset = i * dModel;
    for (let j = 0; j < dModel; j++) {
      model.posEmb.grad[dstOffset + j] += dx.data[srcOffset + j];
    }
  }
}

// ---------------------------------------------------------------------------
// Training step
// ---------------------------------------------------------------------------

export interface TrainStepResult {
  loss: number;
  attentionWeights: Tensor[][];
}

export function trainStep(
  model: ModelWeights,
  adam: AdamState,
  adamConfig: AdamConfig,
  data: number[],
  batchStart: number
): TrainStepResult {
  const blockSize = model.config.blockSize;

  // Slice input and target sequences
  const input = data.slice(batchStart, batchStart + blockSize);
  const target = data.slice(batchStart + 1, batchStart + blockSize + 1);

  // Forward
  const state = forward(input, model);

  // Loss
  const loss = crossEntropyLoss(state.logits, target);

  // Zero gradients
  for (const p of allParams(model)) {
    p.zeroGrad();
  }

  // Backward
  backward(state, target, model);

  // Adam step
  adamStep(model, adam, adamConfig);

  return {
    loss,
    attentionWeights: state.attentionWeights,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** matmul backward that writes dA into a provided tensor (instead of accumulating into A.grad). */
function matmulBackwardInto(
  dC: Tensor, A: Tensor, B: Tensor, dA: Tensor
): void {
  const m = A.rows;
  const k = A.cols;
  const n = B.cols;

  // dA += dC @ B^T
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        sum += dC.data[i * n + j] * B.data[p * n + j];
      }
      dA.data[i * k + p] += sum;
    }
  }

  // dB accumulates into B.grad
  for (let p = 0; p < k; p++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let i = 0; i < m; i++) {
        sum += A.data[i * k + p] * dC.data[i * n + j];
      }
      B.grad[p * n + j] += sum;
    }
  }
}

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

function extractHeadGrad(dConcat: Tensor, headIdx: number, headDim: number): Tensor {
  return extractHead(dConcat, headIdx, headDim);
}

function scatterHead(src: Tensor, dst: Tensor, headIdx: number, headDim: number): void {
  const startCol = headIdx * headDim;
  for (let i = 0; i < src.rows; i++) {
    for (let j = 0; j < headDim; j++) {
      dst.data[i * dst.cols + startCol + j] += src.data[i * headDim + j];
    }
  }
}

function concatHeadsFromState(heads: Tensor[], seqLen: number, dModel: number): Tensor {
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

function scaleInPlace(A: Tensor, s: number): Tensor {
  for (let i = 0; i < A.length; i++) {
    A.data[i] *= s;
  }
  return A;
}
