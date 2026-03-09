/**
 * Integration tests for the Signet LLM engine.
 * Verifies: tensor ops, gradients, training convergence, generation quality.
 */

import { Tensor, matmul, add, softmax, relu, layerNorm, crossEntropyLoss, crossEntropyBackward, softmaxBackward, reluBackward, layerNormBackward, matmulBackward, embedding, embeddingBackward, transpose, scale, causalMask } from './src/engine/tensor';
import { initModel, forward, paramCount, ModelConfig, allParams } from './src/engine/model';
import { initAdam, trainStep, ADAM_DEFAULTS } from './src/engine/train';
import { CharTokeniser } from './src/engine/tokeniser';
import { generate, generateWithMetadata } from './src/engine/generate';

let passed = 0;
let failed = 0;

function assert(condition: boolean, msg: string): void {
  if (condition) {
    passed++;
    console.log(`  PASS: ${msg}`);
  } else {
    failed++;
    console.log(`  FAIL: ${msg}`);
  }
}

function assertClose(a: number, b: number, eps: number, msg: string): void {
  const diff = Math.abs(a - b);
  if (diff < eps) {
    passed++;
    console.log(`  PASS: ${msg} (${a.toFixed(6)} ≈ ${b.toFixed(6)})`);
  } else {
    failed++;
    console.log(`  FAIL: ${msg} (${a.toFixed(6)} ≠ ${b.toFixed(6)}, diff=${diff.toFixed(6)})`);
  }
}

// =========================================================================
// 1. TENSOR OPERATIONS
// =========================================================================

console.log('\n=== 1. Tensor Operations ===\n');

// Matmul
{
  const A = new Tensor(2, 3, new Float32Array([1,2,3, 4,5,6]));
  const B = new Tensor(3, 2, new Float32Array([7,8, 9,10, 11,12]));
  const C = matmul(A, B);
  assertClose(C.at(0,0), 58, 0.01, 'matmul [0,0] = 1*7+2*9+3*11 = 58');
  assertClose(C.at(0,1), 64, 0.01, 'matmul [0,1] = 1*8+2*10+3*12 = 64');
  assertClose(C.at(1,0), 139, 0.01, 'matmul [1,0] = 4*7+5*9+6*11 = 139');
  assertClose(C.at(1,1), 154, 0.01, 'matmul [1,1] = 4*8+5*10+6*12 = 154');
}

// Add with broadcast
{
  const A = new Tensor(2, 3, new Float32Array([1,2,3, 4,5,6]));
  const B = new Tensor(1, 3, new Float32Array([10,20,30]));
  const C = add(A, B);
  assertClose(C.at(0,0), 11, 0.01, 'add broadcast [0,0]');
  assertClose(C.at(1,2), 36, 0.01, 'add broadcast [1,2]');
}

// Softmax
{
  const A = new Tensor(1, 3, new Float32Array([1, 2, 3]));
  const S = softmax(A);
  const sum = S.data[0] + S.data[1] + S.data[2];
  assertClose(sum, 1.0, 0.001, 'softmax sums to 1');
  assert(S.data[2] > S.data[1] && S.data[1] > S.data[0], 'softmax preserves order');
}

// ReLU
{
  const A = new Tensor(1, 4, new Float32Array([-2, -1, 0, 3]));
  const R = relu(A);
  assertClose(R.data[0], 0, 0.01, 'relu(-2) = 0');
  assertClose(R.data[3], 3, 0.01, 'relu(3) = 3');
}

// LayerNorm
{
  const A = new Tensor(1, 4, new Float32Array([1, 2, 3, 4]));
  const gamma = Tensor.ones(1, 4);
  const beta = Tensor.zeros(1, 4);
  const { out, mean, invStd } = layerNorm(A, gamma, beta);
  assertClose(mean[0], 2.5, 0.01, 'layerNorm mean = 2.5');
  // After norm, mean should be ~0
  let outMean = 0;
  for (let j = 0; j < 4; j++) outMean += out.data[j];
  outMean /= 4;
  assertClose(outMean, 0, 0.01, 'layerNorm output mean ≈ 0');
}

// Causal mask
{
  const mask = causalMask(3);
  assertClose(mask.at(0,0), 0, 0.01, 'causal mask [0,0] = 0 (can attend)');
  assert(mask.at(0,1) === -Infinity, 'causal mask [0,1] = -Inf (cannot attend to future)');
  assertClose(mask.at(2,0), 0, 0.01, 'causal mask [2,0] = 0 (can attend to past)');
  assertClose(mask.at(2,2), 0, 0.01, 'causal mask [2,2] = 0 (can attend to self)');
}

// Transpose
{
  const A = new Tensor(2, 3, new Float32Array([1,2,3, 4,5,6]));
  const T = transpose(A);
  assert(T.rows === 3 && T.cols === 2, 'transpose shape');
  assertClose(T.at(1,0), 2, 0.01, 'transpose [1,0] = original [0,1]');
  assertClose(T.at(2,1), 6, 0.01, 'transpose [2,1] = original [1,2]');
}

// =========================================================================
// 2. GRADIENT CHECKING (numerical vs analytical)
// =========================================================================

console.log('\n=== 2. Gradient Checking ===\n');

function numericalGradient(
  fn: (x: Float32Array) => number,
  x: Float32Array,
  eps = 1e-4
): Float32Array {
  const grad = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const orig = x[i];
    x[i] = orig + eps;
    const fPlus = fn(x);
    x[i] = orig - eps;
    const fMinus = fn(x);
    x[i] = orig;
    grad[i] = (fPlus - fMinus) / (2 * eps);
  }
  return grad;
}

function maxRelError(a: Float32Array, b: Float32Array): number {
  let maxErr = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(Math.abs(a[i]), Math.abs(b[i]), 1e-8);
    const err = Math.abs(a[i] - b[i]) / denom;
    if (err > maxErr) maxErr = err;
  }
  return maxErr;
}

// Gradient check: cross-entropy
{
  const logitData = new Float32Array([0.5, 1.2, -0.3, 0.8, 0.1, -0.5]);
  const targets = [1, 2]; // 2 positions, 3 classes

  const fn = (data: Float32Array) => {
    const t = new Tensor(2, 3, data);
    return crossEntropyLoss(t, targets);
  };

  const numGrad = numericalGradient(fn, logitData);
  const logits = new Tensor(2, 3, new Float32Array(logitData));
  const analyticalTensor = crossEntropyBackward(logits, targets);

  const err = maxRelError(numGrad, analyticalTensor.data);
  assert(err < 0.01, `cross-entropy gradient check (max relative error: ${err.toExponential(2)})`);
}

// Gradient check: softmax
{
  const inputData = new Float32Array([1, 2, 3, 0.5, 1.5, 2.5]);
  const input = new Tensor(2, 3, new Float32Array(inputData));
  const sm = softmax(input);

  // Upstream gradient
  const dOut = new Tensor(2, 3, new Float32Array([0.1, -0.2, 0.3, -0.1, 0.2, -0.1]));
  const dInput = softmaxBackward(dOut, sm);

  // Numerical check for each element
  let maxErr = 0;
  for (let i = 0; i < inputData.length; i++) {
    const fn = (data: Float32Array) => {
      const t = new Tensor(2, 3, data);
      const s = softmax(t);
      let loss = 0;
      for (let j = 0; j < s.length; j++) loss += s.data[j] * dOut.data[j];
      return loss;
    };
    const numGrad = numericalGradient(fn, new Float32Array(inputData));
    const err = Math.abs(numGrad[i] - dInput.data[i]) / Math.max(Math.abs(numGrad[i]), 1e-8);
    if (err > maxErr) maxErr = err;
  }
  assert(maxErr < 0.01, `softmax gradient check (max relative error: ${maxErr.toExponential(2)})`);
}

// Gradient check: ReLU
{
  const inputData = new Float32Array([-1, 0.5, -0.3, 2]);
  const input = new Tensor(1, 4, new Float32Array(inputData));
  const dOut = new Tensor(1, 4, new Float32Array([1, 1, 1, 1]));
  const dInput = reluBackward(dOut, input);

  assertClose(dInput.data[0], 0, 0.01, 'relu grad for negative input = 0');
  assertClose(dInput.data[1], 1, 0.01, 'relu grad for positive input = 1');
  assertClose(dInput.data[3], 1, 0.01, 'relu grad for positive input = 1');
}

// Gradient check: layerNorm
{
  const inputData = new Float32Array([1, 2, 3, 4]);
  // Use varying gamma so the loss (sum of outputs) isn't trivially constant
  const gammaData = new Float32Array([0.5, 1.0, 1.5, 2.0]);
  const gamma = new Tensor(1, 4, new Float32Array(gammaData));
  const beta = Tensor.zeros(1, 4);

  const fn = (data: Float32Array) => {
    const t = new Tensor(1, 4, data);
    const g = new Tensor(1, 4, new Float32Array(gammaData));
    const b = new Tensor(1, 4, new Float32Array([0, 0, 0, 0]));
    const { out } = layerNorm(t, g, b);
    // Use sum-of-squares to avoid degenerate constant loss
    let loss = 0;
    for (let j = 0; j < out.length; j++) loss += out.data[j] * out.data[j];
    return loss;
  };

  const numGrad = numericalGradient(fn, new Float32Array(inputData));

  const input = new Tensor(1, 4, new Float32Array(inputData));
  const { out, mean: m, invStd: inv } = layerNorm(input, gamma, beta);
  // dLoss/dOut = 2 * out (derivative of sum-of-squares)
  const dOut = new Tensor(1, 4);
  for (let j = 0; j < 4; j++) dOut.data[j] = 2 * out.data[j];
  const dInput = layerNormBackward(dOut, input, gamma, beta, m, inv);

  const err = maxRelError(numGrad, dInput.data);
  assert(err < 0.01, `layerNorm gradient check (max relative error: ${err.toExponential(2)})`);
}

// Gradient check: matmul
{
  const aData = new Float32Array([1, 2, 3, 4, 5, 6]);
  const bData = new Float32Array([7, 8, 9, 10, 11, 12]);

  // Check gradient w.r.t. A
  const fnA = (data: Float32Array) => {
    const A = new Tensor(2, 3, data);
    const B = new Tensor(3, 2, new Float32Array(bData));
    const C = matmul(A, B);
    let loss = 0;
    for (let j = 0; j < C.length; j++) loss += C.data[j];
    return loss;
  };

  const numGradA = numericalGradient(fnA, new Float32Array(aData));

  const A = new Tensor(2, 3, new Float32Array(aData));
  const B = new Tensor(3, 2, new Float32Array(bData));
  const dC = Tensor.ones(2, 2);
  matmulBackward(dC, A, B);

  const errA = maxRelError(numGradA, A.grad);
  assert(errA < 0.01, `matmul gradient check for A (max relative error: ${errA.toExponential(2)})`);

  // Check gradient w.r.t. B
  const fnB = (data: Float32Array) => {
    const A2 = new Tensor(2, 3, new Float32Array(aData));
    const B2 = new Tensor(3, 2, data);
    const C = matmul(A2, B2);
    let loss = 0;
    for (let j = 0; j < C.length; j++) loss += C.data[j];
    return loss;
  };

  const numGradB = numericalGradient(fnB, new Float32Array(bData));
  const errB = maxRelError(numGradB, B.grad);
  assert(errB < 0.01, `matmul gradient check for B (max relative error: ${errB.toExponential(2)})`);
}

// =========================================================================
// 3. TOKENISER
// =========================================================================

console.log('\n=== 3. Tokeniser ===\n');

{
  const text = "hello world";
  const tok = new CharTokeniser(text);
  assert(tok.vocabSize === 8, `vocab size = 8 for "hello world" (got ${tok.vocabSize})`);

  const encoded = tok.encode("hello");
  const decoded = tok.decode(encoded);
  assert(decoded === "hello", `roundtrip encode/decode: "${decoded}"`);

  const encoded2 = tok.encode(text);
  assert(encoded2.length === text.length, `encoded length = text length (${encoded2.length})`);
}

// =========================================================================
// 4. FORWARD PASS
// =========================================================================

console.log('\n=== 4. Forward Pass ===\n');

{
  const text = "abcabcabcabc";
  const tok = new CharTokeniser(text);
  const config: ModelConfig = {
    vocabSize: tok.vocabSize,
    blockSize: 8,
    dModel: 16,
    nHeads: 2,
    nLayers: 1,
  };
  const model = initModel(config);
  const tokens = tok.encode("abcabc").slice(0, 6);

  const state = forward(tokens, model);

  assert(state.logits.rows === 6, `logits rows = seqLen (${state.logits.rows})`);
  assert(state.logits.cols === tok.vocabSize, `logits cols = vocabSize (${state.logits.cols})`);

  // Attention weights should exist
  assert(state.attentionWeights.length === 1, 'attention weights: 1 layer');
  assert(state.attentionWeights[0].length === 2, 'attention weights: 2 heads');

  // Attention weights per head should be [seqLen, seqLen]
  const attnW = state.attentionWeights[0][0];
  assert(attnW.rows === 6 && attnW.cols === 6, `attention shape = [6,6] (got [${attnW.rows},${attnW.cols}])`);

  // Each row of attention weights should sum to 1
  let sumOk = true;
  for (let i = 0; i < attnW.rows; i++) {
    let rowSum = 0;
    for (let j = 0; j < attnW.cols; j++) rowSum += attnW.at(i, j);
    if (Math.abs(rowSum - 1.0) > 0.01) sumOk = false;
  }
  assert(sumOk, 'attention weights: each row sums to 1');

  // Causal: no attention to future positions
  let causalOk = true;
  for (let i = 0; i < attnW.rows; i++) {
    for (let j = i + 1; j < attnW.cols; j++) {
      if (attnW.at(i, j) > 0.001) causalOk = false;
    }
  }
  assert(causalOk, 'attention weights: causal (no future attention)');
}

// =========================================================================
// 5. TRAINING CONVERGENCE
// =========================================================================

console.log('\n=== 5. Training Convergence ===\n');

{
  // Simple repeating pattern — should converge quickly
  const text = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
  const tok = new CharTokeniser(text);
  const encoded = tok.encode(text);

  const config: ModelConfig = {
    vocabSize: tok.vocabSize,
    blockSize: 8,
    dModel: 16,
    nHeads: 2,
    nLayers: 1,
  };

  const model = initModel(config);
  const adam = initAdam(model);

  let cursor = 0;
  const losses: number[] = [];

  for (let s = 0; s < 300; s++) {
    if (cursor + config.blockSize + 1 > encoded.length) cursor = 0;
    const result = trainStep(model, adam, { ...ADAM_DEFAULTS }, encoded, cursor);
    cursor = (cursor + config.blockSize) % Math.max(encoded.length - config.blockSize - 1, 1);
    losses.push(result.loss);
  }

  const firstLoss = losses[0];
  const lastLoss = losses[losses.length - 1];

  console.log(`    Initial loss: ${firstLoss.toFixed(4)}`);
  console.log(`    Final loss:   ${lastLoss.toFixed(4)}`);

  assert(lastLoss < firstLoss, `loss decreased: ${firstLoss.toFixed(4)} → ${lastLoss.toFixed(4)}`);
  assert(lastLoss < 0.5, `loss < 0.5 after 300 steps on "abc" repeat (got ${lastLoss.toFixed(4)})`);
}

// Shakespeare-like convergence
{
  const text = "To be or not to be, that is the question. To be or not to be, that is the question. To be or not to be, that is the question.";
  const tok = new CharTokeniser(text);
  const encoded = tok.encode(text);

  const config: ModelConfig = {
    vocabSize: tok.vocabSize,
    blockSize: 32,
    dModel: 32,
    nHeads: 2,
    nLayers: 2,
  };

  const model = initModel(config);
  const adam = initAdam(model);

  let cursor = 0;
  const losses: number[] = [];

  for (let s = 0; s < 500; s++) {
    if (cursor + config.blockSize + 1 > encoded.length) cursor = 0;
    const result = trainStep(model, adam, { ...ADAM_DEFAULTS }, encoded, cursor);
    cursor = (cursor + config.blockSize) % Math.max(encoded.length - config.blockSize - 1, 1);
    losses.push(result.loss);
  }

  const firstLoss = losses[0];
  const lastLoss = losses[losses.length - 1];

  console.log(`    Shakespeare initial: ${firstLoss.toFixed(4)}`);
  console.log(`    Shakespeare final:   ${lastLoss.toFixed(4)}`);

  assert(lastLoss < 1.0, `Shakespeare loss < 1.0 after 500 steps (got ${lastLoss.toFixed(4)})`);
  assert(lastLoss < firstLoss * 0.3, `Shakespeare loss dropped by >70%`);
}

// =========================================================================
// 6. GENERATION
// =========================================================================

console.log('\n=== 6. Generation ===\n');

{
  // Train on "abc" pattern then generate
  const text = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
  const tok = new CharTokeniser(text);
  const encoded = tok.encode(text);

  const config: ModelConfig = {
    vocabSize: tok.vocabSize,
    blockSize: 8,
    dModel: 16,
    nHeads: 2,
    nLayers: 1,
  };

  const model = initModel(config);
  const adam = initAdam(model);

  let cursor = 0;
  for (let s = 0; s < 500; s++) {
    if (cursor + config.blockSize + 1 > encoded.length) cursor = 0;
    trainStep(model, adam, { ...ADAM_DEFAULTS }, encoded, cursor);
    cursor = (cursor + config.blockSize) % Math.max(encoded.length - config.blockSize - 1, 1);
  }

  // Generate with low temperature (should be deterministic-ish)
  const seed = tok.encode("abc");
  const generated = generate(model, seed, 30, 0.3);
  const genText = tok.decode(generated);

  console.log(`    Generated (temp=0.3): "${genText}"`);
  assert(generated.length === seed.length + 30, `generation produces exactly 30 new tokens (total ${generated.length})`);

  // Check if the pattern is roughly "abcabc..."
  const genPart = genText.slice(3); // skip seed
  let abcCount = 0;
  for (let i = 0; i < genPart.length; i++) {
    if (genPart[i] === "abc"[i % 3]) abcCount++;
  }
  const accuracy = abcCount / genPart.length;
  console.log(`    Pattern accuracy: ${(accuracy * 100).toFixed(0)}%`);
  assert(accuracy > 0.7, `"abc" pattern accuracy > 70% (got ${(accuracy * 100).toFixed(0)}%)`);
}

// Generation with metadata (for X-Ray mode)
{
  const text = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
  const tok = new CharTokeniser(text);
  const encoded = tok.encode(text);

  const config: ModelConfig = {
    vocabSize: tok.vocabSize,
    blockSize: 8,
    dModel: 16,
    nHeads: 2,
    nLayers: 1,
  };

  const model = initModel(config);
  const adam = initAdam(model);

  let cursor = 0;
  for (let s = 0; s < 300; s++) {
    if (cursor + config.blockSize + 1 > encoded.length) cursor = 0;
    trainStep(model, adam, { ...ADAM_DEFAULTS }, encoded, cursor);
    cursor = (cursor + config.blockSize) % Math.max(encoded.length - config.blockSize - 1, 1);
  }

  const seed = tok.encode("abc");
  const result = generateWithMetadata(model, tok, seed, 10, 0.5);

  assert(result.tokens.length === seed.length + 10, `generateWithMetadata: correct token count`);
  assert(result.meta.length === 10, `generateWithMetadata: 10 metadata entries (got ${result.meta.length})`);

  // Each meta should have confidence, topK, attentionWeights
  const m = result.meta[0];
  assert(m.confidence >= 0 && m.confidence <= 1, `meta confidence in [0,1]: ${m.confidence.toFixed(3)}`);
  assert(m.topK.length === Math.min(5, config.vocabSize), `meta topK has ${Math.min(5, config.vocabSize)} entries (vocabSize=${config.vocabSize})`);
  assert(m.attentionWeights.length === config.nLayers, `meta attentionWeights has ${config.nLayers} layers`);

  // topK should be sorted by probability descending
  let sorted = true;
  for (let i = 1; i < m.topK.length; i++) {
    if (m.topK[i].prob > m.topK[i-1].prob) sorted = false;
  }
  assert(sorted, 'meta topK sorted by probability descending');
}

// =========================================================================
// 7. GENERATION LENGTH STABILITY
// =========================================================================

console.log('\n=== 7. Generation Length Stability ===\n');

{
  const text = "To be or not to be, that is the question. To be or not to be, that is the question. To be or not to be, that is the question.";
  const tok = new CharTokeniser(text);
  const encoded = tok.encode(text);

  const config: ModelConfig = {
    vocabSize: tok.vocabSize,
    blockSize: 32,
    dModel: 32,
    nHeads: 2,
    nLayers: 2,
  };

  const model = initModel(config);
  const adam = initAdam(model);

  // Train for a while
  let cursor = 0;
  for (let s = 0; s < 1000; s++) {
    if (cursor + config.blockSize + 1 > encoded.length) cursor = 0;
    trainStep(model, adam, { ...ADAM_DEFAULTS }, encoded, cursor);
    cursor = (cursor + config.blockSize) % Math.max(encoded.length - config.blockSize - 1, 1);
  }

  // Generate multiple times at different temperatures
  const temps = [0.3, 0.5, 0.8, 1.0, 1.5];
  for (const temp of temps) {
    const seed = encoded.slice(0, 8);
    const genTokens = generate(model, seed, 150, temp);
    const genText = tok.decode(genTokens.slice(8));

    assert(genTokens.length === 158, `temp=${temp}: generated exactly 150 tokens (total=${genTokens.length})`);
    console.log(`    temp=${temp}: "${genText.slice(0, 60)}..."`);
  }

  // Generate with metadata — also check length stability
  const seed = encoded.slice(0, 8);
  const metaResult = generateWithMetadata(model, tok, seed, 150, 0.8);
  assert(metaResult.tokens.length === 158, `generateWithMetadata: 150 new tokens`);
  assert(metaResult.meta.length === 150, `generateWithMetadata: 150 meta entries (got ${metaResult.meta.length})`);

  // All confidences should be valid numbers
  let allValid = true;
  for (const m of metaResult.meta) {
    if (isNaN(m.confidence) || m.confidence < 0 || m.confidence > 1) {
      allValid = false;
      console.log(`    Invalid confidence: ${m.confidence}`);
    }
  }
  assert(allValid, 'all generation confidences are valid numbers in [0,1]');
}

// =========================================================================
// 8. PARAMETER COUNT
// =========================================================================

console.log('\n=== 8. Parameter Count ===\n');

{
  const config: ModelConfig = {
    vocabSize: 65,
    blockSize: 64,
    dModel: 32,
    nHeads: 2,
    nLayers: 2,
  };
  const model = initModel(config);
  const count = paramCount(model);

  // Expected from techspec:
  // tokenEmb: 65*32 = 2080
  // posEmb: 64*32 = 2048
  // Per block: Q,K,V,O each 32*32+32=1056, ln1 64, ff1 32*128+128=4224, ff2 128*32+32=4128, ln2 64
  // Block total: 4*1056 + 64 + 4224 + 4128 + 64 = 12704
  // 2 blocks = 25408
  // Final LN: 64, Output: 32*65+65 = 2145
  // Total: 2080 + 2048 + 25408 + 64 + 2145 = 31745

  console.log(`    Parameter count: ${count}`);
  assert(count > 20000, `param count > 20K (got ${count})`);
  assert(count < 40000, `param count < 40K (got ${count})`);

  // Verify all params have matching grad arrays
  const params = allParams(model);
  let gradOk = true;
  for (const p of params) {
    if (p.grad.length !== p.data.length) gradOk = false;
  }
  assert(gradOk, 'all params have grad arrays of correct size');
}

// =========================================================================
// SUMMARY
// =========================================================================

console.log(`\n${'='.repeat(50)}`);
console.log(`RESULTS: ${passed} passed, ${failed} failed`);
console.log(`${'='.repeat(50)}\n`);

process.exit(failed > 0 ? 1 : 0);
