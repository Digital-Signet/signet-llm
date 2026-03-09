/**
 * Minimal tensor library for the Signet browser LLM.
 *
 * Supports 1D and 2D Float32Arrays with manual gradient tracking.
 * No autograd — gradients are computed explicitly in the backward pass.
 */

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

export class Tensor {
  data: Float32Array;
  grad: Float32Array;
  rows: number;
  cols: number;

  constructor(rows: number, cols: number, data?: Float32Array) {
    this.rows = rows;
    this.cols = cols;
    const len = rows * cols;
    this.data = data ?? new Float32Array(len);
    this.grad = new Float32Array(len);
  }

  get length(): number {
    return this.rows * this.cols;
  }

  /** Get element at (r, c). */
  at(r: number, c: number): number {
    return this.data[r * this.cols + c];
  }

  /** Set element at (r, c). */
  set(r: number, c: number, v: number): void {
    this.data[r * this.cols + c] = v;
  }

  /** Zero out gradients. */
  zeroGrad(): void {
    this.grad.fill(0);
  }

  // ---- Factories ----------------------------------------------------------

  static zeros(rows: number, cols: number): Tensor {
    return new Tensor(rows, cols);
  }

  static ones(rows: number, cols: number): Tensor {
    const t = new Tensor(rows, cols);
    t.data.fill(1);
    return t;
  }

  /** Normal distribution with given std. Uses Box-Muller. */
  static randn(rows: number, cols: number, std = 1.0): Tensor {
    const t = new Tensor(rows, cols);
    for (let i = 0; i < t.length; i += 2) {
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      const r = std * Math.sqrt(-2 * Math.log(u1));
      t.data[i] = r * Math.cos(2 * Math.PI * u2);
      if (i + 1 < t.length) {
        t.data[i + 1] = r * Math.sin(2 * Math.PI * u2);
      }
    }
    return t;
  }

  /** Xavier / Glorot uniform initialisation. */
  static xavier(rows: number, cols: number): Tensor {
    const limit = Math.sqrt(6 / (rows + cols));
    const t = new Tensor(rows, cols);
    for (let i = 0; i < t.length; i++) {
      t.data[i] = (Math.random() * 2 - 1) * limit;
    }
    return t;
  }
}

// ---------------------------------------------------------------------------
// Operations — forward
// ---------------------------------------------------------------------------

/**
 * Matrix multiply: C = A @ B
 * A: [m, k]  B: [k, n]  → C: [m, n]
 */
export function matmul(A: Tensor, B: Tensor): Tensor {
  const m = A.rows;
  const k = A.cols;
  const n = B.cols;
  const C = new Tensor(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) {
        sum += A.data[i * k + p] * B.data[p * n + j];
      }
      C.data[i * n + j] = sum;
    }
  }
  return C;
}

/**
 * Element-wise add. Supports broadcasting B [1, cols] over A [rows, cols].
 */
export function add(A: Tensor, B: Tensor): Tensor {
  const C = new Tensor(A.rows, A.cols);
  if (B.rows === 1 && A.rows > 1) {
    // Broadcast B across rows
    for (let i = 0; i < A.rows; i++) {
      for (let j = 0; j < A.cols; j++) {
        C.data[i * A.cols + j] = A.data[i * A.cols + j] + B.data[j];
      }
    }
  } else {
    for (let i = 0; i < A.length; i++) {
      C.data[i] = A.data[i] + B.data[i];
    }
  }
  return C;
}

/**
 * Transpose: [m, n] → [n, m]
 */
export function transpose(A: Tensor): Tensor {
  const T = new Tensor(A.cols, A.rows);
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < A.cols; j++) {
      T.data[j * A.rows + i] = A.data[i * A.cols + j];
    }
  }
  return T;
}

/**
 * Scale every element by a scalar.
 */
export function scale(A: Tensor, s: number): Tensor {
  const C = new Tensor(A.rows, A.cols);
  for (let i = 0; i < A.length; i++) {
    C.data[i] = A.data[i] * s;
  }
  return C;
}

/**
 * Row-wise softmax. Each row is independently normalised.
 */
export function softmax(A: Tensor): Tensor {
  const C = new Tensor(A.rows, A.cols);
  for (let i = 0; i < A.rows; i++) {
    const offset = i * A.cols;

    // Max for numerical stability
    let max = -Infinity;
    for (let j = 0; j < A.cols; j++) {
      if (A.data[offset + j] > max) max = A.data[offset + j];
    }

    // Exponentiate and sum
    let sum = 0;
    for (let j = 0; j < A.cols; j++) {
      const e = Math.exp(A.data[offset + j] - max);
      C.data[offset + j] = e;
      sum += e;
    }

    // Normalise
    for (let j = 0; j < A.cols; j++) {
      C.data[offset + j] /= sum;
    }
  }
  return C;
}

/**
 * Softmax for a 1D array (convenience for generation).
 */
export function softmax1D(arr: Float32Array): Float32Array {
  const out = new Float32Array(arr.length);
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max) max = arr[i];
  }
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    out[i] = Math.exp(arr[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < arr.length; i++) {
    out[i] /= sum;
  }
  return out;
}

/**
 * ReLU activation: max(0, x).
 */
export function relu(A: Tensor): Tensor {
  const C = new Tensor(A.rows, A.cols);
  for (let i = 0; i < A.length; i++) {
    C.data[i] = A.data[i] > 0 ? A.data[i] : 0;
  }
  return C;
}

/**
 * Layer normalisation over the last dimension (columns).
 * For each row: normalise to mean=0 var=1, then scale by gamma and shift by beta.
 *
 * Returns { out, mean, invStd } — mean and invStd stored for backward pass.
 */
export function layerNorm(
  A: Tensor,
  gamma: Tensor,
  beta: Tensor,
  eps = 1e-5
): { out: Tensor; mean: Float32Array; invStd: Float32Array } {
  const out = new Tensor(A.rows, A.cols);
  const mean = new Float32Array(A.rows);
  const invStd = new Float32Array(A.rows);

  for (let i = 0; i < A.rows; i++) {
    const offset = i * A.cols;

    // Mean
    let mu = 0;
    for (let j = 0; j < A.cols; j++) {
      mu += A.data[offset + j];
    }
    mu /= A.cols;
    mean[i] = mu;

    // Variance
    let variance = 0;
    for (let j = 0; j < A.cols; j++) {
      const d = A.data[offset + j] - mu;
      variance += d * d;
    }
    variance /= A.cols;
    const inv = 1 / Math.sqrt(variance + eps);
    invStd[i] = inv;

    // Normalise, scale, shift
    for (let j = 0; j < A.cols; j++) {
      const normed = (A.data[offset + j] - mu) * inv;
      out.data[offset + j] = normed * gamma.data[j] + beta.data[j];
    }
  }

  return { out, mean, invStd };
}

/**
 * Lookup rows from an embedding table.
 * indices: array of row indices  table: [vocabSize, d_model]
 * Returns: [indices.length, d_model]
 */
export function embedding(indices: number[], table: Tensor): Tensor {
  const out = new Tensor(indices.length, table.cols);
  for (let i = 0; i < indices.length; i++) {
    const srcOffset = indices[i] * table.cols;
    const dstOffset = i * table.cols;
    for (let j = 0; j < table.cols; j++) {
      out.data[dstOffset + j] = table.data[srcOffset + j];
    }
  }
  return out;
}

/**
 * Cross-entropy loss over the full sequence.
 * logits: [seqLen, vocabSize]  targets: array of target indices
 * Returns scalar loss (average over positions).
 */
export function crossEntropyLoss(logits: Tensor, targets: number[]): number {
  let totalLoss = 0;
  for (let i = 0; i < logits.rows; i++) {
    const offset = i * logits.cols;

    // Stable log-softmax
    let max = -Infinity;
    for (let j = 0; j < logits.cols; j++) {
      if (logits.data[offset + j] > max) max = logits.data[offset + j];
    }
    let logSumExp = 0;
    for (let j = 0; j < logits.cols; j++) {
      logSumExp += Math.exp(logits.data[offset + j] - max);
    }
    logSumExp = max + Math.log(logSumExp);

    totalLoss -= logits.data[offset + targets[i]] - logSumExp;
  }
  return totalLoss / logits.rows;
}

// ---------------------------------------------------------------------------
// Operations — backward
// ---------------------------------------------------------------------------

/**
 * Backward for matmul C = A @ B.
 * Given dC, accumulates into A.grad and B.grad.
 */
export function matmulBackward(
  dC: Tensor,
  A: Tensor,
  B: Tensor
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
      A.grad[i * k + p] += sum;
    }
  }

  // dB += A^T @ dC
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

/**
 * Backward for element-wise add with broadcast.
 * dC is the upstream gradient for C = A + B.
 * Accumulates into dA and dB (B may be [1, cols]).
 */
export function addBackward(
  dC: Tensor,
  dA: Tensor,
  dB: Tensor
): void {
  // dA gets the full gradient
  for (let i = 0; i < dC.length; i++) {
    dA.grad[i] += dC.data[i];
  }

  // dB: if broadcast, sum over rows
  if (dB.rows === 1 && dC.rows > 1) {
    for (let i = 0; i < dC.rows; i++) {
      for (let j = 0; j < dC.cols; j++) {
        dB.grad[j] += dC.data[i * dC.cols + j];
      }
    }
  } else {
    for (let i = 0; i < dC.length; i++) {
      dB.grad[i] += dC.data[i];
    }
  }
}

/**
 * Backward for row-wise softmax.
 * Given dOut (upstream grad) and softmaxOut (the softmax output),
 * returns dInput.
 */
export function softmaxBackward(dOut: Tensor, softmaxOut: Tensor): Tensor {
  const dInput = new Tensor(dOut.rows, dOut.cols);
  for (let i = 0; i < dOut.rows; i++) {
    const offset = i * dOut.cols;

    // dot = sum(dOut * softmaxOut) for this row
    let dot = 0;
    for (let j = 0; j < dOut.cols; j++) {
      dot += dOut.data[offset + j] * softmaxOut.data[offset + j];
    }

    // dInput = softmaxOut * (dOut - dot)
    for (let j = 0; j < dOut.cols; j++) {
      dInput.data[offset + j] =
        softmaxOut.data[offset + j] * (dOut.data[offset + j] - dot);
    }
  }
  return dInput;
}

/**
 * Backward for ReLU.
 * dOut: upstream gradient, input: the original input to relu.
 * Returns dInput.
 */
export function reluBackward(dOut: Tensor, input: Tensor): Tensor {
  const dInput = new Tensor(dOut.rows, dOut.cols);
  for (let i = 0; i < dOut.length; i++) {
    dInput.data[i] = input.data[i] > 0 ? dOut.data[i] : 0;
  }
  return dInput;
}

/**
 * Backward for layer normalisation.
 * dOut: upstream gradient [rows, cols]
 * input: original input to layerNorm
 * gamma: scale parameter [1, cols]
 * mean, invStd: from forward pass
 *
 * Accumulates into input grad, gamma.grad, beta.grad.
 * Returns dInput tensor.
 */
export function layerNormBackward(
  dOut: Tensor,
  input: Tensor,
  gamma: Tensor,
  beta: Tensor,
  mean: Float32Array,
  invStd: Float32Array
): Tensor {
  const dInput = new Tensor(input.rows, input.cols);
  const cols = input.cols;

  for (let i = 0; i < input.rows; i++) {
    const offset = i * cols;
    const mu = mean[i];
    const inv = invStd[i];

    // Compute xhat (normalised values) for this row
    // Accumulate dgamma, dbeta
    let dxhatSum = 0;
    let dxhatXhatSum = 0;

    for (let j = 0; j < cols; j++) {
      const xhat = (input.data[offset + j] - mu) * inv;
      const dy = dOut.data[offset + j];
      const dxhat = dy * gamma.data[j];

      // Accumulate gamma and beta gradients
      gamma.grad[j] += dy * xhat;
      beta.grad[j] += dy;

      dxhatSum += dxhat;
      dxhatXhatSum += dxhat * xhat;
    }

    // dInput for this row
    for (let j = 0; j < cols; j++) {
      const xhat = (input.data[offset + j] - mu) * inv;
      const dxhat = dOut.data[offset + j] * gamma.data[j];
      dInput.data[offset + j] =
        (inv / cols) * (cols * dxhat - dxhatSum - xhat * dxhatXhatSum);
    }
  }

  return dInput;
}

/**
 * Backward for cross-entropy loss.
 * Returns dLogits: [seqLen, vocabSize] = softmax(logits) - oneHot(targets).
 * Averaged over sequence length.
 */
export function crossEntropyBackward(logits: Tensor, targets: number[]): Tensor {
  const dLogits = new Tensor(logits.rows, logits.cols);
  const sm = softmax(logits);

  for (let i = 0; i < logits.rows; i++) {
    const offset = i * logits.cols;
    for (let j = 0; j < logits.cols; j++) {
      dLogits.data[offset + j] = sm.data[offset + j] / logits.rows;
    }
    dLogits.data[offset + targets[i]] -= 1 / logits.rows;
  }
  return dLogits;
}

/**
 * Backward for embedding lookup.
 * Scatter-adds dOut gradients into embedding table grad.
 * dOut: [seqLen, d_model]  indices: token indices  table: the embedding tensor
 */
export function embeddingBackward(
  dOut: Tensor,
  indices: number[],
  table: Tensor
): void {
  for (let i = 0; i < indices.length; i++) {
    const srcOffset = i * table.cols;
    const dstOffset = indices[i] * table.cols;
    for (let j = 0; j < table.cols; j++) {
      table.grad[dstOffset + j] += dOut.data[srcOffset + j];
    }
  }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/** Extract a single row from a tensor as a new [1, cols] tensor. */
export function getRow(A: Tensor, row: number): Tensor {
  const out = new Tensor(1, A.cols);
  const offset = row * A.cols;
  for (let j = 0; j < A.cols; j++) {
    out.data[j] = A.data[offset + j];
  }
  return out;
}

/** Slice rows [start, end) from a tensor. */
export function sliceRows(A: Tensor, start: number, end: number): Tensor {
  const rows = end - start;
  const out = new Tensor(rows, A.cols);
  out.data.set(A.data.subarray(start * A.cols, end * A.cols));
  return out;
}

/** Element-wise multiply (Hadamard product). */
export function mul(A: Tensor, B: Tensor): Tensor {
  const C = new Tensor(A.rows, A.cols);
  for (let i = 0; i < A.length; i++) {
    C.data[i] = A.data[i] * B.data[i];
  }
  return C;
}

/** Create a causal mask [size, size] — 0 on lower triangle, -Infinity on upper. */
export function causalMask(size: number): Tensor {
  const mask = new Tensor(size, size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      mask.data[i * size + j] = j > i ? -Infinity : 0;
    }
  }
  return mask;
}
