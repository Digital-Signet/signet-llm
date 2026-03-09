/**
 * Autoregressive text generation with metadata for X-Ray visualisation.
 */

import { Tensor, softmax1D } from './tensor';
import { ModelWeights, forward } from './model';
import { CharTokeniser } from './tokeniser';

export interface TokenMeta {
  token: number;
  confidence: number;
  topK: { token: number; char: string; prob: number }[];
  attentionWeights: Tensor[][]; // [layer][head]
}

export interface GenerationResult {
  tokens: number[];
  meta: TokenMeta[];
}

/**
 * Generate tokens with full metadata for each generated token.
 */
export function generateWithMetadata(
  model: ModelWeights,
  tokeniser: CharTokeniser,
  seed: number[],
  maxTokens: number,
  temperature: number,
  topK = 0
): GenerationResult {
  const tokens = [...seed];
  const meta: TokenMeta[] = [];
  const blockSize = model.config.blockSize;

  for (let i = 0; i < maxTokens; i++) {
    // Use last blockSize tokens as context
    const start = Math.max(0, tokens.length - blockSize);
    const context = tokens.slice(start);

    const { logits, attentionWeights } = forward(context, model);

    // Get logits for last position
    const lastRow = logits.rows - 1;
    const lastLogits = new Float32Array(logits.cols);
    for (let j = 0; j < logits.cols; j++) {
      lastLogits[j] = logits.data[lastRow * logits.cols + j] / temperature;
    }

    // Optional top-k filtering
    if (topK > 0) {
      applyTopKFilter(lastLogits, topK);
    }

    const probs = softmax1D(lastLogits);

    // Sample
    const nextToken = sampleFromDistribution(probs);
    const confidence = probs[nextToken];

    // Get top-k tokens for metadata
    const topKMeta = getTopKTokens(probs, tokeniser, 5);

    tokens.push(nextToken);
    meta.push({
      token: nextToken,
      confidence,
      topK: topKMeta,
      attentionWeights,
    });
  }

  return { tokens, meta };
}

/**
 * Simple generation without metadata (for quick sampling).
 */
export function generate(
  model: ModelWeights,
  seed: number[],
  maxTokens: number,
  temperature: number,
): number[] {
  const tokens = [...seed];
  const blockSize = model.config.blockSize;

  for (let i = 0; i < maxTokens; i++) {
    const start = Math.max(0, tokens.length - blockSize);
    const context = tokens.slice(start);
    const { logits } = forward(context, model);

    const lastRow = logits.rows - 1;
    const lastLogits = new Float32Array(logits.cols);
    for (let j = 0; j < logits.cols; j++) {
      lastLogits[j] = logits.data[lastRow * logits.cols + j] / temperature;
    }

    const probs = softmax1D(lastLogits);
    tokens.push(sampleFromDistribution(probs));
  }

  return tokens;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sampleFromDistribution(probs: Float32Array): number {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1;
}

function applyTopKFilter(logits: Float32Array, k: number): void {
  // Find the k-th largest value
  const sorted = [...logits].sort((a, b) => b - a);
  const threshold = sorted[Math.min(k, sorted.length) - 1];

  for (let i = 0; i < logits.length; i++) {
    if (logits[i] < threshold) {
      logits[i] = -Infinity;
    }
  }
}

function getTopKTokens(
  probs: Float32Array,
  tokeniser: CharTokeniser,
  k: number
): { token: number; char: string; prob: number }[] {
  const indexed = Array.from(probs).map((p, i) => ({ token: i, prob: p }));
  indexed.sort((a, b) => b.prob - a.prob);
  return indexed.slice(0, k).map(({ token, prob }) => ({
    token,
    char: tokeniser.idxToChar.get(token) ?? '?',
    prob,
  }));
}
