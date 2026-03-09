/**
 * VIS: Bigram heatmap — shows P(next char | current char) as learned by the model.
 * Updated periodically from the model's forward pass predictions.
 */

import { COLOURS } from './colours';

export class BigramHeatmap {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  // bigramProbs[i][j] = smoothed P(char j | char i)
  private bigramProbs: Float32Array[] = [];
  private chars: string[] = [];
  private charToIdx: Map<string, number> = new Map();

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
  }

  /** Set the character vocabulary. Call once after tokeniser is built. */
  setVocab(idxToChar: Map<number, string>): void {
    this.chars = [];
    this.charToIdx = new Map();
    // Sort for consistent ordering (space first, then printable, then others)
    const entries = Array.from(idxToChar.entries()).sort((a, b) => a[0] - b[0]);
    for (const [, ch] of entries) {
      this.charToIdx.set(ch, this.chars.length);
      this.chars.push(ch);
    }
    const n = this.chars.length;
    this.bigramProbs = [];
    for (let i = 0; i < n; i++) {
      this.bigramProbs.push(new Float32Array(n));
    }
  }

  /**
   * Update from a training forward pass.
   * inputTokens: the input sequence, probsPerPosition: softmax output per position.
   * Uses EMA to smooth updates over time.
   */
  update(inputTokens: number[], softmaxRows: Float32Array[]): void {
    const alpha = 0.05; // EMA blending — slow enough to be smooth, fast enough to show change
    for (let pos = 0; pos < inputTokens.length && pos < softmaxRows.length; pos++) {
      const row = inputTokens[pos];
      if (row < 0 || row >= this.bigramProbs.length) continue;
      const probs = softmaxRows[pos];
      const target = this.bigramProbs[row];
      for (let j = 0; j < target.length && j < probs.length; j++) {
        target[j] = target[j] * (1 - alpha) + probs[j] * alpha;
      }
    }
  }

  clear(): void {
    const n = this.chars.length;
    for (let i = 0; i < n; i++) {
      this.bigramProbs[i]?.fill(0);
    }
  }

  render(): void {
    const { canvas, ctx, chars, bigramProbs } = this;
    const w = canvas.width;
    const h = canvas.height;
    const n = chars.length;

    ctx.clearRect(0, 0, w, h);
    if (n === 0 || bigramProbs.length === 0) return;

    // Layout
    const labelSpace = 16;
    const axisLabelSpace = 14;
    const pad = { top: axisLabelSpace + 2, left: labelSpace + 2, right: 4, bottom: labelSpace + axisLabelSpace + 2 };
    const gridW = w - pad.left - pad.right;
    const gridH = h - pad.top - pad.bottom;
    const cellW = gridW / n;
    const cellH = gridH / n;

    // Find max probability for colour scaling
    let maxProb = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (bigramProbs[i][j] > maxProb) maxProb = bigramProbs[i][j];
      }
    }
    if (maxProb < 1e-6) maxProb = 1;

    // Draw cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const intensity = bigramProbs[i][j] / maxProb;
        const x = pad.left + j * cellW;
        const y = pad.top + i * cellH;

        if (intensity > 0.01) {
          // Blue glow: more intense = brighter and more opaque
          const r = Math.round(10 + 69 * intensity);
          const g = Math.round(10 + 185 * intensity);
          const b = Math.round(15 + 232 * intensity);
          const a = 0.15 + 0.85 * intensity;
          ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
        } else {
          ctx.fillStyle = 'rgba(255,255,255,0.02)';
        }
        ctx.fillRect(x, y, cellW - 0.5, cellH - 0.5);
      }
    }

    // Character labels
    const fontSize = Math.min(10, Math.max(6, cellW - 1));
    ctx.font = `${fontSize}px monospace`;
    ctx.textAlign = 'center';

    // Top labels (predicted next char)
    ctx.fillStyle = COLOURS.textDim;
    for (let j = 0; j < n; j++) {
      const x = pad.left + j * cellW + cellW / 2;
      const label = this.displayChar(chars[j]);
      ctx.fillText(label, x, pad.top - 3);
    }

    // Left labels (current char)
    ctx.textAlign = 'right';
    for (let i = 0; i < n; i++) {
      const y = pad.top + i * cellH + cellH / 2 + fontSize / 3;
      const label = this.displayChar(chars[i]);
      ctx.fillText(label, pad.left - 3, y);
    }

    // Axis titles
    ctx.fillStyle = COLOURS.textDim;
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('predicted next →', pad.left + gridW / 2, h - 2);

    ctx.save();
    ctx.translate(8, pad.top + gridH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('current char →', 0, 0);
    ctx.restore();
  }

  /** Return top N strongest learned bigrams as human-readable insights. */
  getTopPatterns(n = 8): { from: string; to: string; prob: number; sentence: string }[] {
    const results: { from: string; to: string; prob: number; sentence: string }[] = [];
    const chars = this.chars;
    const bp = this.bigramProbs;

    for (let i = 0; i < chars.length; i++) {
      for (let j = 0; j < chars.length; j++) {
        if (bp[i][j] > 0.01) {
          results.push({
            from: chars[i],
            to: chars[j],
            prob: bp[i][j],
            sentence: this.describePair(chars[i], chars[j], bp[i][j]),
          });
        }
      }
    }

    results.sort((a, b) => b.prob - a.prob);
    return results.slice(0, n);
  }

  /** Make whitespace characters visible */
  private displayChar(ch: string): string {
    if (ch === ' ') return '·';
    if (ch === '\n') return '↵';
    if (ch === '\t') return '→';
    return ch;
  }

  /** Turn a bigram into a readable sentence. */
  private describePair(from: string, to: string, prob: number): string {
    const pct = Math.round(prob * 100);
    const fromName = this.charName(from);
    const toName = this.charName(to);
    return `After ${fromName}, ${toName} is likely (${pct}%)`;
  }

  private charName(ch: string): string {
    if (ch === ' ') return 'a space';
    if (ch === '\n') return 'a newline';
    if (ch === '\t') return 'a tab';
    if (ch === '.') return 'a period';
    if (ch === ',') return 'a comma';
    if (ch === '?') return '"?"';
    if (ch === '!') return '"!"';
    if (ch === ':') return '":"';
    if (ch === ';') return '";"';
    if (ch === "'") return 'an apostrophe';
    if (ch === '"') return 'a quote';
    if (ch === '-') return 'a dash';
    if (ch === '(') return '"("';
    if (ch === ')') return '")"';
    return `"${ch}"`;
  }
}
