/**
 * VIS: 2D embedding scatter plot.
 * Projects learned token embeddings to 2D via PCA and draws a scatter plot
 * so users can see which characters the model considers "similar".
 */

import { Tensor } from '../engine/tensor';
import { COLOURS } from './colours';

interface Point {
  x: number;
  y: number;
  char: string;
  idx: number;
}

export class EmbeddingVis {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private points: Point[] = [];
  private targetPoints: Point[] = [];
  private lerpProgress = 1;
  private idxToChar: Map<number, string> = new Map();
  private insightsContainer: HTMLElement | null = null;

  constructor(canvas: HTMLCanvasElement, insightsContainer?: HTMLElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.insightsContainer = insightsContainer ?? null;
  }

  setVocab(idxToChar: Map<number, string>): void {
    this.idxToChar = idxToChar;
  }

  /** Update with the current token embedding matrix [vocabSize, dModel]. */
  update(tokenEmb: Tensor): void {
    const n = tokenEmb.rows;
    const d = tokenEmb.cols;
    if (n < 2 || d < 2) return;

    // PCA: project to 2D using the top 2 principal components
    const projected = this.pca2D(tokenEmb);

    const newPoints: Point[] = [];
    for (let i = 0; i < n; i++) {
      const char = this.idxToChar.get(i) ?? '?';
      newPoints.push({ x: projected[i * 2], y: projected[i * 2 + 1], char, idx: i });
    }

    if (this.points.length === 0) {
      this.points = newPoints;
      this.targetPoints = newPoints;
      this.lerpProgress = 1;
    } else {
      this.targetPoints = newPoints;
      this.lerpProgress = 0;
    }
  }

  render(): void {
    const { canvas, ctx } = this;
    const w = canvas.width;
    const h = canvas.height;
    const pad = 30;

    ctx.clearRect(0, 0, w, h);
    if (this.points.length === 0) return;

    // Interpolate
    if (this.lerpProgress < 1) {
      this.lerpProgress = Math.min(this.lerpProgress + 0.06, 1);
      const t = this.lerpProgress;
      for (let i = 0; i < this.points.length && i < this.targetPoints.length; i++) {
        this.points[i].x += (this.targetPoints[i].x - this.points[i].x) * t;
        this.points[i].y += (this.targetPoints[i].y - this.points[i].y) * t;
      }
    }

    // Find bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of this.points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const plotW = w - pad * 2;
    const plotH = h - pad * 2;

    const toScreenX = (x: number) => pad + ((x - minX) / rangeX) * plotW;
    const toScreenY = (y: number) => pad + ((y - minY) / rangeY) * plotH;

    // Draw points with category colouring
    for (const p of this.points) {
      const sx = toScreenX(p.x);
      const sy = toScreenY(p.y);
      const colour = this.charColour(p.char);

      // Glow
      ctx.shadowColor = colour;
      ctx.shadowBlur = 6;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = COLOURS.text;
      const label = p.char === ' ' ? '\u00B7' : p.char === '\n' ? '\u21B5' : p.char;
      ctx.fillText(label, sx, sy - 7);
    }
  }

  clear(): void {
    this.points = [];
    this.targetPoints = [];
    const { canvas, ctx } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  /** Get human-readable insights about embedding clusters. */
  getInsights(): string[] {
    if (this.points.length < 4) return [];
    const insights: string[] = [];

    // Find nearest-neighbour pairs
    const pairs: { a: string; b: string; dist: number }[] = [];
    for (let i = 0; i < this.points.length; i++) {
      let bestDist = Infinity;
      let bestJ = -1;
      for (let j = 0; j < this.points.length; j++) {
        if (i === j) continue;
        const dx = this.points[i].x - this.points[j].x;
        const dy = this.points[i].y - this.points[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < bestDist) {
          bestDist = dist;
          bestJ = j;
        }
      }
      if (bestJ >= 0) {
        pairs.push({
          a: this.points[i].char,
          b: this.points[bestJ].char,
          dist: bestDist,
        });
      }
    }

    pairs.sort((a, b) => a.dist - b.dist);

    // Report top 3 closest pairs (avoiding duplicates)
    const seen = new Set<string>();
    let count = 0;
    for (const pair of pairs) {
      const key = [pair.a, pair.b].sort().join('');
      if (seen.has(key)) continue;
      seen.add(key);
      const a = this.charName(pair.a);
      const b = this.charName(pair.b);
      insights.push(`${a} and ${b} are close — the model sees them as similar`);
      count++;
      if (count >= 3) break;
    }

    // Check if vowels cluster together
    const vowels = 'aeiou';
    const vowelPoints = this.points.filter(p => vowels.includes(p.char.toLowerCase()));
    if (vowelPoints.length >= 3) {
      const cx = vowelPoints.reduce((s, p) => s + p.x, 0) / vowelPoints.length;
      const cy = vowelPoints.reduce((s, p) => s + p.y, 0) / vowelPoints.length;
      const avgDist = vowelPoints.reduce((s, p) =>
        s + Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2), 0) / vowelPoints.length;

      // Compare to overall spread
      const allCx = this.points.reduce((s, p) => s + p.x, 0) / this.points.length;
      const allCy = this.points.reduce((s, p) => s + p.y, 0) / this.points.length;
      const allAvgDist = this.points.reduce((s, p) =>
        s + Math.sqrt((p.x - allCx) ** 2 + (p.y - allCy) ** 2), 0) / this.points.length;

      if (avgDist < allAvgDist * 0.6) {
        insights.push('Vowels are clustering together — the model groups them by role');
      }
    }

    return insights;
  }

  /** Colour-code by character type. */
  private charColour(ch: string): string {
    if ('aeiouAEIOU'.includes(ch)) return '#4fc3f7'; // blue - vowels
    if ('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'.includes(ch)) return '#81c784'; // green - consonants
    if (' \t\n'.includes(ch)) return '#ffb74d'; // amber - whitespace
    return '#ce93d8'; // purple - punctuation
  }

  private charName(ch: string): string {
    if (ch === ' ') return 'Space';
    if (ch === '\n') return 'Newline';
    if (ch === '\t') return 'Tab';
    return `"${ch}"`;
  }

  /**
   * Simple PCA: project N x D data to N x 2.
   * Uses power iteration to find top 2 eigenvectors of the covariance matrix.
   */
  private pca2D(data: Tensor): Float32Array {
    const n = data.rows;
    const d = data.cols;

    // 1. Center the data
    const mean = new Float32Array(d);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        mean[j] += data.data[i * d + j];
      }
    }
    for (let j = 0; j < d; j++) mean[j] /= n;

    const centered = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        centered[i * d + j] = data.data[i * d + j] - mean[j];
      }
    }

    // 2. Power iteration for top 2 eigenvectors
    const pc1 = this.powerIteration(centered, n, d, null);
    const pc2 = this.powerIteration(centered, n, d, pc1);

    // 3. Project: result[i] = (dot(row_i, pc1), dot(row_i, pc2))
    const result = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      let d1 = 0, d2 = 0;
      for (let j = 0; j < d; j++) {
        const v = centered[i * d + j];
        d1 += v * pc1[j];
        d2 += v * pc2[j];
      }
      result[i * 2] = d1;
      result[i * 2 + 1] = d2;
    }

    return result;
  }

  /** Power iteration to find a principal component. deflate is the previous PC to remove. */
  private powerIteration(
    centered: Float32Array,
    n: number,
    d: number,
    deflate: Float32Array | null
  ): Float32Array {
    // Random initial vector
    const v = new Float32Array(d);
    for (let j = 0; j < d; j++) v[j] = Math.random() - 0.5;

    for (let iter = 0; iter < 50; iter++) {
      // Multiply: Av = X^T * (X * v)
      // First: Xv = centered * v (n-dimensional)
      const xv = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let dot = 0;
        for (let j = 0; j < d; j++) {
          dot += centered[i * d + j] * v[j];
        }
        xv[i] = dot;
      }

      // Then: X^T * xv (d-dimensional)
      const result = new Float32Array(d);
      for (let j = 0; j < d; j++) {
        let dot = 0;
        for (let i = 0; i < n; i++) {
          dot += centered[i * d + j] * xv[i];
        }
        result[j] = dot;
      }

      // Deflate: remove component along previous PC
      if (deflate) {
        let proj = 0;
        for (let j = 0; j < d; j++) proj += result[j] * deflate[j];
        for (let j = 0; j < d; j++) result[j] -= proj * deflate[j];
      }

      // Normalize
      let norm = 0;
      for (let j = 0; j < d; j++) norm += result[j] * result[j];
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < d; j++) v[j] = result[j] / norm;
    }

    return v;
  }
}
