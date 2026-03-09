/**
 * VIS 4: Animated probability bar chart.
 * Shows top-10 next-token predictions with smooth transitions.
 */

import { Tensor, softmax1D } from '../engine/tensor';
import { COLOURS } from './colours';

interface BarData {
  char: string;
  prob: number;
  targetY: number;
  currentY: number;
  targetWidth: number;
  currentWidth: number;
}

export class TokenProbs {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private bars: BarData[] = [];
  private entropyLabel: HTMLElement | null;

  constructor(canvas: HTMLCanvasElement, entropyLabel?: HTMLElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.entropyLabel = entropyLabel ?? null;
  }

  /**
   * Update with logits from the last position of a forward pass.
   * vocabMap: index → character string.
   */
  update(logits: Tensor, row: number, vocabMap: Map<number, string>): void {
    const cols = logits.cols;
    const offset = row * cols;
    const rawLogits = new Float32Array(cols);
    for (let j = 0; j < cols; j++) {
      rawLogits[j] = logits.data[offset + j];
    }
    const probs = softmax1D(rawLogits);

    // Get top 10
    const indexed: { idx: number; prob: number }[] = [];
    for (let i = 0; i < probs.length; i++) {
      indexed.push({ idx: i, prob: probs[i] });
    }
    indexed.sort((a, b) => b.prob - a.prob);
    const top10 = indexed.slice(0, 10);

    // Compute entropy
    let entropy = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > 0) entropy -= probs[i] * Math.log2(probs[i]);
    }

    // Update entropy label
    if (this.entropyLabel) {
      const maxEntropy = Math.log2(cols);
      const ratio = entropy / maxEntropy;
      if (ratio > 0.7) {
        this.entropyLabel.textContent = `Entropy: ${entropy.toFixed(1)} — Uncertain`;
        this.entropyLabel.style.color = COLOURS.textDim;
      } else if (ratio > 0.3) {
        this.entropyLabel.textContent = `Entropy: ${entropy.toFixed(1)} — Learning`;
        this.entropyLabel.style.color = COLOURS.blue;
      } else {
        this.entropyLabel.textContent = `Entropy: ${entropy.toFixed(1)} — Confident`;
        this.entropyLabel.style.color = COLOURS.white;
      }
    }

    // Build target bar data
    const barHeight = 22;
    const gap = 4;
    const maxBarWidth = this.canvas.width - 90;

    const newBars: BarData[] = top10.map((item, i) => {
      const char = vocabMap.get(item.idx) ?? '?';
      const displayChar = char === ' ' ? '\u2423' : char === '\n' ? '\\n' : char;
      const targetY = i * (barHeight + gap) + 10;
      const targetWidth = item.prob * maxBarWidth;

      // Find existing bar for smooth transition
      const existing = this.bars.find(b => b.char === displayChar);
      return {
        char: displayChar,
        prob: item.prob,
        targetY,
        currentY: existing?.currentY ?? targetY,
        targetWidth,
        currentWidth: existing?.currentWidth ?? 0,
      };
    });

    this.bars = newBars;
  }

  /** Call each frame to animate towards target positions. */
  render(): void {
    const { canvas, ctx, bars } = this;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (bars.length === 0) return;

    const barHeight = 22;
    const labelWidth = 30;
    const probLabelWidth = 50;
    const barStartX = labelWidth + 5;
    const maxBarWidth = w - barStartX - probLabelWidth;
    const lerp = 0.15; // animation smoothing

    for (const bar of bars) {
      // Animate position and width
      bar.currentY += (bar.targetY - bar.currentY) * lerp;
      bar.currentWidth += (bar.targetWidth - bar.currentWidth) * lerp;

      const y = bar.currentY;
      const barW = Math.max(bar.currentWidth / (this.canvas.width - 90) * maxBarWidth, 1);

      // Bar gradient
      const gradient = ctx.createLinearGradient(barStartX, 0, barStartX + barW, 0);
      gradient.addColorStop(0, 'rgba(79, 195, 247, 0.2)');
      gradient.addColorStop(1, COLOURS.blue);
      ctx.fillStyle = gradient;

      // Rounded bar
      const radius = 3;
      ctx.beginPath();
      ctx.roundRect(barStartX, y, barW, barHeight, [0, radius, radius, 0]);
      ctx.fill();

      // Character label
      ctx.fillStyle = COLOURS.white;
      ctx.font = '13px monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(bar.char, labelWidth, y + barHeight / 2);

      // Probability label
      ctx.fillStyle = COLOURS.textDim;
      ctx.textAlign = 'left';
      ctx.fillText(
        `${(bar.prob * 100).toFixed(0)}%`,
        barStartX + barW + 8,
        y + barHeight / 2
      );
    }
  }

  clear(): void {
    this.bars = [];
    const { canvas, ctx } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}
