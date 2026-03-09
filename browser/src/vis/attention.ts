/**
 * VIS 3: Attention arcs visualisation.
 * Draws Bézier arcs between tokens, thickness = attention weight.
 */

import { Tensor } from '../engine/tensor';
import { COLOURS, HEAD_COLOURS } from './colours';

export class AttentionVis {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private currentWeights: Tensor[][] | null = null; // [layer][head]
  private targetWeights: Tensor[][] | null = null;
  private chars: string[] = [];
  private displayLayer = 0; // which layer to display
  private lerpProgress = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
  }

  setChars(chars: string[]): void {
    this.chars = chars;
  }

  setLayer(layer: number): void {
    this.displayLayer = layer;
  }

  /** Smooth transition to new attention weights. */
  animateTo(weights: Tensor[][]): void {
    if (this.currentWeights === null) {
      this.currentWeights = weights;
      this.targetWeights = weights;
      this.lerpProgress = 1;
    } else {
      this.targetWeights = weights;
      this.lerpProgress = 0;
    }
  }

  render(): void {
    const { canvas, ctx } = this;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (!this.currentWeights || !this.targetWeights) return;
    if (this.displayLayer >= this.currentWeights.length) return;

    // Interpolate weights
    if (this.lerpProgress < 1) {
      this.lerpProgress = Math.min(this.lerpProgress + 0.08, 1);
    }

    const heads = this.currentWeights[this.displayLayer];
    const targetHeads = this.targetWeights[this.displayLayer];
    const nHeads = heads.length;
    const seqLen = heads[0].rows;
    const displayLen = Math.min(seqLen, 32); // cap for visual clarity

    // Layout: characters spaced evenly
    const charWidth = 18;
    const startX = 20;
    const textY = h / 2;
    const arcMaxHeight = (h / 2) - 30;

    // Draw characters
    ctx.font = '13px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = COLOURS.text;

    for (let i = 0; i < displayLen && i < this.chars.length; i++) {
      const x = startX + i * charWidth + charWidth / 2;
      const ch = this.chars[i] === ' ' ? '\u00B7' : this.chars[i]; // middle dot for space
      ctx.fillText(ch, x, textY);
    }

    // Draw arcs for each head
    for (let headIdx = 0; headIdx < nHeads; headIdx++) {
      const currentHead = heads[headIdx];
      const targetHead = targetHeads[headIdx];
      const colour = HEAD_COLOURS[headIdx % HEAD_COLOURS.length];
      const drawAbove = headIdx % 2 === 0; // alternate above/below

      for (let q = 0; q < displayLen; q++) {
        for (let k = 0; k <= q; k++) { // causal: only attend to past
          // Interpolate weight
          const cw = currentHead.at(q, k);
          const tw = targetHead.at(q, k);
          const weight = cw + (tw - cw) * this.lerpProgress;

          if (weight < 0.05) continue; // skip weak connections

          const x1 = startX + k * charWidth + charWidth / 2;
          const x2 = startX + q * charWidth + charWidth / 2;
          const distance = Math.abs(q - k);
          const heightFactor = Math.min(distance / displayLen, 1);
          const arcHeight = heightFactor * arcMaxHeight;

          const cpY = drawAbove ? textY - arcHeight - 15 : textY + arcHeight + 15;

          ctx.strokeStyle = colour;
          ctx.globalAlpha = Math.min(weight * 2, 0.9); // scale opacity
          ctx.lineWidth = weight * 3;

          ctx.beginPath();
          ctx.moveTo(x1, textY + (drawAbove ? -8 : 8));
          ctx.quadraticCurveTo(
            (x1 + x2) / 2,
            cpY,
            x2,
            textY + (drawAbove ? -8 : 8)
          );
          ctx.stroke();
        }
      }
    }

    ctx.globalAlpha = 1;

    // Head labels
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let h = 0; h < nHeads; h++) {
      ctx.fillStyle = HEAD_COLOURS[h % HEAD_COLOURS.length];
      const yPos = h % 2 === 0 ? 15 : canvas.height - 10;
      ctx.fillText(`Head ${h + 1}`, w - 10, yPos);
    }
  }

  clear(): void {
    this.currentWeights = null;
    this.targetWeights = null;
    const { canvas, ctx } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  /** Analyze attention patterns and return human-readable descriptions per head. */
  getInsights(): string[] {
    const weights = this.targetWeights ?? this.currentWeights;
    if (!weights || this.displayLayer >= weights.length) return [];

    const heads = weights[this.displayLayer];
    const insights: string[] = [];

    for (let h = 0; h < heads.length; h++) {
      const head = heads[h];
      const seqLen = head.rows;
      if (seqLen < 2) continue;

      // Analyze: what distance does this head mostly attend to?
      // For each query position, find where it attends most strongly
      let selfAttendTotal = 0;
      let prevAttendTotal = 0;
      let farAttendTotal = 0;
      let avgDistance = 0;
      let maxWeight = 0;
      let maxFrom = 0;
      let maxTo = 0;
      let count = 0;

      for (let q = 1; q < seqLen; q++) {
        for (let k = 0; k <= q; k++) {
          const w = head.at(q, k);
          const dist = q - k;

          if (dist === 0) selfAttendTotal += w;
          else if (dist === 1) prevAttendTotal += w;
          else farAttendTotal += w;

          avgDistance += w * dist;
          count++;

          if (w > maxWeight) {
            maxWeight = w;
            maxFrom = q;
            maxTo = k;
          }
        }
      }

      const positions = seqLen - 1; // number of query positions with at least one key
      selfAttendTotal /= positions;
      prevAttendTotal /= positions;
      farAttendTotal /= positions;
      avgDistance /= positions;

      const headName = `Head ${h + 1}`;
      const colour = h === 0 ? 'blue' : 'amber';

      // Describe the pattern
      if (selfAttendTotal > 0.5) {
        insights.push(`${headName} (${colour}): mostly looks at itself — a "copy" pattern`);
      } else if (prevAttendTotal > 0.4) {
        insights.push(`${headName} (${colour}): focuses on the previous character — learning local sequences`);
      } else if (avgDistance < 3) {
        insights.push(`${headName} (${colour}): attends to nearby characters (avg ${avgDistance.toFixed(1)} back)`);
      } else if (avgDistance > 8) {
        insights.push(`${headName} (${colour}): looks far back (avg ${avgDistance.toFixed(1)} chars) — searching for long-range patterns`);
      } else {
        insights.push(`${headName} (${colour}): attends ~${avgDistance.toFixed(0)} characters back on average`);
      }

      // Add a specific example if we have chars
      if (this.chars.length > maxFrom && this.chars.length > maxTo && maxWeight > 0.2) {
        const fromCh = this.chars[maxFrom] === ' ' ? 'space' : this.chars[maxFrom] === '\n' ? 'newline' : `"${this.chars[maxFrom]}"`;
        const toCh = this.chars[maxTo] === ' ' ? 'space' : this.chars[maxTo] === '\n' ? 'newline' : `"${this.chars[maxTo]}"`;
        insights.push(`  Strongest link: ${fromCh} looking back at ${toCh} (${Math.round(maxWeight * 100)}%)`);
      }
    }

    return insights;
  }
}
