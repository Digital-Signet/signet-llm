/**
 * VIS 2: Animated loss curve with pulsing current-position dot.
 */

import { COLOURS } from './colours';

export class LossCurve {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private rawData: number[] = [];
  private emaData: number[] = [];
  private emaAlpha = 0.99;
  private maxPoints = 5000;
  private milestones: number[] = []; // step numbers where samples were generated
  private animationTime = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
  }

  addPoint(loss: number): void {
    this.rawData.push(loss);

    // EMA
    const prev = this.emaData.length > 0 ? this.emaData[this.emaData.length - 1] : loss;
    this.emaData.push(this.emaAlpha * prev + (1 - this.emaAlpha) * loss);

    // Trim buffer
    if (this.rawData.length > this.maxPoints) {
      this.rawData.shift();
      this.emaData.shift();
    }
  }

  addMilestone(step: number): void {
    this.milestones.push(step);
  }

  clear(): void {
    this.rawData = [];
    this.emaData = [];
    this.milestones = [];
  }

  render(): void {
    const { canvas, ctx } = this;
    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 20, right: 20, bottom: 30, left: 50 };

    ctx.clearRect(0, 0, w, h);

    if (this.rawData.length < 2) return;

    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Y range — based on VISIBLE window of EMA data so the curve stays readable
    // Use the last `plotW` worth of points (what's actually on screen)
    const visibleCount = Math.min(this.emaData.length, Math.ceil(plotW / 2));
    const visibleStart = this.emaData.length - visibleCount;
    const visibleEma = this.emaData.slice(visibleStart);
    const visibleRaw = this.rawData.slice(visibleStart);

    // Base range on EMA (smooth) with some headroom for raw spikes
    let emaMin = Infinity, emaMax = -Infinity;
    for (let i = 0; i < visibleEma.length; i++) {
      if (visibleEma[i] < emaMin) emaMin = visibleEma[i];
      if (visibleEma[i] > emaMax) emaMax = visibleEma[i];
    }
    // Include the 95th percentile of raw data so spikes are visible but don't blow out the scale
    const sortedRaw = [...visibleRaw].sort((a, b) => a - b);
    const p95 = sortedRaw[Math.floor(sortedRaw.length * 0.95)] ?? emaMax;

    let yMin = Math.max(0, emaMin - (emaMax - emaMin) * 0.2);
    let yMax = Math.max(p95, emaMax) * 1.15;
    if (yMax - yMin < 0.05) { yMin = Math.max(0, emaMin - 0.05); yMax = emaMax + 0.05; }

    const xScale = plotW / Math.max(this.rawData.length - 1, 1);
    const yScale = plotH / (yMax - yMin);

    const toX = (i: number) => pad.left + i * xScale;
    const toY = (v: number) => pad.top + plotH - (v - yMin) * yScale;

    // Grid lines
    ctx.strokeStyle = COLOURS.grid;
    ctx.lineWidth = 1;
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const v = yMin + (yMax - yMin) * (i / yTicks);
      const y = toY(v);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();

      // Y-axis label
      ctx.fillStyle = COLOURS.textDim;
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(v.toFixed(2), pad.left - 6, y + 3);
    }

    // Raw loss line (thin, transparent)
    ctx.strokeStyle = COLOURS.blueDim;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < this.rawData.length; i++) {
      const x = toX(i);
      const y = toY(this.rawData[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // EMA line (thick, solid)
    ctx.strokeStyle = COLOURS.blue;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < this.emaData.length; i++) {
      const x = toX(i);
      const y = toY(this.emaData[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Pulsing dot at current position
    this.animationTime += 0.05;
    const pulseRadius = 4 + Math.sin(this.animationTime * 4) * 2;
    const lastX = toX(this.emaData.length - 1);
    const lastY = toY(this.emaData[this.emaData.length - 1]);

    // Glow
    ctx.shadowColor = COLOURS.blue;
    ctx.shadowBlur = 10;
    ctx.fillStyle = COLOURS.blue;
    ctx.beginPath();
    ctx.arc(lastX, lastY, pulseRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // X-axis label
    ctx.fillStyle = COLOURS.textDim;
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('training steps', pad.left + plotW / 2, h - 5);
  }
}
