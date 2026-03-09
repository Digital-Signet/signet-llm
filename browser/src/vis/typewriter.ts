/**
 * VIS 1: The Learning Typewriter — hero visualisation.
 * Shows generated text evolving over training, with per-character confidence colouring.
 */

import { confidenceColour, COLOURS } from './colours';
import type { TokenMeta } from '../engine/generate';

interface Sample {
  step: number;
  text: string;
  meta: TokenMeta[];
  revealedCount: number;
  isMilestone?: boolean;
  milestoneText?: string;
}

export class Typewriter {
  private container: HTMLElement;
  private samples: Sample[] = [];
  private revealInterval: number | null = null;
  private onHover: ((meta: TokenMeta | null) => void) | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.startRevealLoop();
  }

  setHoverCallback(cb: (meta: TokenMeta | null) => void): void {
    this.onHover = cb;
  }

  addSample(step: number, text: string, meta: TokenMeta[]): void {
    this.samples.push({ step, text, meta, revealedCount: 0 });
    this.renderAll();
  }

  addMilestone(message: string): void {
    this.samples.push({
      step: 0,
      text: message,
      meta: [],
      revealedCount: message.length,
      isMilestone: true,
      milestoneText: message,
    });
    this.renderAll();
  }

  clear(): void {
    this.samples = [];
    this.container.innerHTML = '';
  }

  private startRevealLoop(): void {
    this.revealInterval = window.setInterval(() => {
      let didReveal = false;
      for (const sample of this.samples) {
        if (sample.revealedCount < sample.text.length) {
          sample.revealedCount = Math.min(
            sample.revealedCount + 2, // reveal 2 chars per tick for speed
            sample.text.length
          );
          didReveal = true;
        }
      }
      if (didReveal) this.renderAll();
    }, 30);
  }

  private renderAll(): void {
    this.container.innerHTML = '';

    for (const sample of this.samples) {
      // Milestone banner
      if (sample.isMilestone) {
        const banner = document.createElement('div');
        banner.className = 'typewriter-milestone';
        banner.textContent = sample.milestoneText!;
        this.container.appendChild(banner);
        continue;
      }

      const line = document.createElement('div');
      line.className = 'typewriter-line';

      // Step label
      const label = document.createElement('span');
      label.className = 'typewriter-step';
      label.textContent = `Step ${sample.step.toLocaleString()}`;
      line.appendChild(label);

      // Characters
      const textSpan = document.createElement('span');
      textSpan.className = 'typewriter-text';

      for (let i = 0; i < sample.revealedCount; i++) {
        const ch = document.createElement('span');
        ch.className = 'typewriter-char';
        ch.textContent = sample.text[i];

        if (i < sample.meta.length) {
          ch.style.color = confidenceColour(sample.meta[i].confidence);

          const meta = sample.meta[i];
          ch.addEventListener('mouseenter', () => {
            ch.style.background = 'rgba(79, 195, 247, 0.15)';
            this.onHover?.(meta);
          });
          ch.addEventListener('mouseleave', () => {
            ch.style.background = 'transparent';
            this.onHover?.(null);
          });
        } else {
          ch.style.color = COLOURS.textDim;
        }

        textSpan.appendChild(ch);
      }

      // Blinking cursor if still revealing
      if (sample.revealedCount < sample.text.length) {
        const cursor = document.createElement('span');
        cursor.className = 'typewriter-cursor';
        cursor.textContent = '\u258C'; // ▌
        textSpan.appendChild(cursor);
      }

      line.appendChild(textSpan);
      this.container.appendChild(line);
    }

    // Auto-scroll to bottom
    this.container.scrollTop = this.container.scrollHeight;
  }

  destroy(): void {
    if (this.revealInterval !== null) {
      clearInterval(this.revealInterval);
    }
  }
}
