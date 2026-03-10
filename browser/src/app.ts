/**
 * Entry point. Wires engine, UI, and visualisations together.
 * Owns the training loop.
 */

import { ModelConfig, initModel, paramCount, forward, ModelWeights, BlockWeights } from './engine/model';
import { Tensor, softmax } from './engine/tensor';
import { initAdam, trainStep, ADAM_DEFAULTS, AdamConfig } from './engine/train';
import { CharTokeniser } from './engine/tokeniser';
import { generateWithMetadata } from './engine/generate';
import { Typewriter } from './vis/typewriter';
import { LossCurve } from './vis/loss-curve';
import { TokenProbs } from './vis/token-probs';
import { AttentionVis } from './vis/attention';
import { BigramHeatmap } from './vis/bigram-heatmap';
import { EmbeddingVis } from './vis/embedding-vis';
import { PRESETS, PRESET_NAMES } from './presets';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let isTraining = false;
let step = 0;
let cursor = 0;

let tokeniser: CharTokeniser;
let encodedData: number[];
let model: ReturnType<typeof initModel>;
let adam: ReturnType<typeof initAdam>;
let adamConfig: AdamConfig = { ...ADAM_DEFAULTS };

let temperature = 0.4;
let stepsPerFrame = 100;

// Current architecture config
let archConfig = {
  dModel: 64,
  nHeads: 2,
  nLayers: 2,
  blockSize: 64,
};

// Vis instances
let typewriter: Typewriter;
let lossCurve: LossCurve;
let tokenProbs: TokenProbs;
let attentionVis: AttentionVis;
let bigramHeatmap: BigramHeatmap;
let embeddingVis: EmbeddingVis;

// ---------------------------------------------------------------------------
// Milestone tracking
// ---------------------------------------------------------------------------

interface MilestoneState {
  lossBelow3: boolean;
  lossBelow1: boolean;
  lossBelow05: boolean;
  firstRealWord: boolean;
  recognisablePhrase: boolean;
}

let milestones: MilestoneState = {
  lossBelow3: false,
  lossBelow1: false,
  lossBelow05: false,
  firstRealWord: false,
  recognisablePhrase: false,
};

const COMMON_WORDS = new Set([
  'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but',
  'his', 'from', 'they', 'been', 'one', 'had', 'her', 'what', 'there', 'can',
  'all', 'will', 'each', 'make', 'like', 'long', 'look', 'many', 'day', 'are',
  'was', 'were', 'who', 'which', 'their', 'said', 'she', 'him', 'how', 'than',
  'its', 'let', 'may', 'would', 'could', 'into', 'our', 'just', 'about', 'know',
  'take', 'come', 'some', 'time', 'very', 'when', 'out', 'than', 'them', 'then',
  'now', 'way', 'down', 'did', 'get', 'has', 'man', 'too', 'any', 'own',
]);

function checkMilestones(loss: number, generatedText?: string): void {
  if (!milestones.lossBelow3 && loss < 3.0) {
    milestones.lossBelow3 = true;
    typewriter.addMilestone('Loss dropped below 3.0 — the model is starting to learn character frequencies');
  }

  if (!milestones.lossBelow1 && loss < 1.0) {
    milestones.lossBelow1 = true;
    typewriter.addMilestone('Loss below 1.0 — predictions are getting confident');
  }

  if (!milestones.lossBelow05 && loss < 0.5) {
    milestones.lossBelow05 = true;
    typewriter.addMilestone('Loss below 0.5 — the model is memorising the training text');
  }

  if (generatedText && !milestones.firstRealWord) {
    const words = generatedText.toLowerCase().replace(/[^a-z ]/g, '').split(/\s+/);
    const realWords = words.filter(w => w.length >= 3 && COMMON_WORDS.has(w));
    if (realWords.length >= 2) {
      milestones.firstRealWord = true;
      typewriter.addMilestone(`Real words appearing: "${realWords.slice(0, 3).join('", "')}"`);
    }
  }

  if (generatedText && !milestones.recognisablePhrase && milestones.firstRealWord) {
    // Check for any 3+ word sequence from training data
    const trainingText = ($('training-text') as HTMLTextAreaElement).value.toLowerCase();
    const genLower = generatedText.toLowerCase();
    const words = genLower.split(/\s+/).filter(w => w.length > 0);
    for (let i = 0; i <= words.length - 3; i++) {
      const phrase = words.slice(i, i + 3).join(' ');
      if (phrase.length >= 8 && trainingText.includes(phrase)) {
        milestones.recognisablePhrase = true;
        typewriter.addMilestone(`Recognisable phrase from training text: "${phrase}"`);
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const $ = (id: string) => document.getElementById(id)!;

function initDOM() {
  // Presets
  const presetContainer = $('presets');
  for (const preset of PRESET_NAMES) {
    const btn = document.createElement('button');
    btn.className = 'preset-btn';
    btn.textContent = preset.label;
    btn.dataset.key = preset.key;
    btn.addEventListener('click', () => {
      (document.querySelectorAll('.preset-btn') as NodeListOf<HTMLElement>).forEach(
        b => b.classList.remove('active')
      );
      btn.classList.add('active');
      ($('training-text') as HTMLTextAreaElement).value = PRESETS[preset.key];
    });
    presetContainer.appendChild(btn);
  }

  // Custom button
  const customBtn = document.createElement('button');
  customBtn.className = 'preset-btn';
  customBtn.textContent = 'Paste your own';
  customBtn.addEventListener('click', () => {
    (document.querySelectorAll('.preset-btn') as NodeListOf<HTMLElement>).forEach(
      b => b.classList.remove('active')
    );
    customBtn.classList.add('active');
    const ta = $('training-text') as HTMLTextAreaElement;
    ta.value = '';
    ta.focus();
  });
  presetContainer.appendChild(customBtn);

  // Set default
  ($('training-text') as HTMLTextAreaElement).value = PRESETS.shakespeare;
  presetContainer.querySelector('button')!.classList.add('active');

  // Buttons
  $('btn-train').addEventListener('click', startTraining);
  $('btn-pause').addEventListener('click', pauseTraining);
  $('btn-reset').addEventListener('click', resetAll);
  $('btn-export').addEventListener('click', exportModel);

  // Architecture controls
  setupToggleGroup('heads', [1, 2, 4], archConfig.nHeads, v => {
    archConfig.nHeads = v;
    resetModel();
  });
  setupToggleGroup('layers', [1, 2, 4], archConfig.nLayers, v => {
    archConfig.nLayers = v;
    resetModel();
  });
  setupToggleGroup('embed', [16, 32, 64], archConfig.dModel, v => {
    archConfig.dModel = v;
    resetModel();
  });
  setupToggleGroup('context', [32, 64, 128], archConfig.blockSize, v => {
    archConfig.blockSize = v;
    resetModel();
  });

  // Sliders
  const lrSlider = $('lr-slider') as HTMLInputElement;
  const lrValue = $('lr-value');
  lrSlider.addEventListener('input', () => {
    adamConfig.lr = parseFloat(lrSlider.value);
    lrValue.textContent = adamConfig.lr.toExponential(0);
  });

  const tempSlider = $('temp-slider') as HTMLInputElement;
  const tempValue = $('temp-value');
  tempSlider.addEventListener('input', () => {
    temperature = parseFloat(tempSlider.value);
    tempValue.textContent = temperature.toFixed(1);
  });

  // Speed slider
  const speedSlider = $('speed-slider') as HTMLInputElement;
  const speedValue = $('speed-value');
  speedSlider.addEventListener('input', () => {
    stepsPerFrame = parseInt(speedSlider.value, 10);
    speedValue.textContent = stepsPerFrame.toString();
  });

  // Init vis
  typewriter = new Typewriter($('typewriter'));
  lossCurve = new LossCurve($('loss-canvas') as HTMLCanvasElement);
  tokenProbs = new TokenProbs(
    $('probs-canvas') as HTMLCanvasElement,
    $('entropy-label')
  );
  attentionVis = new AttentionVis($('attention-canvas') as HTMLCanvasElement);
  bigramHeatmap = new BigramHeatmap($('bigram-canvas') as HTMLCanvasElement);
  embeddingVis = new EmbeddingVis(
    $('embedding-canvas') as HTMLCanvasElement,
    $('embedding-insights')
  );

  // Canvas sizing
  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);
}

function resizeCanvases(): void {
  const setSize = (id: string, w: number, h: number) => {
    const c = $(id) as HTMLCanvasElement;
    c.width = w;
    c.height = h;
  };

  const isMobile = window.innerWidth <= 480;
  const isTablet = window.innerWidth <= 768;

  const lossPanel = $('loss-panel');
  const probsPanel = $('probs-panel');
  const bigramChart = document.querySelector('.bigram-chart') as HTMLElement;
  const attnPanel = $('attention-panel');
  const embPanel = $('embedding-panel');

  const lossH = isMobile ? 150 : 200;
  const probsH = isMobile ? 200 : 280;
  const attnH = isMobile ? 150 : 200;
  const embH = isMobile ? 220 : isTablet ? 260 : 300;

  setSize('loss-canvas', lossPanel.clientWidth - 2, lossH);
  setSize('probs-canvas', probsPanel.clientWidth - 2, probsH);
  const bigramSize = bigramChart ? bigramChart.clientWidth : 300;
  setSize('bigram-canvas', bigramSize, bigramSize); // square
  setSize('attention-canvas', attnPanel.clientWidth - 2, attnH);
  setSize('embedding-canvas', embPanel.clientWidth - 2, embH);
}

// ---------------------------------------------------------------------------
// Toggle button groups
// ---------------------------------------------------------------------------

function setupToggleGroup(
  containerId: string,
  values: number[],
  initial: number,
  onChange: (v: number) => void
): void {
  const container = $(containerId);
  for (const v of values) {
    const btn = document.createElement('button');
    btn.className = 'toggle-btn' + (v === initial ? ' active' : '');
    btn.textContent = String(v);
    btn.addEventListener('click', () => {
      container.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      onChange(v);
    });
    container.appendChild(btn);
  }
}

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

function buildModel(): void {
  const text = ($('training-text') as HTMLTextAreaElement).value;
  if (text.length < archConfig.blockSize + 1) {
    alert(`Training text too short. Need at least ${archConfig.blockSize + 1} characters.`);
    return;
  }

  tokeniser = new CharTokeniser(text);
  encodedData = tokeniser.encode(text);

  const config: ModelConfig = {
    vocabSize: tokeniser.vocabSize,
    blockSize: archConfig.blockSize,
    dModel: archConfig.dModel,
    nHeads: archConfig.nHeads,
    nLayers: archConfig.nLayers,
  };

  model = initModel(config);
  adam = initAdam(model);
  step = 0;
  cursor = 0;

  // Reset milestones
  milestones = {
    lossBelow3: false,
    lossBelow1: false,
    lossBelow05: false,
    firstRealWord: false,
    recognisablePhrase: false,
  };

  $('param-count').textContent = `${paramCount(model).toLocaleString()} parameters`;
  bigramHeatmap.setVocab(tokeniser.idxToChar);
  embeddingVis.setVocab(tokeniser.idxToChar);
  updateCounters();
}

function resetModel(): void {
  const wasTraining = isTraining;
  pauseTraining();
  lossCurve.clear();
  tokenProbs.clear();
  attentionVis.clear();
  typewriter.clear();
  bigramHeatmap.clear();
  embeddingVis.clear();
  buildModel();
  if (wasTraining) startTraining();
}

function resetAll(): void {
  pauseTraining();
  lossCurve.clear();
  tokenProbs.clear();
  attentionVis.clear();
  typewriter.clear();
  bigramHeatmap.clear();
  embeddingVis.clear();
  step = 0;
  cursor = 0;
  updateCounters();
  buildModel();
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

function startTraining(): void {
  if (!model) buildModel();
  if (!model) return; // buildModel may have failed (text too short)
  isTraining = true;
  $('btn-train').classList.add('hidden');
  $('btn-pause').classList.remove('hidden');
  requestAnimationFrame(tick);
}

function pauseTraining(): void {
  isTraining = false;
  $('btn-pause').classList.add('hidden');
  $('btn-train').classList.remove('hidden');
}

function tick(): void {
  if (!isTraining) return;

  let lastLoss = 0;

  for (let i = 0; i < stepsPerFrame; i++) {
    // Ensure cursor doesn't overrun
    if (cursor + archConfig.blockSize + 1 > encodedData.length) {
      cursor = 0;
    }

    const result = trainStep(model, adam, adamConfig, encodedData, cursor);
    cursor = (cursor + archConfig.blockSize) % (encodedData.length - archConfig.blockSize - 1);
    step++;
    lastLoss = result.loss;
    lossCurve.addPoint(result.loss);

    // Update attention vis every 50 steps
    if (step % 50 === 0) {
      attentionVis.animateTo(result.attentionWeights);
      // Set chars for attention display
      const contextStart = Math.max(0, cursor - archConfig.blockSize);
      const contextChars = tokeniser.decode(
        encodedData.slice(contextStart, contextStart + Math.min(32, archConfig.blockSize))
      ).split('');
      attentionVis.setChars(contextChars);
    }

    // Update token probs + bigram heatmap every frame (using last forward pass)
    if (i === stepsPerFrame - 1) {
      const probeInput = encodedData.slice(cursor, cursor + archConfig.blockSize);
      if (probeInput.length === archConfig.blockSize) {
        const probeState = forward(probeInput, model);
        tokenProbs.update(probeState.logits, probeState.logits.rows - 1, tokeniser.idxToChar);

        // Feed bigram heatmap: softmax the logits to get per-position probability distributions
        const sm = softmax(probeState.logits);
        const rows: Float32Array[] = [];
        for (let r = 0; r < sm.rows; r++) {
          rows.push(sm.data.slice(r * sm.cols, (r + 1) * sm.cols));
        }
        bigramHeatmap.update(probeInput, rows);
      }
    }
  }

  // Check loss milestones
  checkMilestones(lastLoss);

  // Update embedding vis every 500 steps (PCA is relatively expensive)
  if (step % 500 < stepsPerFrame) {
    embeddingVis.update(model.tokenEmb);
  }

  // Generate sample every 200 steps
  if (step % 200 < stepsPerFrame) {
    // Use a full blockSize seed from the training text for better context
    const seedLen = Math.min(archConfig.blockSize, encodedData.length - 1);
    const seedStart = Math.floor(Math.random() * (encodedData.length - seedLen));
    const seed = encodedData.slice(seedStart, seedStart + seedLen);

    const { tokens, meta } = generateWithMetadata(
      model, tokeniser, seed, 80, temperature
    );
    const text = tokeniser.decode(tokens.slice(seedLen)); // exclude seed
    typewriter.addSample(step, text, meta);
    lossCurve.addMilestone(step);

    // Check text milestones
    checkMilestones(lastLoss, text);
  }

  // Render vis
  lossCurve.render();
  tokenProbs.render();
  attentionVis.render();
  bigramHeatmap.render();
  embeddingVis.render();
  updateBigramInsights();
  updateAttentionInsights();
  updateEmbeddingInsights();
  updateCounters();

  requestAnimationFrame(tick);
}

let lastInsightsUpdate = 0;
function updateBigramInsights(): void {
  // Only update every 100 steps to avoid DOM thrashing
  if (step - lastInsightsUpdate < 100) return;
  lastInsightsUpdate = step;

  const container = $('bigram-insights');
  const patterns = bigramHeatmap.getTopPatterns(6);

  if (patterns.length === 0) {
    container.innerHTML = '<span style="opacity:0.4">Training will reveal patterns...</span>';
    return;
  }

  const maxProb = patterns[0]?.prob ?? 1;
  container.innerHTML = patterns.map(p => {
    const barW = Math.round((p.prob / maxProb) * 50);
    return `<div class="insight"><span class="insight-bar" style="width:${barW}px"></span><span class="insight-text">${p.sentence}</span></div>`;
  }).join('');
}

let lastAttentionInsightsUpdate = 0;
function updateAttentionInsights(): void {
  if (step - lastAttentionInsightsUpdate < 100) return;
  lastAttentionInsightsUpdate = step;

  const container = $('attention-insights');
  const insights = attentionVis.getInsights();

  if (insights.length === 0) {
    container.innerHTML = '<span style="opacity:0.4">Training will reveal attention patterns...</span>';
    return;
  }

  container.innerHTML = insights.map(text => {
    const isSubline = text.startsWith('  ');
    const cls = isSubline ? 'insight insight-sub' : 'insight';
    return `<div class="${cls}"><span class="insight-text">${text.trim()}</span></div>`;
  }).join('');
}

let lastEmbeddingInsightsUpdate = 0;
function updateEmbeddingInsights(): void {
  if (step - lastEmbeddingInsightsUpdate < 500) return;
  lastEmbeddingInsightsUpdate = step;

  const container = $('embedding-insights');
  if (!container) return;

  const insights = embeddingVis.getInsights();

  if (insights.length === 0) {
    container.innerHTML = '<span style="opacity:0.4">Training will reveal character groupings...</span>';
    return;
  }

  container.innerHTML = insights.map(text => {
    return `<div class="insight"><span class="insight-text">${text}</span></div>`;
  }).join('');
}

function updateCounters(): void {
  $('step-count').textContent = step.toLocaleString();
  const lastLoss = lossCurve['rawData']?.[lossCurve['rawData'].length - 1];
  $('loss-value').textContent = lastLoss !== undefined ? lastLoss.toFixed(3) : '—';
}

// ---------------------------------------------------------------------------
// Export model
// ---------------------------------------------------------------------------

function serializeTensor(t: Tensor): { rows: number; cols: number; data: number[] } {
  return { rows: t.rows, cols: t.cols, data: Array.from(t.data) };
}

function serializeBlock(b: BlockWeights) {
  return {
    ln1Gamma: serializeTensor(b.ln1Gamma),
    ln1Beta: serializeTensor(b.ln1Beta),
    Wq: serializeTensor(b.Wq), Bq: serializeTensor(b.Bq),
    Wk: serializeTensor(b.Wk), Bk: serializeTensor(b.Bk),
    Wv: serializeTensor(b.Wv), Bv: serializeTensor(b.Bv),
    Wo: serializeTensor(b.Wo), Bo: serializeTensor(b.Bo),
    ln2Gamma: serializeTensor(b.ln2Gamma),
    ln2Beta: serializeTensor(b.ln2Beta),
    ff1W: serializeTensor(b.ff1W), ff1B: serializeTensor(b.ff1B),
    ff2W: serializeTensor(b.ff2W), ff2B: serializeTensor(b.ff2B),
  };
}

function exportModel(): void {
  if (!model || !tokeniser) return;

  const exported = {
    signetLLM: '1.0',
    exportedAt: new Date().toISOString(),
    step,
    trainingText: ($('training-text') as HTMLTextAreaElement).value,
    config: model.config,
    vocab: Object.fromEntries(tokeniser.idxToChar),
    weights: {
      tokenEmb: serializeTensor(model.tokenEmb),
      posEmb: serializeTensor(model.posEmb),
      blocks: model.blocks.map(serializeBlock),
      lnFGamma: serializeTensor(model.lnFGamma),
      lnFBeta: serializeTensor(model.lnFBeta),
      outputW: serializeTensor(model.outputW),
      outputB: serializeTensor(model.outputB),
    },
  };

  const json = JSON.stringify(exported);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `signet-llm-${step}steps.json`;
  a.click();

  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initDOM();
  buildModel();
  // Auto-start training so visitors see something happening immediately
  startTraining();
});
