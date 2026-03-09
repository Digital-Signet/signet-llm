/**
 * Run 30K steps, log loss every 100 steps, analyze spikes.
 */
import { initModel, paramCount, forward, ModelConfig } from './src/engine/model';
import { initAdam, trainStep, ADAM_DEFAULTS } from './src/engine/train';
import { CharTokeniser } from './src/engine/tokeniser';
import * as fs from 'fs';

// Use the Shakespeare preset
const text = `To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.`;

const tokeniser = new CharTokeniser(text);
const encoded = tokeniser.encode(text);

const config: ModelConfig = {
  vocabSize: tokeniser.vocabSize,
  blockSize: 64,
  dModel: 32,
  nHeads: 2,
  nLayers: 2,
};

const model = initModel(config);
const adam = initAdam(model);

console.log(`Vocab: ${tokeniser.vocabSize}, Tokens: ${encoded.length}, Params: ${paramCount(model)}`);

let cursor = 0;
const rows: string[] = ['step,loss'];
const losses: number[] = [];

for (let step = 0; step < 30000; step++) {
  if (cursor + config.blockSize + 1 > encoded.length) cursor = 0;

  const result = trainStep(model, adam, { ...ADAM_DEFAULTS }, encoded, cursor);
  cursor = (cursor + config.blockSize) % (encoded.length - config.blockSize - 1);
  losses.push(result.loss);

  if (step % 100 === 0) {
    rows.push(`${step},${result.loss}`);
  }

  if (step % 5000 === 0) {
    console.log(`Step ${step}: loss = ${result.loss.toFixed(4)}`);
  }
}

// Write CSV
fs.writeFileSync('loss-data.csv', rows.join('\n'));
console.log(`\nWrote ${rows.length} rows to loss-data.csv`);

// Analyze spikes
console.log('\n=== Spike Analysis ===');
const windowSize = 500;
for (let start = 0; start < losses.length; start += 5000) {
  const window = losses.slice(start, start + 5000);
  const avg = window.reduce((a, b) => a + b, 0) / window.length;
  const max = Math.max(...window);
  const min = Math.min(...window);
  const spikes = window.filter(l => l > avg * 2).length;
  console.log(`Steps ${start}-${start + 5000}: avg=${avg.toFixed(4)} min=${min.toFixed(4)} max=${max.toFixed(4)} spikes(>2x avg)=${spikes}`);
}

// Check last 5000 steps specifically
const last5k = losses.slice(-5000);
const avgLast = last5k.reduce((a, b) => a + b, 0) / last5k.length;
const bigSpikes = last5k.filter(l => l > avgLast * 3);
console.log(`\nLast 5K: avg=${avgLast.toFixed(4)}`);
console.log(`Big spikes (>3x avg): ${bigSpikes.length}`);
if (bigSpikes.length > 0) {
  console.log(`Spike values: ${bigSpikes.slice(0, 10).map(s => s.toFixed(4)).join(', ')}`);
}

// Check if cursor wrapping causes spikes
console.log('\n=== Cursor Pattern ===');
let testCursor = 0;
const cursorValues: number[] = [];
for (let i = 0; i < 200; i++) {
  if (testCursor + config.blockSize + 1 > encoded.length) testCursor = 0;
  cursorValues.push(testCursor);
  testCursor = (testCursor + config.blockSize) % (encoded.length - config.blockSize - 1);
}
const uniqueCursors = new Set(cursorValues).size;
console.log(`Unique cursor positions in 200 steps: ${uniqueCursors}`);
console.log(`Text length: ${encoded.length}, Block size: ${config.blockSize}`);
console.log(`First 20 cursors: ${cursorValues.slice(0, 20).join(', ')}`);
