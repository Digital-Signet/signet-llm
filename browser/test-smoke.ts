/**
 * Quick smoke test: does the model train and loss decrease?
 */
import { initModel, paramCount, forward, ModelConfig } from './src/engine/model';
import { initAdam, trainStep, ADAM_DEFAULTS } from './src/engine/train';
import { CharTokeniser } from './src/engine/tokeniser';
import { generate } from './src/engine/generate';

const text = "To be or not to be, that is the question. To be or not to be, that is the question. To be or not to be, that is the question.";
const tokeniser = new CharTokeniser(text);
const encoded = tokeniser.encode(text);

const config: ModelConfig = {
  vocabSize: tokeniser.vocabSize,
  blockSize: 32,
  dModel: 32,
  nHeads: 2,
  nLayers: 2,
};

console.log(`Vocab size: ${tokeniser.vocabSize}`);
console.log(`Text length: ${encoded.length} tokens`);

const model = initModel(config);
const adam = initAdam(model);

console.log(`Parameters: ${paramCount(model).toLocaleString()}`);
console.log('');

let cursor = 0;
const losses: number[] = [];

for (let step = 0; step < 500; step++) {
  if (cursor + config.blockSize + 1 > encoded.length) cursor = 0;

  const result = trainStep(model, adam, { ...ADAM_DEFAULTS }, encoded, cursor);
  cursor = (cursor + config.blockSize) % (encoded.length - config.blockSize - 1);
  losses.push(result.loss);

  if (step % 100 === 0) {
    console.log(`Step ${step}: loss = ${result.loss.toFixed(4)}`);
  }
}

console.log('');
console.log(`Loss went from ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`);
console.log(`Loss decreased: ${losses[losses.length - 1] < losses[0] ? 'YES' : 'NO'}`);
console.log('');

// Generate some text
const seed = encoded.slice(0, 8);
const generated = generate(model, seed, 50, 0.8);
console.log(`Generated: "${tokeniser.decode(generated)}"`);
