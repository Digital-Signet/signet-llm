# Signet LLM

A tiny GPT-style transformer that trains in your browser. No frameworks, no libraries — just TypeScript, matrix multiplication, and gradient descent.

**[Try it live](https://llm.digitalsignet.com)**

## What is this?

A decoder-only transformer language model built from scratch. It learns character-level patterns from any text you give it and generates new text in real time. Everything runs client-side — no server, no API calls.

Pick some training text (Shakespeare, nursery rhymes, JSON), hit Train, and watch the model go from random gibberish to recognisable words and phrases.

## Architecture

| Parameter | Default |
|-----------|---------|
| Layers | 2 |
| Attention heads | 2 |
| Embedding dimension | 32 |
| Context window | 64 characters |
| Total parameters | ~31,000 |
| Bundle size | ~40KB minified |

All configurable from the UI. The model uses:

- Character-level tokenisation
- Learned position embeddings
- Multi-head causal self-attention
- Layer normalisation (pre-norm)
- Feed-forward network with ReLU activation
- Manual backpropagation (no autograd)
- Adam optimiser with gradient clipping

## Visualisations

- **Typewriter** — generated text with per-character confidence colouring
- **Loss curve** — training error over time with EMA smoothing
- **Token probabilities** — next-character prediction confidence with entropy
- **Character embeddings** — 2D PCA projection showing learned similarity
- **Bigram heatmap** — learned character-pair associations
- **Attention arcs** — which characters the model attends to

## Project structure

```
browser/
  src/
    engine/       # Tensor ops, model, training, generation
      tensor.ts   # Float32Array-backed tensor with manual gradients
      model.ts    # Transformer architecture (init + forward pass)
      train.ts    # Backward pass, Adam optimiser, gradient clipping
      generate.ts # Autoregressive sampling with metadata
      tokeniser.ts
    vis/          # Canvas-based visualisations
    app.ts        # Main application wiring
  dist/           # Static site (deployed as-is)
```

## Run locally

```bash
cd browser
npm install
npm run dev
```

Opens at `http://localhost:8080`.

## Build

```bash
npm run build
```

Produces `dist/app.js` — the entire application in a single file.

## Export

Trained models can be exported as JSON files containing all weights, config, and vocabulary.

## License

MIT

---

Built by [Digital Signet](https://digitalsignet.com)
