/**
 * Character-level tokeniser.
 * Builds vocabulary dynamically from training text.
 */

export class CharTokeniser {
  charToIdx: Map<string, number>;
  idxToChar: Map<number, string>;
  vocabSize: number;

  constructor(text: string) {
    const chars = [...new Set(text.split(''))].sort();
    this.charToIdx = new Map(chars.map((c, i) => [c, i]));
    this.idxToChar = new Map(chars.map((c, i) => [i, c]));
    this.vocabSize = chars.length;
  }

  encode(text: string): number[] {
    return text.split('').map(c => {
      const idx = this.charToIdx.get(c);
      if (idx === undefined) throw new Error(`Unknown character: "${c}"`);
      return idx;
    });
  }

  decode(tokens: number[]): string {
    return tokens.map(t => this.idxToChar.get(t) ?? '?').join('');
  }
}
