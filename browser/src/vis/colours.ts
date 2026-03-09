/**
 * Shared colour palette and mapping functions.
 */

export const COLOURS = {
  bg: '#0a0a0f',
  panelBg: '#12121a',
  panelBorder: 'rgba(255,255,255,0.08)',
  blue: '#4fc3f7',
  blueDim: 'rgba(79, 195, 247, 0.3)',
  amber: '#ffb74d',
  amberDim: 'rgba(255, 183, 77, 0.3)',
  text: '#e0e0e0',
  textDim: '#666',
  white: '#ffffff',
  grid: 'rgba(255,255,255,0.05)',
};

export const HEAD_COLOURS = [
  COLOURS.blue,
  COLOURS.amber,
  '#81c784',  // green
  '#ce93d8',  // purple
];

/** Map confidence (0–1) to a CSS colour string. */
export function confidenceColour(confidence: number): string {
  if (confidence > 0.8) return COLOURS.white;
  if (confidence > 0.3) return COLOURS.blue;
  return COLOURS.textDim;
}

/** Map a value (0–1) to a blue→white→orange diverging scale. */
export function attentionColour(weight: number): string {
  const r = Math.round(79 + (255 - 79) * weight);
  const g = Math.round(195 + (183 - 195) * weight);
  const b = Math.round(247 + (77 - 247) * weight);
  return `rgb(${r},${g},${b})`;
}
