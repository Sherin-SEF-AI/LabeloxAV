// Deterministic color per ontology class_id, so each class reads as a consistent swatch across the
// canvas and the layers panel. The 47x hue step spreads adjacent ids far apart on the wheel.

export function classHue(id: number): number {
  return (id * 47) % 360;
}

export function classColor(id: number): string {
  return `hsl(${classHue(id)} 70% 62%)`;
}

export function classFill(id: number, alpha = 0.18): string {
  return `hsl(${classHue(id)} 70% 62% / ${alpha})`;
}
