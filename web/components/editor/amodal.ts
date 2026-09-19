/** Props for the dashed rectangle that shows an object's whole extent.

    No fill and no perfect-draw. Konva draws a shape that has a fill, a stroke and an opacity below 1 through
    an offscreen buffer the size of the layer, and fill="transparent" still counts as a fill. The stage
    mounts at 0 by 0 until its container is measured, so that buffer was a zero sized canvas and drawImage
    threw: any frame carrying an amodal box crashed the editor the moment it opened. The rectangle is an
    outline that is never hit-tested, so a fill never did anything. */
export function amodalRectProps(b: number[], color: string, s: number) {
  return {
    listening: false, perfectDrawEnabled: false,
    x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1],
    stroke: color, strokeWidth: 1 / s, opacity: 0.55, dash: [8 / s, 5 / s],
  };
}
