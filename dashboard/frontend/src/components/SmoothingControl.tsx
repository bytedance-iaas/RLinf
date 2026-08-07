/**
 * The smoothing slider, shared by the metrics view and the compare view.
 *
 * The scale is in points, not in a 0..1 "weight" like TensorBoard's: an RL operator
 * reasons in iterations ("average over ten iterations"), and a dimensionless 0.6
 * means nothing without knowing the series length. Zero is off and is the default,
 * because a smoothed curve hides exactly the spikes an RL run is read for -- the
 * user has to ask for it.
 */

export const SMOOTHING_STOPS = [0, 3, 5, 9, 15, 25, 49] as const;

export function SmoothingControl(props: { value: number; onChange: (value: number) => void }) {
  const index = Math.max(0, SMOOTHING_STOPS.indexOf(props.value as (typeof SMOOTHING_STOPS)[number]));

  return (
    <label className="control">
      <span>Smoothing</span>
      <input
        type="range"
        min={0}
        max={SMOOTHING_STOPS.length - 1}
        step={1}
        value={index}
        // Discrete stops rather than a continuous range: adjacent continuous
        // values are visually identical, so the extra precision only makes the
        // control harder to return to a known setting.
        onChange={(event) => props.onChange(SMOOTHING_STOPS[Number(event.target.value)] ?? 0)}
        aria-label="Smoothing window in points"
      />
      <span className="control-value">{props.value === 0 ? "off" : `${props.value}pt`}</span>
    </label>
  );
}
