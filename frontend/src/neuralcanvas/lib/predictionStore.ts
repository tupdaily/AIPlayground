/**
 * Shared store for the last predicted class index (Model → Display).
 * Used so the Display block always receives the prediction even when
 * React Flow renders nodes outside the PredictionContext tree.
 */

type Listener = (classIndex: number | null) => void;

let value: number | null = null;
const listeners = new Set<Listener>();

export function getPrediction(): number | null {
  return value;
}

export function setPrediction(classIndex: number | null): void {
  value = classIndex;
  listeners.forEach((fn) => fn(classIndex));
}

/** Subscribe to prediction updates; returns unsubscribe. Calls listener with current value on subscribe. */
export function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  listener(value);
  return () => listeners.delete(listener);
}
