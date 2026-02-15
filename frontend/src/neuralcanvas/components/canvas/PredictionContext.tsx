"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";

interface PredictionContextValue {
  /** Last predicted class index (argmax of logits). Null when no prediction yet. */
  predictedClassIndex: number | null;
  setPredictedClassIndex: (index: number | null) => void;
}

const PredictionContext = createContext<PredictionContextValue>({
  predictedClassIndex: null,
  setPredictedClassIndex: () => {},
});

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [predictedClassIndex, setPredictedClassIndex] = useState<number | null>(null);
  return (
    <PredictionContext.Provider value={{ predictedClassIndex, setPredictedClassIndex }}>
      {children}
    </PredictionContext.Provider>
  );
}

export function usePrediction() {
  return useContext(PredictionContext);
}
