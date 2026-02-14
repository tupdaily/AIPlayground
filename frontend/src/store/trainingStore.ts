import { create } from "zustand";
import {
  TrainingConfig,
  EpochMetric,
  TrainingStatus,
} from "@/types/training";

interface TrainingState {
  status: TrainingStatus;
  jobId: string | null;
  datasetId: string;
  config: TrainingConfig;
  metrics: EpochMetric[];
  currentEpoch: number;
  currentBatchLoss: number | null;
  errorMessage: string | null;
  ws: WebSocket | null;

  setDataset: (id: string) => void;
  setConfig: (config: Partial<TrainingConfig>) => void;
  startTraining: (jobId: string) => void;
  addMetric: (metric: EpochMetric) => void;
  setBatchLoss: (loss: number, epoch: number) => void;
  setCompleted: () => void;
  setError: (message: string) => void;
  setStopped: () => void;
  reset: () => void;
  setWs: (ws: WebSocket | null) => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  status: "idle",
  jobId: null,
  datasetId: "mnist",
  config: {
    epochs: 10,
    batch_size: 64,
    learning_rate: 0.001,
    optimizer: "adam",
    train_split: 0.8,
  },
  metrics: [],
  currentEpoch: 0,
  currentBatchLoss: null,
  errorMessage: null,
  ws: null,

  setDataset: (id) => set({ datasetId: id }),

  setConfig: (partial) =>
    set((state) => ({ config: { ...state.config, ...partial } })),

  startTraining: (jobId) =>
    set({
      status: "running",
      jobId,
      metrics: [],
      currentEpoch: 0,
      currentBatchLoss: null,
      errorMessage: null,
    }),

  addMetric: (metric) =>
    set((state) => ({
      metrics: [...state.metrics, metric],
      currentEpoch: metric.epoch,
    })),

  setBatchLoss: (loss, epoch) =>
    set({ currentBatchLoss: loss, currentEpoch: epoch }),

  setCompleted: () => set({ status: "completed" }),

  setError: (message) => set({ status: "error", errorMessage: message }),

  setStopped: () => set({ status: "stopped" }),

  reset: () =>
    set({
      status: "idle",
      jobId: null,
      metrics: [],
      currentEpoch: 0,
      currentBatchLoss: null,
      errorMessage: null,
    }),

  setWs: (ws) => set({ ws }),
}));
