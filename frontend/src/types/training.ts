export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: "adam" | "sgd" | "adamw";
  train_split: number;
}

export interface EpochMetric {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc?: number;
  val_acc?: number;
  elapsed_sec: number;
}

export type TrainingStatus =
  | "idle"
  | "starting"
  | "running"
  | "completed"
  | "error"
  | "stopped";

export interface TrainingMessage {
  type: "started" | "batch" | "epoch" | "completed" | "error";
  [key: string]: unknown;
}
