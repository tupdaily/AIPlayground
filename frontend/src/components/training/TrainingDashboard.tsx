"use client";

import { useTrainingStore } from "@/store/trainingStore";
import { useGraphStore } from "@/store/graphStore";
import { serializeGraph } from "@/lib/serialization";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

const DATASETS = [
  { id: "mnist", label: "MNIST", description: "Handwritten digits, 28x28" },
  {
    id: "fashion_mnist",
    label: "Fashion-MNIST",
    description: "Clothing items, 28x28",
  },
  {
    id: "cifar10",
    label: "CIFAR-10",
    description: "Color images 32x32, 10 classes",
  },
];

export default function TrainingDashboard() {
  const { nodes, edges } = useGraphStore();
  const {
    status,
    datasetId,
    config,
    metrics,
    currentEpoch,
    currentBatchLoss,
    errorMessage,
    setDataset,
    setConfig,
    startTraining,
    addMetric,
    setBatchLoss,
    setCompleted,
    setError,
    setStopped,
    reset,
    ws,
    setWs,
  } = useTrainingStore();

  const handleStartTraining = async () => {
    try {
      const graph = serializeGraph(nodes, edges);
      const res = await fetch(`${API_BASE}/api/training/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          graph,
          dataset_id: datasetId,
          training_config: config,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        setError(err.detail || "Failed to start training");
        return;
      }

      const { job_id } = await res.json();
      startTraining(job_id);

      // Connect WebSocket for real-time updates
      const socket = new WebSocket(`${WS_BASE}/ws/training/${job_id}`);
      setWs(socket);

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case "batch":
            setBatchLoss(msg.loss, msg.epoch);
            break;
          case "epoch":
            addMetric({
              epoch: msg.epoch,
              train_loss: msg.train_loss,
              val_loss: msg.val_loss,
              train_acc: msg.train_acc,
              val_acc: msg.val_acc,
              elapsed_sec: msg.elapsed_sec,
            });
            break;
          case "completed":
            setCompleted();
            socket.close();
            break;
          case "error":
            setError(msg.message);
            socket.close();
            break;
        }
      };

      socket.onerror = () => {
        setError("WebSocket connection failed");
      };
    } catch {
      setError("Failed to connect to backend");
    }
  };

  const handleStop = () => {
    if (ws) {
      ws.send(JSON.stringify({ type: "stop" }));
      ws.close();
      setWs(null);
    }
    setStopped();
  };

  const isIdle = status === "idle" || status === "completed" || status === "error" || status === "stopped";

  return (
    <div className="w-80 bg-gray-50 border-l border-gray-200 overflow-y-auto flex-shrink-0 flex flex-col">
      <div className="p-4 border-b">
        <h2 className="font-bold text-sm text-gray-700 mb-3">Training</h2>

        {/* Dataset selector */}
        <label className="block text-xs font-medium text-gray-600 mb-1">
          Dataset
        </label>
        <select
          value={datasetId}
          onChange={(e) => setDataset(e.target.value)}
          disabled={!isIdle}
          className="w-full px-2 py-1 border border-gray-300 rounded text-sm mb-3 disabled:opacity-50"
        >
          {DATASETS.map((d) => (
            <option key={d.id} value={d.id}>
              {d.label} - {d.description}
            </option>
          ))}
        </select>

        {/* Hyperparameters */}
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Epochs
            </label>
            <input
              type="number"
              value={config.epochs}
              min={1}
              max={100}
              onChange={(e) =>
                setConfig({ epochs: parseInt(e.target.value) || 1 })
              }
              disabled={!isIdle}
              className="w-full px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Batch Size
            </label>
            <input
              type="number"
              value={config.batch_size}
              min={1}
              onChange={(e) =>
                setConfig({ batch_size: parseInt(e.target.value) || 1 })
              }
              disabled={!isIdle}
              className="w-full px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Learning Rate
            </label>
            <input
              type="number"
              value={config.learning_rate}
              min={0.0001}
              max={1}
              step={0.0001}
              onChange={(e) =>
                setConfig({
                  learning_rate: parseFloat(e.target.value) || 0.001,
                })
              }
              disabled={!isIdle}
              className="w-full px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Optimizer
            </label>
            <select
              value={config.optimizer}
              onChange={(e) =>
                setConfig({
                  optimizer: e.target.value as "adam" | "sgd" | "adamw",
                })
              }
              disabled={!isIdle}
              className="w-full px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
            >
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
              <option value="adamw">AdamW</option>
            </select>
          </div>
        </div>

        {/* Action buttons */}
        {isIdle ? (
          <button
            onClick={handleStartTraining}
            disabled={nodes.length === 0}
            className="w-full px-3 py-2 bg-green-500 text-white text-sm font-semibold rounded hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Start Training
          </button>
        ) : (
          <button
            onClick={handleStop}
            className="w-full px-3 py-2 bg-red-500 text-white text-sm font-semibold rounded hover:bg-red-600 transition-colors"
          >
            Stop Training
          </button>
        )}

        {status === "error" && (
          <div className="mt-2 text-xs text-red-600 bg-red-50 p-2 rounded">
            {errorMessage}
          </div>
        )}

        {status === "completed" && (
          <div className="mt-2 text-xs text-green-600 bg-green-50 p-2 rounded">
            Training completed!
            <button
              onClick={reset}
              className="ml-2 underline hover:no-underline"
            >
              Reset
            </button>
          </div>
        )}
      </div>

      {/* Live metrics */}
      {(status === "running" || metrics.length > 0) && (
        <div className="p-4 flex-1">
          {/* Progress */}
          {status === "running" && (
            <div className="mb-3">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>
                  Epoch {currentEpoch} / {config.epochs}
                </span>
                {currentBatchLoss !== null && (
                  <span>Loss: {currentBatchLoss.toFixed(4)}</span>
                )}
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{
                    width: `${(currentEpoch / config.epochs) * 100}%`,
                  }}
                />
              </div>
            </div>
          )}

          {/* Loss chart */}
          {metrics.length > 0 && (
            <>
              <h3 className="text-xs font-semibold text-gray-600 mb-2">
                Loss
              </h3>
              <div className="h-40 mb-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ fontSize: 11 }}
                      formatter={(value) => typeof value === "number" ? value.toFixed(4) : value}
                    />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Line
                      type="monotone"
                      dataKey="train_loss"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      name="Train"
                    />
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      name="Validation"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Accuracy chart */}
              {metrics[0]?.val_acc !== undefined && (
                <>
                  <h3 className="text-xs font-semibold text-gray-600 mb-2">
                    Accuracy
                  </h3>
                  <div className="h-40">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={metrics}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                        <YAxis
                          tick={{ fontSize: 10 }}
                          domain={[0, 1]}
                          tickFormatter={(v) =>
                            `${(Number(v) * 100).toFixed(0)}%`
                          }
                        />
                        <Tooltip
                          contentStyle={{ fontSize: 11 }}
                          formatter={(value) =>
                            typeof value === "number" ? `${(value * 100).toFixed(1)}%` : value
                          }
                        />
                        <Legend wrapperStyle={{ fontSize: 10 }} />
                        <Line
                          type="monotone"
                          dataKey="train_acc"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          dot={false}
                          name="Train"
                        />
                        <Line
                          type="monotone"
                          dataKey="val_acc"
                          stroke="#ef4444"
                          strokeWidth={2}
                          dot={false}
                          name="Validation"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
