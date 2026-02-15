"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { LiveTrainingState } from "./types";

interface LiveTrainingOverlayProps extends LiveTrainingState {
  totalEpochs?: number;
  embedded?: boolean;
}

export function LiveTrainingOverlay({
  status,
  metrics,
  lastMessage,
  totalEpochs = 10,
  totalBatches,
  latestBatch,
  embedded = false,
}: LiveTrainingOverlayProps) {
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const chartData = useMemo(
    () =>
      metrics.map((m) => ({
        epoch: m.epoch,
        train_loss: m.train_loss,
        val_loss: m.val_loss,
        train_acc: m.train_acc * 100,
        val_acc: m.val_acc * 100,
      })),
    [metrics]
  );

  const lossDomain = useMemo(() => {
    if (chartData.length === 0) return [0, 1] as [number, number];
    const losses = chartData.flatMap((d) => [d.train_loss, d.val_loss]);
    const min = Math.min(...losses);
    const max = Math.max(...losses);
    const padding = (max - min) * 0.05 || 0.1;
    return [Math.max(0, min - padding), max + padding] as [number, number];
  }, [chartData]);

  const accuracyDomain = useMemo((): [number, number] => {
    if (chartData.length === 0) return [0, 100];
    const accs = chartData.flatMap((d) => [d.train_acc, d.val_acc]);
    const min = Math.min(...accs);
    const max = Math.max(...accs);
    const padding = Math.max(5, (max - min) * 0.05);
    return [Math.max(0, min - padding), Math.min(100, max + padding)];
  }, [chartData]);

  if (status === "idle") return null;

  const isActive = status === "starting" || status === "running";
  const statusLabel =
    status === "starting"
      ? "Starting..."
      : status === "running"
        ? "Training..."
        : status === "completed"
          ? "Completed"
          : status === "stopped"
            ? "Stopped"
            : status === "error"
              ? "Error"
              : status;

  return (
    <div
      className={
        embedded
          ? "w-full rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] overflow-hidden"
          : "absolute top-4 left-4 z-20 w-72 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-lg overflow-hidden"
      }
    >
      {/* Header */}
      <div
        className={`px-3 py-2 border-b border-[var(--border)] flex items-center justify-between ${
          isActive ? "bg-[var(--accent-muted)]" : ""
        }`}
      >
        <span className={`text-[12px] font-semibold ${isActive ? "text-[var(--accent)]" : "text-[var(--foreground)]"}`}>
          {statusLabel}
        </span>
        {(latestBatch || latest) && (
          <span className="text-[10px] font-mono text-[var(--foreground-muted)]">
            {latestBatch
              ? `Epoch ${latestBatch.epoch} · batch ${latestBatch.batch}${typeof totalBatches === "number" ? ` / ${totalBatches}` : ""}`
              : latest
                ? `Epoch ${latest.epoch}${totalEpochs > 0 ? ` / ${totalEpochs}` : ""}`
                : null}
          </span>
        )}
      </div>

      {/* Metrics */}
      <div className="p-3 space-y-3">
        {/* Batch progress */}
        {latestBatch && (
          <div className="rounded-lg bg-[var(--surface)] border border-[var(--border)] px-3 py-2 space-y-1.5">
            <div className="text-[10px] font-medium text-[var(--foreground-muted)] uppercase tracking-wider">
              Current batch
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-[12px] font-semibold text-emerald-600">
                Loss {latestBatch.loss.toFixed(4)}
              </span>
              {typeof totalBatches === "number" && totalBatches > 0 && (
                <span className="text-[10px] text-[var(--foreground-muted)]">
                  {Math.round((latestBatch.batch / totalBatches) * 100)}%
                </span>
              )}
            </div>
            {typeof totalBatches === "number" && (
              <div className="h-1.5 w-full rounded-full bg-[var(--surface-elevated)] overflow-hidden">
                <div
                  className="h-full bg-indigo-500 rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.min(100, (latestBatch.batch / totalBatches) * 100)}%`,
                  }}
                />
              </div>
            )}
          </div>
        )}

        {latest ? (
          <>
            {/* Metric cards */}
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2">
                <div className="text-[9px] text-[var(--foreground-muted)] uppercase tracking-wider">Train Loss</div>
                <div className="text-[16px] font-bold text-[var(--accent)] mt-0.5">{latest.train_loss.toFixed(4)}</div>
              </div>
              <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2">
                <div className="text-[9px] text-[var(--foreground-muted)] uppercase tracking-wider">Val Loss</div>
                <div className="text-[16px] font-bold text-amber-600 mt-0.5">{latest.val_loss.toFixed(4)}</div>
              </div>
              <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2">
                <div className="text-[9px] text-[var(--foreground-muted)] uppercase tracking-wider">Train Acc</div>
                <div className="text-[16px] font-bold text-[var(--accent)] mt-0.5">{(latest.train_acc * 100).toFixed(1)}%</div>
              </div>
              <div className="bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2">
                <div className="text-[9px] text-[var(--foreground-muted)] uppercase tracking-wider">Val Acc</div>
                <div className="text-[16px] font-bold text-amber-600 mt-0.5">{(latest.val_acc * 100).toFixed(1)}%</div>
              </div>
            </div>

            {/* Charts */}
            {chartData.length >= 1 && (
              <div className="space-y-3 pt-2 border-t border-[var(--border)]">
                <div className="text-[10px] font-medium text-[var(--foreground-muted)] uppercase tracking-wider">
                  Metrics over epochs
                </div>
                <div className="space-y-4">
                  <div>
                    <div className="text-[10px] text-[var(--foreground-muted)] mb-1">Loss</div>
                    <div className="h-[80px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 40 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#F0F1F3" />
                          <XAxis dataKey="epoch" tick={{ fontSize: 9, fill: "#9CA3AF" }} tickLine={false} axisLine={{ stroke: "#E5E7EB" }} interval="preserveStartEnd" />
                          <YAxis tick={{ fontSize: 9, fill: "#9CA3AF" }} tickLine={false} axisLine={false} width={40} tickFormatter={(v: number) => v.toFixed(2)} domain={lossDomain} />
                          <Tooltip
                            contentStyle={{ backgroundColor: "var(--surface)", border: "1px solid #E5E7EB", borderRadius: "8px", fontSize: "11px", boxShadow: "0 4px 12px rgba(0,0,0,0.06)" }}
                            labelFormatter={(epoch) => `Epoch ${epoch}`}
                            formatter={(value?: number, name?: string) => [value != null ? value.toFixed(4) : "", name === "train_loss" ? "Train" : "Val"]}
                          />
                          <Line type="monotone" dataKey="train_loss" name="Train" stroke="#6366F1" strokeWidth={2} dot={false} isAnimationActive animationDuration={400} />
                          <Line type="monotone" dataKey="val_loss" name="Val" stroke="#F59E0B" strokeWidth={2} dot={false} isAnimationActive animationDuration={400} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-[var(--foreground-muted)] mb-1">Accuracy (%)</div>
                    <div className="h-[80px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 40 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#F0F1F3" />
                          <XAxis dataKey="epoch" tick={{ fontSize: 9, fill: "#9CA3AF" }} tickLine={false} axisLine={{ stroke: "#E5E7EB" }} interval="preserveStartEnd" />
                          <YAxis tick={{ fontSize: 9, fill: "#9CA3AF" }} tickLine={false} axisLine={false} width={40} tickFormatter={(v: number) => `${v}%`} domain={accuracyDomain} />
                          <Tooltip
                            contentStyle={{ backgroundColor: "var(--surface)", border: "1px solid #E5E7EB", borderRadius: "8px", fontSize: "11px", boxShadow: "0 4px 12px rgba(0,0,0,0.06)" }}
                            labelFormatter={(epoch) => `Epoch ${epoch}`}
                            formatter={(value?: number, name?: string) => [value != null ? `${value.toFixed(2)}%` : "", name === "train_acc" ? "Train" : "Val"]}
                          />
                          <Line type="monotone" dataKey="train_acc" name="Train" stroke="#6366F1" strokeWidth={2} dot={false} isAnimationActive animationDuration={400} />
                          <Line type="monotone" dataKey="val_acc" name="Val" stroke="#F59E0B" strokeWidth={2} dot={false} isAnimationActive animationDuration={400} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-4 text-[10px] text-[var(--foreground-muted)]">
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block w-3 h-0.5 bg-indigo-500 rounded" />
                    Train
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block w-3 h-0.5 bg-amber-500 rounded" />
                    Validation
                  </span>
                </div>
              </div>
            )}
          </>
        ) : status === "error" && lastMessage?.type === "error" ? (
          <div className="text-[12px] text-red-600 py-1">
            {(lastMessage.message as string) ?? "Training failed"}
          </div>
        ) : (
          <div className="text-[12px] text-[var(--foreground-muted)] py-1">
            {status === "starting" && !lastMessage
              ? "Connecting..."
              : status === "starting" && lastMessage?.type === "connected"
                ? "Building model & loading data..."
                : lastMessage?.type === "started"
                  ? `Device: ${(lastMessage.device as string) ?? "—"} · Waiting for first epoch...`
                  : "Waiting for first epoch..."}
          </div>
        )}
      </div>
    </div>
  );
}
