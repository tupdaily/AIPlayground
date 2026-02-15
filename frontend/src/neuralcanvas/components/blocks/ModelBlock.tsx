"use client";

/**
 * Model block — run inference with a trained model.
 * Connect a data source (Board, Input Space) to the input; connect output to Display to see predictions.
 */

import { memo, useCallback, useEffect, useMemo, useState } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { useReactFlow, useEdges, useStore } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";
import { usePlaygroundId } from "@/neuralcanvas/components/canvas/PlaygroundIdContext";
import { usePrediction } from "@/neuralcanvas/components/canvas/PredictionContext";
import { useShapes } from "@/neuralcanvas/components/canvas/ShapeContext";
import { getShapeLabel } from "@/neuralcanvas/lib/shapeEngine";
import { listPlaygroundModels, runInference, type TrainedModel } from "@/neuralcanvas/lib/modelsApi";
import { BLOCK_REGISTRY } from "@/neuralcanvas/lib/blockRegistry";
import { setPrediction } from "@/neuralcanvas/lib/predictionStore";

const MODEL_COLOR = "#10B981";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Model block SVG: chip/inference symbol */
function ModelViz() {
  return (
    <svg width={140} height={36} viewBox="0 0 140 36">
      <rect x="12" y="6" width="32" height="24" rx="3" fill={`${MODEL_COLOR}25`} stroke={MODEL_COLOR} strokeWidth="1.2" />
      <circle cx="20" cy="14" r="2" fill={MODEL_COLOR} opacity={0.9} />
      <circle cx="28" cy="14" r="2" fill={MODEL_COLOR} opacity={0.9} />
      <circle cx="20" cy="22" r="2" fill={MODEL_COLOR} opacity={0.9} />
      <circle cx="28" cy="22" r="2" fill={MODEL_COLOR} opacity={0.9} />
      <path d="M46 18h20M56 12v12" stroke={MODEL_COLOR} strokeWidth="1.2" opacity={0.7} strokeLinecap="round" />
      <rect x="78" y="8" width="44" height="20" rx="2" fill={`${MODEL_COLOR}15`} stroke={MODEL_COLOR} strokeWidth="1" />
      <text x="100" y="22" fontSize="8" fill={MODEL_COLOR} fontWeight="600" textAnchor="middle">Out</text>
    </svg>
  );
}

function dataUrlToFile(dataUrl: string, filename = "image.png"): Promise<File> {
  return fetch(dataUrl)
    .then((r) => r.blob())
    .then((blob) => new File([blob], filename, { type: blob.type || "image/png" }));
}

function ModelBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const params = data?.params ?? {};
  const { shapes } = useShapes();
  const { setNodes, getNode } = useReactFlow();
  const edges = useEdges();
  const playgroundId = usePlaygroundId();
  const { setPredictedClassIndex } = usePrediction();
  const result = shapes.get(id);
  const outLabel = getShapeLabel(result?.outputShape ?? null);

  const [models, setModels] = useState<TrainedModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<string | null>(null);

  const selectedModelId = (params.model_id as string) || null;

  const sourceNode = useMemo(() => {
    const edge = edges.find((e) => e.target === id);
    if (!edge) return null;
    return getNode(edge.source);
  }, [id, edges, getNode]);

  // Subscribe to the source node's data so we re-render when Board (or Input Space) captures new image
  const sourcePayload = useStore(
    useCallback(
      (state) => {
        const edge = state.edges.find((e: { target: string }) => e.target === id);
        if (!edge) return null;
        const node = state.nodes.find((n: Node) => n.id === edge.source);
        const p = node?.data?.params as Record<string, unknown> | undefined;
        const payload = p?.custom_data_payload;
        return typeof payload === "string" ? payload : null;
      },
      [id],
    ),
  );

  const sourceBlockLabel = sourceNode
    ? (BLOCK_REGISTRY[sourceNode.type as keyof typeof BLOCK_REGISTRY]?.label ?? sourceNode.type)
    : null;

  const canGetInputFromSource = sourcePayload !== null && sourcePayload.length > 0 && sourcePayload.startsWith("data:");

  useEffect(() => {
    if (!playgroundId) return;
    setLoading(true);
    listPlaygroundModels(playgroundId)
      .then(setModels)
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [playgroundId]);

  const handleModelChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value || "";
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prev = (n.data?.params && typeof n.data.params === "object") ? (n.data.params as Record<string, number | string>) : {};
          return { ...n, data: { ...n.data, params: { ...prev, model_id: value } } };
        }),
      );
      setError(null);
      setLastResult(null);
    },
    [id, setNodes],
  );

  const handleRun = useCallback(async () => {
    if (!selectedModelId) {
      setError("Select a model");
      return;
    }
    if (!canGetInputFromSource) {
      setError("Connect Board or Input Space and add input");
      return;
    }

    const dataUrl = sourcePayload ?? (sourceNode?.data?.params as Record<string, unknown> | undefined)?.custom_data_payload as string | undefined;
    if (!dataUrl?.startsWith("data:")) {
      setError("No image in connected block");
      return;
    }

    setRunning(true);
    setError(null);
    setLastResult(null);

    try {
      const file = await dataUrlToFile(dataUrl);
      const response = await runInference(selectedModelId, file, "image");
      // Parse logits: backend returns { output: number[][] }; support 1D or nested as fallback
      const raw = (response as { output?: unknown }).output;
      const flat = (arr: unknown): number[] => {
        if (arr == null) return [];
        if (Array.isArray(arr) && arr.length > 0 && typeof arr[0] === "number") return arr as number[];
        if (Array.isArray(arr) && arr.length > 0) return flat(arr[0]);
        return [];
      };
      const logits = flat(raw);
      if (logits.length > 0 && logits.every((x) => typeof x === "number")) {
        let maxIdx = 0;
        let maxVal = logits[0];
        for (let i = 1; i < logits.length; i++) {
          if (logits[i] > maxVal) {
            maxVal = logits[i];
            maxIdx = i;
          }
        }
        setPredictedClassIndex(maxIdx);
        setPrediction(maxIdx); // Update shared store so Display block can show the prediction
        setLastResult(`Class ${maxIdx}`);
      } else {
        setPredictedClassIndex(null);
        setPrediction(null);
        const badMsg = raw == null ? "No output" : (Array.isArray(raw) ? `Bad output (${raw.length})` : "Bad output");
        setLastResult(badMsg);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Inference failed";
      setError(msg);
      setPredictedClassIndex(null);
      setPrediction(null);
    } finally {
      setRunning(false);
    }
  }, [selectedModelId, canGetInputFromSource, sourcePayload, sourceNode, setPredictedClassIndex]);

  return (
    <BaseBlock id={id} blockType="Model" params={params} selected={!!selected} data={data}>
      <ModelViz />
      <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] p-2 mt-1.5 space-y-2">
        {!playgroundId ? (
          <p className="text-[11px] text-[var(--foreground-muted)]">Save playground to use models</p>
        ) : (
          <>
            <div>
              <label className="block text-[10px] font-medium text-[var(--foreground-secondary)] mb-0.5">Model</label>
              <select
                value={selectedModelId || ""}
                onChange={handleModelChange}
                disabled={loading}
                className="w-full px-2 py-1.5 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-[11px] text-[var(--foreground)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
              >
                <option value="">Choose model...</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} {m.final_accuracy != null ? `(${(m.final_accuracy * 100).toFixed(0)}%)` : ""}
                  </option>
                ))}
              </select>
            </div>
            <p className="text-[10px] text-[var(--foreground-muted)]">
              Input: {sourceBlockLabel
                ? `${sourceBlockLabel}${canGetInputFromSource ? " ✓" : sourceNode?.type === "Board" ? " — draw then release to capture" : " (no data)"}`
                : "Not connected"}
            </p>
            <button
              type="button"
              onClick={handleRun}
              disabled={running || !selectedModelId || !canGetInputFromSource}
              className="w-full flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg bg-[var(--accent)] text-white text-[11px] font-medium hover:bg-[var(--accent-hover)] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {running ? "Running…" : "Run"}
            </button>
            {error && <p className="text-[10px] text-[var(--danger)]">{error}</p>}
            {lastResult && !error && <p className="text-[10px] text-[var(--success)]">{lastResult}</p>}
          </>
        )}
      </div>
    </BaseBlock>
  );
}

export const ModelBlock = memo(ModelBlockComponent);
