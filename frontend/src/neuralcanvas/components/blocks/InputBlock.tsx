"use client";

import { memo, useCallback, useEffect, useMemo, useState } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { useReactFlow, useEdges } from "@xyflow/react";
import { Upload } from "lucide-react";
import { BaseBlock } from "./BaseBlock";
import { useShapes } from "@/neuralcanvas/components/canvas/ShapeContext";
import { getShapeLabel } from "@/neuralcanvas/lib/shapeEngine";
import { fetchDatasets, type DatasetInfo } from "@/neuralcanvas/lib/trainingApi";
import { DatasetUploadModal } from "@/neuralcanvas/components/datasets/DatasetUploadModal";
import { createClient } from "@/lib/supabase/client";

const CUSTOM_DATASET_ID = "__custom__";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Rounded "entry" node with data flow arrows */
function InputViz() {
  return (
    <svg width={160} height={36} viewBox="0 0 160 36">
      {/* Incoming data arrows */}
      {[10, 18, 26].map((y, i) => (
        <g key={i}>
          <line x1={12} y1={y} x2={40} y2={y} stroke="#F59E0B" strokeWidth="1" opacity={0.55 + i * 0.15} />
          <polygon points={`38,${y - 2} 44,${y} 38,${y + 2}`} fill="#F59E0B" opacity={0.55 + i * 0.15} />
        </g>
      ))}
      {/* Entry funnel */}
      <path d="M48,4 L70,12 L70,24 L48,32 Z" fill="#F59E0B30" stroke="#F59E0B" strokeWidth="1" opacity="0.85" rx="4" />
      {/* Output stream */}
      <line x1={74} y1={18} x2={148} y2={18} stroke="#F59E0B" strokeWidth="1.5" opacity="0.75" />
      <polygon points="146,15 152,18 146,21" fill="#F59E0B" opacity="0.85" />

      {/* Label */}
      <text x={80} y={12} fontSize="7" fill="#F59E0B" opacity="0.9" fontWeight="600">
        Data → Training
      </text>
    </svg>
  );
}

function InputBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const { shapes } = useShapes();
  const { setNodes, getNode } = useReactFlow();
  const edges = useEdges();
  const result = shapes.get(id);
  const outLabel = getShapeLabel(result?.outputShape ?? null);
  const params = data?.params ?? {};
  const datasetId = (params.dataset_id as string) ?? "";

  const hasCustomInputConnected = useMemo(() => {
    const incoming = edges.filter((e) => e.target === id);
    for (const e of incoming) {
      const sourceNode = getNode(e.source);
      if (sourceNode?.type === "InputSpace" || sourceNode?.type === "Board") return true;
    }
    return false;
  }, [id, edges, getNode]);

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [accessToken, setAccessToken] = useState<string | null>(null);

  // Get Supabase session for auth
  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getSession().then(({ data: { session } }) => {
      setAccessToken(session?.access_token ?? null);
    });
  }, []);

  // Fetch datasets (with auth token if available)
  const loadDatasets = useCallback(() => {
    fetchDatasets(accessToken ?? undefined)
      .then((list) => {
        setDatasets(list);
      })
      .catch((e) => {
        setDatasetError(e instanceof Error ? e.message : "Failed to load datasets");
      });
  }, [accessToken]);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  useEffect(() => {
    if (datasetId === CUSTOM_DATASET_ID || !datasets.length || (params.input_shape as string)?.length) return;
    const match = datasets.find((d) => d.id === datasetId);
    if (!match?.input_shape?.length) return;
    const shapeStr = match.input_shape.join(",");
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        const prevParams = (n.data?.params && typeof n.data.params === "object") ? (n.data.params as Record<string, number | string>) : {};
        return { ...n, data: { ...n.data, params: { ...prevParams, input_shape: shapeStr } } };
      }),
    );
  }, [datasetId, datasets, id, setNodes, params.input_shape]);

  useEffect(() => {
    if (hasCustomInputConnected && datasetId !== CUSTOM_DATASET_ID) {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prevParams = (n.data?.params && typeof n.data.params === "object") ? (n.data.params as Record<string, number | string>) : {};
          return { ...n, data: { ...n.data, params: { ...prevParams, dataset_id: CUSTOM_DATASET_ID } } };
        }),
      );
    }
  }, [hasCustomInputConnected, datasetId, id, setNodes]);

  const onDatasetChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value;
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prevParams = (n.data?.params && typeof n.data.params === "object")
            ? (n.data.params as Record<string, number | string>) : {};
          if (value === CUSTOM_DATASET_ID) {
            return { ...n, data: { ...n.data, params: { ...prevParams, dataset_id: CUSTOM_DATASET_ID } } };
          }
          const selectedDataset = datasets.find((d) => d.id === value);
          const inputShape = selectedDataset?.input_shape?.length ? selectedDataset.input_shape.join(",") : "";
          return {
            ...n,
            data: {
              ...n.data,
              params: { ...prevParams, dataset_id: value, input_shape: inputShape },
            },
          };
        }),
      );
    },
    [id, setNodes, datasets],
  );

  const handleUploaded = useCallback(
    (dataset: DatasetInfo) => {
      // Refresh the list and auto-select the new dataset
      loadDatasets();
      // Auto-select by updating the node params
      const inputShape = dataset.input_shape?.length ? dataset.input_shape.join(",") : "";
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prevParams = (n.data?.params && typeof n.data.params === "object")
            ? (n.data.params as Record<string, number | string>) : {};
          return {
            ...n,
            data: {
              ...n.data,
              params: { ...prevParams, dataset_id: dataset.id, input_shape: inputShape },
            },
          };
        }),
      );
    },
    [id, setNodes, loadDatasets],
  );

  const builtinDatasets = datasets.filter((d) => d.is_builtin);
  const customDatasets = datasets.filter((d) => !d.is_builtin);

  const displayValue = hasCustomInputConnected ? CUSTOM_DATASET_ID : datasetId;
  const isCustom = displayValue === CUSTOM_DATASET_ID;

  return (
    <BaseBlock id={id} blockType="Input" params={params} selected={!!selected} data={data}>
      <InputViz />
      <div className="space-y-2 pt-0.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] text-[var(--foreground-muted)] font-medium shrink-0">
            Dataset
          </span>
          <span className="text-[10px] font-mono text-[var(--foreground-muted)] truncate min-w-0" title="Output shape">
            {outLabel}
          </span>
        </div>
        <div className="flex items-center gap-0.5">
          <select
            value={displayValue}
            onChange={onDatasetChange}
            disabled={!!datasetError || isCustom}
            className="
              nodrag nopan w-full
              px-2.5 py-1.5 rounded-lg text-[12px] font-medium
              bg-[var(--surface-elevated)] border border-[var(--border)]
              text-[var(--foreground)]
              outline-none focus:border-[var(--block-input)] focus:ring-1 focus:ring-[var(--block-input)]/20
              disabled:opacity-50 disabled:cursor-not-allowed
              transition-colors duration-100
              cursor-pointer
            "
            title={isCustom ? "Custom data from Input Space" : datasetError ?? "Choose a dataset for training"}
          >
            <option value="">Choose a dataset…</option>
            <option value={CUSTOM_DATASET_ID}>Custom (from Input Space)</option>
            {builtinDatasets.length > 0 && (
              <optgroup label="Built-in">
                {builtinDatasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </optgroup>
            )}
            {customDatasets.length > 0 && (
              <optgroup label="My Datasets">
                {customDatasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </optgroup>
            )}
            {builtinDatasets.length === 0 && customDatasets.length === 0 && datasets.map((d) => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>
          {accessToken && (
            <button
              onClick={() => setUploadOpen(true)}
              className="nodrag nopan p-1.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] text-[var(--foreground-muted)] hover:text-[var(--foreground)] hover:bg-[var(--accent-muted)] transition-colors"
              title="Upload custom dataset"
            >
              <Upload size={12} />
            </button>
          )}
        </div>
        {isCustom && (
          <p className="text-[10px] text-[var(--foreground-muted)]">
            Using custom data from Custom Data block.
          </p>
        )}
      </div>

      {/* Upload modal (rendered via portal, outside canvas scaling) */}
      {accessToken && (
        <DatasetUploadModal
          open={uploadOpen}
          onClose={() => setUploadOpen(false)}
          onUploaded={handleUploaded}
          accessToken={accessToken}
        />
      )}
    </BaseBlock>
  );
}

export const InputBlock = memo(InputBlockComponent);
