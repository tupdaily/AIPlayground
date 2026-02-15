"use client";

import { memo, useCallback, useEffect, useState } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { useReactFlow } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";
import { useShapes } from "@/neuralcanvas/components/canvas/ShapeContext";
import { getShapeLabel } from "@/neuralcanvas/lib/shapeEngine";
import { CANVAS_UI_SCALE } from "@/neuralcanvas/lib/canvasConstants";
import { fetchDatasets } from "@/neuralcanvas/lib/trainingApi";

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
      <text x={90} y={12} fontSize="8" fill="#F59E0B" opacity="0.9" fontWeight="600">
        Dataset → Training Panel
      </text>
    </svg>
  );
}

function InputBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const s = CANVAS_UI_SCALE;
  const { shapes } = useShapes();
  const { setNodes } = useReactFlow();
  const result = shapes.get(id);
  const outLabel = getShapeLabel(result?.outputShape ?? null);
  const params = data?.params ?? {};
  const datasetId = (params.dataset_id as string) ?? "";

  const [datasets, setDatasets] = useState<{ id: string; name: string; input_shape: number[] }[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchDatasets()
      .then((list) => {
        if (!cancelled) setDatasets(list.map((d) => ({ id: d.id, name: d.name, input_shape: d.input_shape ?? [1, 28, 28] })));
      })
      .catch((e) => {
        if (!cancelled) setDatasetError(e instanceof Error ? e.message : "Failed to load datasets");
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!datasetId || !datasets.length || (params.input_shape as string)?.length) return;
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

  const onDatasetChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value;
      const selectedDataset = datasets.find((d) => d.id === value);
      const inputShape = selectedDataset?.input_shape?.length ? selectedDataset.input_shape.join(",") : "";
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prevParams = (n.data?.params && typeof n.data.params === "object")
            ? (n.data.params as Record<string, number | string>) : {};
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

  return (
    <BaseBlock id={id} blockType="Input" params={params} selected={!!selected} data={data}>
      <InputViz />
      <div className="space-y-px mt-0.5 leading-none">
        <div className="flex items-center justify-between gap-1">
          <span className="text-[var(--foreground-muted)] font-mono shrink-0" style={{ fontSize: `${7 * s}px` }}>out</span>
          <span className="font-mono text-[var(--block-input)] truncate min-w-0 opacity-90" style={{ fontSize: `${7 * s}px` }}>{outLabel}</span>
        </div>
        <select
          value={datasetId}
          onChange={onDatasetChange}
          disabled={!!datasetError}
          className="nodrag nopan w-full mt-0.5 px-1 py-1 rounded bg-[var(--surface-elevated)] border border-[var(--border)] text-[var(--foreground)] font-mono focus:outline-none focus:ring-1 focus:ring-[var(--accent-muted)] disabled:opacity-50 min-h-[18px]"
          style={{ fontSize: `${8 * s}px` }}
          title={datasetError ?? "Dataset"}
        >
          <option value="">Dataset…</option>
          {datasets.map((d) => (
            <option key={d.id} value={d.id} style={{ fontSize: `${8 * s}px` }}>{d.name}</option>
          ))}
        </select>
      </div>
    </BaseBlock>
  );
}

export const InputBlock = memo(InputBlockComponent);
