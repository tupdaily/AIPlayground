"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Mini SVG line chart of the activation function curve */
function ActivationViz({ fn }: { fn: string }) {
  const w = 160;
  const h = 40;
  const pts = 50;
  const xMin = -4;
  const xMax = 4;
  const padding = { left: 16, right: 16, top: 6, bottom: 6 };
  const chartW = w - padding.left - padding.right;
  const chartH = h - padding.top - padding.bottom;

  const computeY = (x: number): number => {
    switch (fn) {
      case "relu":
        return Math.max(0, x);
      case "gelu":
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
      case "sigmoid":
        return 1 / (1 + Math.exp(-x));
      case "tanh":
        return Math.tanh(x);
      default:
        return Math.max(0, x);
    }
  };

  const samples = Array.from({ length: pts }, (_, i) => {
    const x = xMin + (i / (pts - 1)) * (xMax - xMin);
    return { x, y: computeY(x) };
  });

  const yMin = Math.min(...samples.map((s) => s.y));
  const yMax = Math.max(...samples.map((s) => s.y));
  const yRange = yMax - yMin || 1;

  const points = samples
    .map(({ x, y }) => {
      const px = padding.left + ((x - xMin) / (xMax - xMin)) * chartW;
      const py = padding.top + ((yMax - y) / yRange) * chartH;
      return `${px},${py}`;
    })
    .join(" ");

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="overflow-visible">
        {/* Axis lines */}
        <line
          x1={padding.left}
          y1={padding.top + chartH / 2}
          x2={w - padding.right}
          y2={padding.top + chartH / 2}
          stroke="var(--border-strong)"
          strokeWidth="0.8"
        />
        <line
          x1={padding.left + chartW / 2}
          y1={padding.top}
          x2={padding.left + chartW / 2}
          y2={h - padding.bottom}
          stroke="var(--border-strong)"
          strokeWidth="0.8"
        />
        {/* Curve */}
        <polyline
          points={points}
          fill="none"
          stroke="#EF4444"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className="text-[10px] font-semibold uppercase tracking-wider text-[#EF4444]">
        {fn}
      </span>
    </div>
  );
}

function ActivationBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const fn = String(data?.params?.activation ?? "relu");

  return (
    <BaseBlock id={id} blockType="Activation" params={data?.params ?? {}} selected={!!selected} data={data}>
      <ActivationViz fn={fn} />
    </BaseBlock>
  );
}

export const ActivationBlock = memo(ActivationBlockComponent);
