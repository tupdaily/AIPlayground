"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Attention: Q·K^T heatmap + "focus" metaphor — one position lights up attending to others */
function AttentionViz({ numHeads }: { numHeads: number }) {
  const n = 5;
  const w = 160;
  const h = 48;
  const cellSize = 14;
  const gap = 2;
  const gridLeft = (w - (n * (cellSize + gap) - gap)) / 2;
  const gridTop = 4;

  // Deterministic attention pattern: strong on diagonal (self) + soft off-diagonal (context)
  const getStrength = (row: number, col: number) => {
    const d = Math.abs(row - col);
    if (d === 0) return 0.95;
    if (d === 1) return 0.5;
    if (d === 2) return 0.25;
    return 0.12;
  };

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        {/* Mini attention heatmap (Q·K^T style) */}
        {Array.from({ length: n }).flatMap((_, row) =>
          Array.from({ length: n }).map((_, col) => {
            const strength = getStrength(row, col);
            const x = gridLeft + col * (cellSize + gap);
            const y = gridTop + row * (cellSize + gap);
            return (
              <rect
                key={`${row}-${col}`}
                x={x}
                y={y}
                width={cellSize}
                height={cellSize}
                rx={2}
                fill="#F97316"
                opacity={strength}
              />
            );
          })
        )}
      </svg>
      <span className="text-[9px] font-semibold uppercase tracking-wider text-[#F97316] opacity-90">
        {numHeads} heads
      </span>
    </div>
  );
}

function AttentionBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const numHeads = Number(data?.params?.num_heads ?? 8);

  return (
    <BaseBlock id={id} blockType="Attention" params={data?.params ?? {}} selected={!!selected} data={data}>
      <AttentionViz numHeads={numHeads} />
    </BaseBlock>
  );
}

export const AttentionBlock = memo(AttentionBlockComponent);
