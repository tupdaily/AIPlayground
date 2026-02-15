"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Mini kernel-on-grid visualization */
function Conv2DViz({ kernelSize }: { kernelSize: number }) {
  const gridSize = 6;
  const cellSize = 8;
  const gap = 1;
  const w = 160, h = 56;
  const startX = (w - gridSize * (cellSize + gap)) / 2;
  const startY = (h - gridSize * (cellSize + gap)) / 2;
  const k = Math.min(kernelSize, gridSize);

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Feature map grid */}
      {Array.from({ length: gridSize }).flatMap((_, row) =>
        Array.from({ length: gridSize }).map((_, col) => {
          const isKernel = row < k && col < k;
          return (
            <rect
              key={`${row}-${col}`}
              x={startX + col * (cellSize + gap)}
              y={startY + row * (cellSize + gap)}
              width={cellSize}
              height={cellSize}
              rx={1.5}
              fill={isKernel ? "#8B5CF6" : "#E5E7EB"}
              opacity={isKernel ? 0.95 : 0.75}
            />
          );
        })
      )}
      {/* Kernel outline */}
      <rect
        x={startX - 1}
        y={startY - 1}
        width={k * (cellSize + gap) + 1}
        height={k * (cellSize + gap) + 1}
        rx={2}
        fill="none"
        stroke="#8B5CF6"
        strokeWidth="1.5"
        strokeDasharray="3 2"
        opacity="0.9"
      />
      {/* Label */}
      <text x={w - 16} y={h - 6} textAnchor="end" fontSize="9" fill="#8B5CF6" fontWeight="600" opacity="0.95">
        {kernelSize}x{kernelSize}
      </text>
    </svg>
  );
}

function Conv2DBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const kernelSize = Number(data?.params?.kernel_size ?? 3);

  return (
    <BaseBlock id={id} blockType="Conv2D" params={data?.params ?? {}} selected={!!selected}>
      <Conv2DViz kernelSize={kernelSize} />
    </BaseBlock>
  );
}

export const Conv2DBlock = memo(Conv2DBlockComponent);
