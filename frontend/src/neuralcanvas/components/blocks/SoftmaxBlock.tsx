"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Bar chart normalizing to sum=1 */
function SoftmaxViz() {
  const bars = [0.65, 0.2, 0.08, 0.04, 0.03];
  const w = 160, h = 44;
  const barW = 16, gap = 6;
  const startX = (w - bars.length * (barW + gap)) / 2;

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Baseline */}
      <line x1={startX - 4} y1={h - 8} x2={startX + bars.length * (barW + gap)} y2={h - 8} stroke="#E5E7EB" strokeWidth="1" />

      {/* Bars */}
      {bars.map((val, i) => {
        const barH = val * (h - 16);
        return (
          <g key={i}>
            <rect
              x={startX + i * (barW + gap)}
              y={h - 8 - barH}
              width={barW}
              height={barH}
              rx={3}
              fill="#EF4444"
              opacity={0.6 + val * 0.4}
            />
            <text
              x={startX + i * (barW + gap) + barW / 2}
              y={h - 10 - barH}
              textAnchor="middle"
              fontSize="6"
              fill="#EF4444"
              opacity="0.95"
              fontWeight="600"
            >
              {Math.round(val * 100)}%
            </text>
          </g>
        );
      })}

      {/* Sum label */}
      <text x={w / 2} y={h - 1} textAnchor="middle" fontSize="7" fill="#EF4444" opacity="0.85">
        Σ = 1.00
      </text>
    </svg>
  );
}

function SoftmaxBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType="Softmax" params={data?.params ?? {}} selected={!!selected}>
      <SoftmaxViz />
    </BaseBlock>
  );
}

export const SoftmaxBlock = memo(SoftmaxBlockComponent);
