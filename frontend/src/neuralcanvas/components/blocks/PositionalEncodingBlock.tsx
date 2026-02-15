"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Numbered sequence with sinusoidal wave overlay */
function PosEncViz({ maxLen }: { maxLen: number }) {
  const w = 160, h = 40;
  const positions = Math.min(maxLen, 8);

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Sine wave */}
      <path
        d={Array.from({ length: 40 }, (_, i) => {
          const x = 8 + (i / 39) * (w - 16);
          const y = h / 2 + Math.sin(i * 0.4) * 8;
          return `${i === 0 ? "M" : "L"}${x},${y}`;
        }).join(" ")}
        fill="none"
        stroke="#0EA5E9"
        strokeWidth="1"
        opacity="0.5"
      />

      {/* Position markers */}
      {Array.from({ length: positions }).map((_, i) => {
        const x = 14 + (i / (positions - 1)) * (w - 28);
        return (
          <g key={i}>
            <circle cx={x} cy={h / 2} r={8} fill="#0EA5E930" stroke="#0EA5E9" strokeWidth="1" />
            <text x={x} y={h / 2 + 3} textAnchor="middle" fontSize="7" fill="#0EA5E9" fontWeight="600">
              {i + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function PositionalEncodingBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const maxLen = Number(data?.params?.max_len ?? 512);

  return (
    <BaseBlock id={id} blockType="PositionalEncoding" params={data?.params ?? {}} selected={!!selected}>
      <PosEncViz maxLen={maxLen} />
    </BaseBlock>
  );
}

export const PositionalEncodingBlock = memo(PositionalEncodingBlockComponent);
