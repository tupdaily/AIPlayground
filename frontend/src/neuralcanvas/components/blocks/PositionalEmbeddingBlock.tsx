"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Word + position combined visualization */
function PosEmbViz() {
  const w = 160, h = 40;
  const positions = 6;

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Position boxes with gradient */}
      {Array.from({ length: positions }).map((_, i) => {
        const x = 12 + (i / (positions - 1)) * (w - 36);
        return (
          <g key={i}>
            <rect
              x={x - 6}
              y={8}
              width={12}
              height={24}
              rx={3}
              fill="#0EA5E9"
              opacity={0.4 + (i / positions) * 0.4}
            />
            <text
              x={x}
              y={24}
              textAnchor="middle"
              fontSize="7"
              fill="#0EA5E9"
              fontWeight="600"
              opacity={0.85 + (i / positions) * 0.15}
            >
              {i + 1}
            </text>
          </g>
        );
      })}

      {/* Learned label */}
      <text x={w / 2} y={h - 2} textAnchor="middle" fontSize="7" fill="#0EA5E9" opacity="0.75">
        learned positions
      </text>
    </svg>
  );
}

function PositionalEmbeddingBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType="PositionalEmbedding" params={data?.params ?? {}} selected={!!selected} data={data}>
      <PosEmbViz />
    </BaseBlock>
  );
}

export const PositionalEmbeddingBlock = memo(PositionalEmbeddingBlockComponent);
