"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** 2x2 pool window reducing spatial size */
function MaxPool2DViz() {
  return (
    <svg width={160} height={44} viewBox="0 0 160 44">
      {/* Grid in (4 cells) */}
      <rect x={24} y={8} width={32} height={32} rx={2} fill="none" stroke="#8B5CF6" strokeWidth="1" opacity="0.7" />
      {[0, 1].map((i) =>
        [0, 1].map((j) => (
          <rect
            key={`${i}-${j}`}
            x={26 + i * 16}
            y={10 + j * 16}
            width={14}
            height={14}
            rx={1}
            fill="#8B5CF6"
            opacity={0.4 + (i + j) * 0.15}
          />
        ))
      )}
      {/* Arrow */}
      <line x1={62} y1={24} x2={82} y2={24} stroke="#8B5CF6" strokeWidth="1.5" opacity="0.85" />
      <polygon points="80,21 86,24 80,27" fill="#8B5CF6" opacity="0.85" />
      {/* Single cell (max pooled) */}
      <rect x={92} y={14} width={36} height={20} rx={2} fill="#8B5CF6" opacity="0.6" stroke="#8B5CF6" strokeWidth="1" />
      <text x={110} y={40} textAnchor="middle" fontSize="8" fill="#8B5CF6" opacity="0.85">
        MaxPool2D
      </text>
    </svg>
  );
}

function MaxPool2DBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType="MaxPool2D" params={data?.params ?? {}} selected={!!selected} data={data}>
      <MaxPool2DViz />
    </BaseBlock>
  );
}

export const MaxPool2DBlock = memo(MaxPool2DBlockComponent);
