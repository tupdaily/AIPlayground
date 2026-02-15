"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Flag/finish node visualization */
function OutputViz() {
  return (
    <svg width={160} height={36} viewBox="0 0 160 36">
      {/* Input stream */}
      <line x1={8} y1={18} x2={56} y2={18} stroke="#10B981" strokeWidth="1.5" opacity="0.75" />
      <polygon points="54,15 60,18 54,21" fill="#10B981" opacity="0.85" />

      {/* Checkered flag pattern */}
      <rect x={64} y={6} width={6} height={6} fill="#10B981" opacity="0.85" />
      <rect x={76} y={6} width={6} height={6} fill="#10B981" opacity="0.85" />
      <rect x={70} y={12} width={6} height={6} fill="#10B981" opacity="0.85" />
      <rect x={64} y={18} width={6} height={6} fill="#10B981" opacity="0.85" />
      <rect x={76} y={18} width={6} height={6} fill="#10B981" opacity="0.85" />
      <rect x={70} y={24} width={6} height={6} fill="#10B981" opacity="0.85" />

      {/* Border */}
      <rect x={63} y={5} width={20} height={26} rx={3} fill="none" stroke="#10B981" strokeWidth="1" opacity="0.75" />

      {/* Predictions label */}
      <text x={96} y={16} fontSize="8" fill="#10B981" opacity="0.9" fontWeight="600">Predictions</text>
      <text x={96} y={26} fontSize="7" fill="#10B981" opacity="0.75">(logits / loss)</text>
    </svg>
  );
}

function OutputBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType="Output" params={data?.params ?? {}} selected={!!selected}>
      <OutputViz />
    </BaseBlock>
  );
}

export const OutputBlock = memo(OutputBlockComponent);
