"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** 3D cube unwrapping into a flat line */
function FlattenViz() {
  return (
    <svg width={160} height={44} viewBox="0 0 160 44">
      {/* 3D cube (left) */}
      <rect x={20} y={8} width={20} height={20} rx={2} fill="#8B5CF640" stroke="#8B5CF6" strokeWidth="1" />
      <rect x={26} y={4} width={20} height={20} rx={2} fill="#8B5CF630" stroke="#8B5CF6" strokeWidth="1" />
      <line x1={20} y1={8} x2={26} y2={4} stroke="#8B5CF6" strokeWidth="1" opacity="0.85" />
      <line x1={40} y1={8} x2={46} y2={4} stroke="#8B5CF6" strokeWidth="1" opacity="0.85" />
      <line x1={40} y1={28} x2={46} y2={24} stroke="#8B5CF6" strokeWidth="1" opacity="0.85" />

      {/* Arrow */}
      <line x1={54} y1={18} x2={74} y2={18} stroke="#8B5CF6" strokeWidth="1" opacity="0.85" />
      <polygon points="72,15 78,18 72,21" fill="#8B5CF6" opacity="0.85" />

      {/* Flat line (right) */}
      {Array.from({ length: 8 }).map((_, i) => (
        <rect
          key={i}
          x={86 + i * 8}
          y={14}
          width={6}
          height={8}
          rx={1.5}
          fill="#8B5CF6"
          opacity={0.6 + (i / 8) * 0.4}
        />
      ))}

      {/* Labels */}
      <text x={33} y={40} textAnchor="middle" fontSize="8" fill="#8B5CF6" opacity="0.85">[B,C,H,W]</text>
      <text x={118} y={40} textAnchor="middle" fontSize="8" fill="#8B5CF6" opacity="0.85">[B, N]</text>
    </svg>
  );
}

function FlattenBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType="Flatten" params={data?.params ?? {}} selected={!!selected} data={data}>
      <FlattenViz />
    </BaseBlock>
  );
}

export const FlattenBlock = memo(FlattenBlockComponent);
