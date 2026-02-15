"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Two data streams joining side by side */
function ConcatViz() {
  return (
    <svg width={160} height={40} viewBox="0 0 160 40">
      {/* Stream A blocks */}
      {[0, 1, 2].map((i) => (
        <rect key={`a-${i}`} x={10 + i * 14} y={6} width={12} height={14} rx={2} fill="#8B5CF6" opacity={0.5 + i * 0.1} />
      ))}
      <text x={28} y={30} textAnchor="middle" fontSize="7" fill="#8B5CF6" opacity="0.85">A</text>

      {/* Stream B blocks */}
      {[0, 1, 2].map((i) => (
        <rect key={`b-${i}`} x={10 + i * 14} y={22} width={12} height={14} rx={2} fill="#8B5CF6" opacity={0.4 + i * 0.1} />
      ))}

      {/* Arrows */}
      <line x1={56} y1={13} x2={68} y2={20} stroke="#8B5CF6" strokeWidth="1" opacity="0.75" />
      <line x1={56} y1={29} x2={68} y2={20} stroke="#8B5CF6" strokeWidth="1" opacity="0.75" />

      {/* Combined blocks */}
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <rect key={`c-${i}`} x={74 + i * 12} y={12} width={10} height={16} rx={2} fill="#8B5CF6" opacity={0.5 + (i / 6) * 0.4} />
      ))}

      {/* Arrow out */}
      <line x1={148} y1={20} x2={156} y2={20} stroke="#8B5CF6" strokeWidth="1" opacity="0.75" />
      <text x={110} y={38} textAnchor="middle" fontSize="7" fill="#8B5CF6" opacity="0.75">[A | B]</text>
    </svg>
  );
}

function ConcatBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType="Concat" params={data?.params ?? {}} selected={!!selected} data={data}>
      <ConcatViz />
    </BaseBlock>
  );
}

export const ConcatBlock = memo(ConcatBlockComponent);
