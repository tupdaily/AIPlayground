"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Two streams merging with "+" */
function AddViz() {
  return (
    <svg width={160} height={40} viewBox="0 0 160 40">
      {/* Stream A */}
      <line x1={8} y1={12} x2={54} y2={20} stroke="#8B5CF6" strokeWidth="1.5" opacity="0.75" />
      <text x={8} y={8} fontSize="8" fill="#8B5CF6" opacity="0.85" fontWeight="600">A</text>

      {/* Stream B */}
      <line x1={8} y1={30} x2={54} y2={20} stroke="#8B5CF6" strokeWidth="1.5" opacity="0.75" />
      <text x={8} y={38} fontSize="8" fill="#8B5CF6" opacity="0.85" fontWeight="600">B</text>

      {/* Plus circle */}
      <circle cx={66} cy={20} r={10} fill="#8B5CF640" stroke="#8B5CF6" strokeWidth="1" />
      <line x1={61} y1={20} x2={71} y2={20} stroke="#8B5CF6" strokeWidth="1.5" />
      <line x1={66} y1={15} x2={66} y2={25} stroke="#8B5CF6" strokeWidth="1.5" />

      {/* Output */}
      <line x1={78} y1={20} x2={148} y2={20} stroke="#8B5CF6" strokeWidth="1.5" opacity="0.75" />
      <polygon points="146,17 152,20 146,23" fill="#8B5CF6" opacity="0.85" />
      <text x={100} y={14} fontSize="7" fill="#8B5CF6" opacity="0.75">A + B</text>
    </svg>
  );
}

function AddBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType="Add" params={data?.params ?? {}} selected={!!selected}>
      <AddViz />
    </BaseBlock>
  );
}

export const AddBlock = memo(AddBlockComponent);
