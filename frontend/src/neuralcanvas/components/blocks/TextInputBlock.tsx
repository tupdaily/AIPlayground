"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Text tokens flowing in visualization */
function TextInputViz({ seqLen }: { seqLen: number }) {
  const tokens = Math.min(seqLen, 8);
  const w = 160, h = 36;

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Token boxes */}
      {Array.from({ length: tokens }).map((_, i) => (
        <g key={i}>
          <rect
            x={10 + i * 18}
            y={8}
            width={14}
            height={16}
            rx={3}
            fill="#F59E0B"
            opacity={0.4 + (i / tokens) * 0.4}
          />
          <text
            x={17 + i * 18}
            y={20}
            textAnchor="middle"
            fontSize="7"
            fill="#F59E0B"
            opacity={0.85 + (i / tokens) * 0.15}
            fontWeight="600"
          >
            {i + 1}
          </text>
        </g>
      ))}
      {tokens < seqLen && (
        <text x={14 + tokens * 18} y={20} fontSize="8" fill="#F59E0B" opacity="0.75">...</text>
      )}
      {/* Label */}
      <text x={10} y={h - 2} fontSize="7" fill="#F59E0B" opacity="0.85">
        Token IDs [B, {seqLen}]
      </text>
    </svg>
  );
}

function TextInputBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const seqLen = Number(data?.params?.seq_len ?? 128);

  return (
    <BaseBlock id={id} blockType="TextInput" params={data?.params ?? {}} selected={!!selected}>
      <TextInputViz seqLen={seqLen} />
    </BaseBlock>
  );
}

export const TextInputBlock = memo(TextInputBlockComponent);
