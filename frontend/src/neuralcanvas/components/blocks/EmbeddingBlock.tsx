"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Table lookup icon: rows light up showing "lookup" */
function EmbeddingViz({ vocab, dim }: { vocab: number; dim: number }) {
  const rows = 5;
  const cols = 6;
  const w = 160, h = 48;
  const cellW = 10, cellH = 6;
  const startX = 24, startY = 6;

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Table grid */}
      {Array.from({ length: rows }).flatMap((_, r) =>
        Array.from({ length: cols }).map((_, c) => (
          <rect
            key={`${r}-${c}`}
            x={startX + c * (cellW + 1)}
            y={startY + r * (cellH + 1)}
            width={cellW}
            height={cellH}
            rx={1}
            fill={r === 2 ? "#06B6D4" : "#E5E7EB"}
            opacity={r === 2 ? 0.85 : 0.5}
          />
        ))
      )}
      {/* Highlighted row arrow */}
      <line x1={10} y1={startY + 2 * (cellH + 1) + cellH / 2} x2={22} y2={startY + 2 * (cellH + 1) + cellH / 2} stroke="#06B6D4" strokeWidth="1" />
      <polygon points={`20,${startY + 2 * (cellH + 1) + cellH / 2 - 2} 24,${startY + 2 * (cellH + 1) + cellH / 2} 20,${startY + 2 * (cellH + 1) + cellH / 2 + 2}`} fill="#06B6D4" />

      {/* Output vector */}
      <line x1={startX + cols * (cellW + 1) + 4} y1={startY + 2 * (cellH + 1) + cellH / 2} x2={w - 20} y2={startY + 2 * (cellH + 1) + cellH / 2} stroke="#06B6D4" strokeWidth="1" opacity="0.85" />
      <text x={w - 16} y={startY + 2 * (cellH + 1) + cellH / 2 + 3} fontSize="7" fill="#06B6D4" opacity="0.9" fontWeight="600">→ vec</text>

      {/* Stats */}
      <text x={24} y={h - 2} fontSize="7" fill="#06B6D4" opacity="0.85">
        {(vocab * dim).toLocaleString()} params
      </text>
    </svg>
  );
}

function EmbeddingBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const vocab = Number(data?.params?.num_embeddings ?? 10000);
  const dim = Number(data?.params?.embedding_dim ?? 128);

  return (
    <BaseBlock id={id} blockType="Embedding" params={data?.params ?? {}} selected={!!selected}>
      <EmbeddingViz vocab={vocab} dim={dim} />
    </BaseBlock>
  );
}

export const EmbeddingBlock = memo(EmbeddingBlockComponent);
