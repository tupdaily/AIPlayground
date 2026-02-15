"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Words to vectors visualization */
function TextEmbeddingViz({ vocabSize, dim }: { vocabSize: number; dim: number }) {
  const totalParams = vocabSize * dim;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={160} height={40} viewBox="0 0 160 40">
        {/* Word tokens on left */}
        {["the", "cat", "sat"].map((word, i) => (
          <g key={i}>
            <rect x={8} y={4 + i * 12} width={28} height={10} rx={3} fill="#06B6D430" stroke="#06B6D4" strokeWidth="1" />
            <text x={22} y={12 + i * 12} textAnchor="middle" fontSize="6" fill="#06B6D4" fontWeight="600">{word}</text>
          </g>
        ))}

        {/* Arrows */}
        {[0, 1, 2].map((i) => (
          <line key={i} x1={38} y1={9 + i * 12} x2={56} y2={9 + i * 12} stroke="#06B6D4" strokeWidth="1" opacity="0.75" />
        ))}

        {/* Vector bars on right */}
        {[0, 1, 2].map((row) => (
          <g key={row}>
            {Array.from({ length: 6 }).map((_, c) => (
              <rect
                key={c}
                x={60 + c * 12}
                y={4 + row * 12}
                width={10}
                height={8}
                rx={1.5}
                fill="#06B6D4"
                opacity={0.4 + Math.random() * 0.4}
              />
            ))}
          </g>
        ))}
      </svg>
      <span className="text-[9px] font-medium text-[#06B6D4] opacity-90">
        {totalParams.toLocaleString()} params
      </span>
    </div>
  );
}

function TextEmbeddingBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const vocabSize = Number(data?.params?.vocab_size ?? 10000);
  const dim = Number(data?.params?.embedding_dim ?? 128);

  return (
    <BaseBlock id={id} blockType="TextEmbedding" params={data?.params ?? {}} selected={!!selected} data={data}>
      <TextEmbeddingViz vocabSize={vocabSize} dim={dim} />
    </BaseBlock>
  );
}

export const TextEmbeddingBlock = memo(TextEmbeddingBlockComponent);
