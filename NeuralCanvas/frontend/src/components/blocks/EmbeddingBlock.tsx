"use client";

// ---------------------------------------------------------------------------
// EmbeddingBlock — maps token IDs to dense vectors
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";

interface BlockData {
  params: Record<string, number | string>;
}

function EmbeddingBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  const vocab = Number(data?.params?.num_embeddings ?? 10000);
  const dim = Number(data?.params?.embedding_dim ?? 128);
  const totalParams = vocab * dim;

  return (
    <BaseBlock
      id={id}
      blockType="Embedding"
      params={data?.params ?? {}}
      selected={!!selected}
    >
      <p className="text-[8px] text-neutral-600 font-mono">
        table: {vocab.toLocaleString()} x {dim} = {totalParams.toLocaleString()} params
      </p>
    </BaseBlock>
  );
}

export const EmbeddingBlock = memo(EmbeddingBlockComponent);
