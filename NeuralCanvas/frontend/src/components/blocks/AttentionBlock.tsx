"use client";

// ---------------------------------------------------------------------------
// AttentionBlock — multi-head self-attention
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";

interface BlockData {
  params: Record<string, number | string>;
}

function AttentionBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  return (
    <BaseBlock
      id={id}
      blockType="Attention"
      params={data?.params ?? {}}
      selected={!!selected}
    >
      {/* Extra: head-size hint */}
      {(() => {
        const embed = Number(data?.params?.embed_dim) || 512;
        const heads = Number(data?.params?.num_heads) || 8;
        const headDim = embed % heads === 0 ? embed / heads : null;
        return headDim ? (
          <p className="text-[8px] text-neutral-600 font-mono">
            head_dim: {headDim}
          </p>
        ) : null;
      })()}
    </BaseBlock>
  );
}

export const AttentionBlock = memo(AttentionBlockComponent);
