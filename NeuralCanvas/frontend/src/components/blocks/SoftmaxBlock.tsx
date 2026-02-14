"use client";

// ---------------------------------------------------------------------------
// SoftmaxBlock — normalise logits to probabilities
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";

interface BlockData {
  params: Record<string, number | string>;
}

function SoftmaxBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  return (
    <BaseBlock
      id={id}
      blockType="Softmax"
      params={data?.params ?? {}}
      selected={!!selected}
    >
      <p className="text-[8px] text-neutral-600 font-mono">
        exp(xᵢ) / Σ exp(xⱼ)
      </p>
    </BaseBlock>
  );
}

export const SoftmaxBlock = memo(SoftmaxBlockComponent);
