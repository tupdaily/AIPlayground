"use client";

// ---------------------------------------------------------------------------
// FlattenBlock — flattens all dims except batch
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";

interface BlockData {
  params: Record<string, number | string>;
}

function FlattenBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  return (
    <BaseBlock
      id={id}
      blockType="Flatten"
      params={data?.params ?? {}}
      selected={!!selected}
    >
      <p className="text-[8px] text-neutral-600 font-mono italic">
        [B, C, H, W] → [B, C*H*W]
      </p>
    </BaseBlock>
  );
}

export const FlattenBlock = memo(FlattenBlockComponent);
