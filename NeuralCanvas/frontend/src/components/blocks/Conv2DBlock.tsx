"use client";

// ---------------------------------------------------------------------------
// Conv2DBlock — 2D convolutional layer
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";

interface BlockData {
  params: Record<string, number | string>;
}

function Conv2DBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  return (
    <BaseBlock
      id={id}
      blockType="Conv2D"
      params={data?.params ?? {}}
      selected={!!selected}
    />
  );
}

export const Conv2DBlock = memo(Conv2DBlockComponent);
