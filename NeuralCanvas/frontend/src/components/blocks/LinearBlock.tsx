"use client";

// ---------------------------------------------------------------------------
// LinearBlock — fully-connected dense layer
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";

interface BlockData {
  params: Record<string, number | string>;
}

function LinearBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  return (
    <BaseBlock
      id={id}
      blockType="Linear"
      params={data?.params ?? {}}
      selected={!!selected}
    />
  );
}

export const LinearBlock = memo(LinearBlockComponent);
