"use client";

// ---------------------------------------------------------------------------
// NormBlock — LayerNorm & BatchNorm (shared component)
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";
import type { BlockType } from "@/lib/blockRegistry";

interface BlockData {
  params: Record<string, number | string>;
}

// LayerNorm
function LayerNormBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  return (
    <BaseBlock
      id={id}
      blockType={"LayerNorm" as BlockType}
      params={data?.params ?? {}}
      selected={!!selected}
    />
  );
}

export const LayerNormBlock = memo(LayerNormBlockComponent);

// BatchNorm
function BatchNormBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  return (
    <BaseBlock
      id={id}
      blockType={"BatchNorm" as BlockType}
      params={data?.params ?? {}}
      selected={!!selected}
    />
  );
}

export const BatchNormBlock = memo(BatchNormBlockComponent);
