"use client";

// ---------------------------------------------------------------------------
// FlattenBlock — flattens all dims except batch
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";
import { CANVAS_UI_SCALE } from "@/neuralcanvas/lib/canvasConstants";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

const s = CANVAS_UI_SCALE;

function FlattenBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock
      id={id}
      blockType="Flatten"
      params={data?.params ?? {}}
      selected={!!selected}
    >
      <p className="text-neutral-600 font-mono not-italic" style={{ fontSize: `${7 * s}px` }}>
        [B, C, H, W] → [B, C×H×W]
      </p>
    </BaseBlock>
  );
}

export const FlattenBlock = memo(FlattenBlockComponent);
