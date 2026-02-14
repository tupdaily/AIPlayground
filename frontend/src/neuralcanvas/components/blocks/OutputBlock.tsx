"use client";

// ---------------------------------------------------------------------------
// OutputBlock — sink for model output (logits, loss, etc.)
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";
import { useShapes } from "@/neuralcanvas/components/canvas/ShapeContext";
import { getShapeLabel } from "@/neuralcanvas/lib/shapeEngine";
import { CANVAS_UI_SCALE } from "@/neuralcanvas/lib/canvasConstants";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

const s = CANVAS_UI_SCALE;

function OutputBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const { shapes } = useShapes();
  const result = shapes.get(id);
  const inLabel = getShapeLabel(result?.inputShape ?? null);

  return (
    <BaseBlock
      id={id}
      blockType="Output"
      params={data?.params ?? {}}
      selected={!!selected}
    >
      <div className="flex items-center justify-between gap-1 mt-0.5">
        <span className="text-neutral-600 font-mono shrink-0" style={{ fontSize: `${7 * s}px` }}>in</span>
        <span className="font-mono text-emerald-400/80 truncate min-w-0" style={{ fontSize: `${7 * s}px` }}>{inLabel}</span>
      </div>
    </BaseBlock>
  );
}

export const OutputBlock = memo(OutputBlockComponent);
