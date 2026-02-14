"use client";

// ---------------------------------------------------------------------------
// InputBlock — dataset input (special: dropdown for dataset, shows output shape)
// ---------------------------------------------------------------------------

import { memo, useMemo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";
import { useShapes } from "@/components/canvas/ShapeContext";
import { getShapeLabel } from "@/lib/shapeEngine";

interface BlockData {
  params: Record<string, number | string>;
}

const DATASET_INFO: Record<string, { desc: string; classes: number }> = {
  MNIST: { desc: "28x28 grayscale handwritten digits", classes: 10 },
  CIFAR: { desc: "32x32 RGB natural images", classes: 10 },
  TinyShakespeare: { desc: "Character-level Shakespeare corpus", classes: 65 },
};

function InputBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  const dataset = String(data?.params?.dataset ?? "MNIST");
  const info = DATASET_INFO[dataset];
  const { shapes } = useShapes();
  const result = shapes.get(id);
  const outLabel = getShapeLabel(result?.outputShape ?? null);

  return (
    <BaseBlock
      id={id}
      blockType="Input"
      params={data?.params ?? {}}
      selected={!!selected}
    >
      {/* Dataset info card */}
      <div className="space-y-1 mt-0.5">
        {info && (
          <p className="text-[8px] text-neutral-500 leading-relaxed">
            {info.desc}
          </p>
        )}
        <div className="flex items-center justify-between">
          <span className="text-[8px] text-neutral-600 font-mono">output</span>
          <span className="text-[9px] font-mono text-amber-400/80">{outLabel}</span>
        </div>
        {info && (
          <div className="flex items-center justify-between">
            <span className="text-[8px] text-neutral-600 font-mono">classes</span>
            <span className="text-[9px] font-mono text-amber-400/80">{info.classes}</span>
          </div>
        )}
      </div>
    </BaseBlock>
  );
}

export const InputBlock = memo(InputBlockComponent);
