"use client";

// ---------------------------------------------------------------------------
// ActivationBlock — non-linear activation function
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";

interface BlockData {
  params: Record<string, number | string>;
}

function ActivationBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  return (
    <BaseBlock
      id={id}
      blockType="Activation"
      params={data?.params ?? {}}
      selected={!!selected}
    >
      {/* Visual hint of the selected function */}
      <p className="text-[9px] text-neutral-500 font-mono">
        f(x) = {String(data?.params?.activation ?? "relu")}(x)
      </p>
    </BaseBlock>
  );
}

export const ActivationBlock = memo(ActivationBlockComponent);
