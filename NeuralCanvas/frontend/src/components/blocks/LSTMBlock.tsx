"use client";

// ---------------------------------------------------------------------------
// LSTMBlock — Long Short-Term Memory recurrent layer
// ---------------------------------------------------------------------------

import { memo } from "react";
import type { NodeProps } from "reactflow";
import { BaseBlock } from "./BaseBlock";

interface BlockData {
  params: Record<string, number | string>;
}

function LSTMBlockComponent({ id, data, selected }: NodeProps<BlockData>) {
  return (
    <BaseBlock
      id={id}
      blockType="LSTM"
      params={data?.params ?? {}}
      selected={!!selected}
    />
  );
}

export const LSTMBlock = memo(LSTMBlockComponent);
