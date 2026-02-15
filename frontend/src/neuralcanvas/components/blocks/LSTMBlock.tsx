"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Looping arrow + gate diagram */
function LSTMViz() {
  return (
    <svg width={160} height={48} viewBox="0 0 160 48">
      {/* Cell body */}
      <rect x={40} y={10} width={80} height={28} rx={8} fill="#EC489940" stroke="#EC4899" strokeWidth="1" opacity="0.85" />

      {/* Gates */}
      <circle cx={56} cy={24} r={6} fill="#EC489960" stroke="#EC4899" strokeWidth="1" />
      <text x={56} y={27} textAnchor="middle" fontSize="7" fill="#EC4899" fontWeight="700">f</text>

      <circle cx={80} cy={24} r={6} fill="#EC489960" stroke="#EC4899" strokeWidth="1" />
      <text x={80} y={27} textAnchor="middle" fontSize="7" fill="#EC4899" fontWeight="700">i</text>

      <circle cx={104} cy={24} r={6} fill="#EC489960" stroke="#EC4899" strokeWidth="1" />
      <text x={104} y={27} textAnchor="middle" fontSize="7" fill="#EC4899" fontWeight="700">o</text>

      {/* Input arrow */}
      <line x1={12} y1={24} x2={38} y2={24} stroke="#EC4899" strokeWidth="1" opacity="0.85" />
      <polygon points="36,21 42,24 36,27" fill="#EC4899" opacity="0.85" />

      {/* Output arrow */}
      <line x1={122} y1={24} x2={148} y2={24} stroke="#EC4899" strokeWidth="1" opacity="0.85" />
      <polygon points="146,21 152,24 146,27" fill="#EC4899" opacity="0.85" />

      {/* Recurrent loop */}
      <path d="M120,12 C136,12 136,-2 80,-2 C24,-2 24,12 40,12" fill="none" stroke="#EC4899" strokeWidth="1" strokeDasharray="3 2" opacity="0.75" />
      <polygon points="42,10 38,12 42,14" fill="#EC4899" opacity="0.75" />

      {/* Labels */}
      <text x={56} y={45} textAnchor="middle" fontSize="7" fill="#EC4899" opacity="0.85">forget</text>
      <text x={80} y={45} textAnchor="middle" fontSize="7" fill="#EC4899" opacity="0.85">input</text>
      <text x={104} y={45} textAnchor="middle" fontSize="7" fill="#EC4899" opacity="0.85">output</text>
    </svg>
  );
}

function LSTMBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType="LSTM" params={data?.params ?? {}} selected={!!selected}>
      <LSTMViz />
    </BaseBlock>
  );
}

export const LSTMBlock = memo(LSTMBlockComponent);
