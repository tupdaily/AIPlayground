"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Mini MLP diagram: input neurons connected to output neurons */
function LinearViz({ inF, outF }: { inF: number; outF: number }) {
  const inCount = Math.min(Math.max(Math.round(inF / 200), 2), 5);
  const outCount = Math.min(Math.max(Math.round(outF / 40), 2), 5);
  const w = 160, h = 48;
  const inX = 20, outX = w - 20;

  const getY = (i: number, total: number) =>
    (h / (total + 1)) * (i + 1);

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Connections */}
      {Array.from({ length: inCount }).flatMap((_, ii) =>
        Array.from({ length: outCount }).map((_, oi) => (
          <line
            key={`${ii}-${oi}`}
            x1={inX} y1={getY(ii, inCount)}
            x2={outX} y2={getY(oi, outCount)}
            stroke="#6366F1" strokeWidth="1" opacity="0.5"
          />
        ))
      )}
      {/* Input neurons */}
      {Array.from({ length: inCount }).map((_, i) => (
        <circle key={`in-${i}`} cx={inX} cy={getY(i, inCount)} r={4} fill="#6366F1" opacity="0.9" />
      ))}
      {/* Output neurons */}
      {Array.from({ length: outCount }).map((_, i) => (
        <circle key={`out-${i}`} cx={outX} cy={getY(i, outCount)} r={4} fill="#6366F1" opacity="1" />
      ))}
    </svg>
  );
}

function LinearBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const inF = Number(data?.params?.in_features ?? 784);
  const outF = Number(data?.params?.out_features ?? 128);

  return (
    <BaseBlock id={id} blockType="Linear" params={data?.params ?? {}} selected={!!selected} data={data}>
      <LinearViz inF={inF} outF={outF} />
    </BaseBlock>
  );
}

export const LinearBlock = memo(LinearBlockComponent);
