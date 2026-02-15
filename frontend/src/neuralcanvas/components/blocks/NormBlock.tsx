"use client";

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";
import type { BlockType } from "@/neuralcanvas/lib/blockRegistry";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** LayerNorm: normalize per row (each token/position independently). Rows → normalized rows. */
function LayerNormViz() {
  const w = 160, h = 44;
  const rows = 2;
  const cols = 6;
  const barW = 8;
  const gap = 2;
  const leftBlockW = cols * (barW + gap) - gap;
  const leftStart = 16;
  const rightStart = leftStart + leftBlockW + 20;

  // "Before": varying heights per row (simulating unnormalized features)
  const beforeHeights = [
    [0.4, 0.8, 0.5, 0.9, 0.3, 0.7],
    [0.6, 0.4, 0.9, 0.5, 0.8, 0.4],
  ];
  // "After": normalized (more uniform scale)
  const afterHeights = [
    [0.5, 0.85, 0.55, 0.9, 0.45, 0.75],
    [0.6, 0.5, 0.85, 0.55, 0.8, 0.5],
  ];
  const maxH = 10;

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Before: two horizontal strips (rows) with varying bars */}
      {beforeHeights.map((row, ri) =>
        row.map((val, ci) => (
          <rect
            key={`b-${ri}-${ci}`}
            x={leftStart + ci * (barW + gap)}
            y={12 + ri * 14 + (maxH - val * maxH)}
            width={barW}
            height={val * maxH}
            rx={1.5}
            fill="#14B8A6"
            opacity={0.7}
          />
        ))
      )}
      {/* Arrow: "normalize per row" */}
      <line x1={leftStart + leftBlockW + 6} y1={h / 2} x2={rightStart - 6} y2={h / 2} stroke="#14B8A6" strokeWidth="1.2" opacity={0.85} />
      <polygon points={`${rightStart - 10},${h / 2 - 4} ${rightStart - 4},${h / 2} ${rightStart - 10},${h / 2 + 4}`} fill="#14B8A6" opacity={0.85} />
      {/* After: same rows, normalized (more uniform) */}
      {afterHeights.map((row, ri) =>
        row.map((val, ci) => (
          <rect
            key={`a-${ri}-${ci}`}
            x={rightStart + ci * (barW + gap)}
            y={12 + ri * 14 + (maxH - val * maxH)}
            width={barW}
            height={val * maxH}
            rx={1.5}
            fill="#14B8A6"
            opacity={0.95}
          />
        ))
      )}
      <text x={leftStart + leftBlockW / 2} y={h - 2} textAnchor="middle" fontSize="6" fill="#14B8A6" opacity={0.8}>per layer</text>
      <text x={rightStart + leftBlockW / 2} y={h - 2} textAnchor="middle" fontSize="6" fill="#14B8A6" opacity={0.9}>norm</text>
    </svg>
  );
}

/** BatchNorm: normalize per column (each feature across batch). Columns → normalized columns. */
function BatchNormViz() {
  const w = 160, h = 44;
  const rows = 3;
  const cols = 4;
  const cellW = 12;
  const cellH = 8;
  const gap = 2;
  const gridW = cols * (cellW + gap) - gap;
  const gridH = rows * (cellH + gap) - gap;
  const leftStart = 12;
  const rightStart = leftStart + gridW + 18;

  // Grid "before": varying opacity per cell (simulating different batch values per feature)
  const cellOpacities = [
    [0.4, 0.7, 0.5, 0.8],
    [0.6, 0.5, 0.8, 0.4],
    [0.5, 0.6, 0.4, 0.7],
  ];
  // "After": normalized per column (each column has more similar opacity)
  const normedOpacities = [
    [0.55, 0.65, 0.6, 0.65],
    [0.6, 0.55, 0.65, 0.5],
    [0.55, 0.6, 0.55, 0.6],
  ];

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Before: batch × features grid */}
      {cellOpacities.map((row, ri) =>
        row.map((op, ci) => (
          <rect
            key={`b-${ri}-${ci}`}
            x={leftStart + ci * (cellW + gap)}
            y={8 + ri * (cellH + gap)}
            width={cellW}
            height={cellH}
            rx={2}
            fill="#14B8A6"
            opacity={0.5 + op * 0.5}
          />
        ))
      )}
      {/* Arrows: one per column (normalize down the column) */}
      {Array.from({ length: cols }).map((_, ci) => {
        const cx = leftStart + ci * (cellW + gap) + cellW / 2;
        const cy = 8 + gridH / 2;
        return (
          <g key={ci}>
            <line x1={cx} y1={8 + gridH} x2={cx} y2={8} stroke="#14B8A6" strokeWidth="0.8" strokeDasharray="2 1" opacity="0.6" />
          </g>
        );
      })}
      {/* Arrow to "after" */}
      <line x1={leftStart + gridW + 4} y1={h / 2} x2={rightStart - 4} y2={h / 2} stroke="#14B8A6" strokeWidth="1.2" opacity={0.85} />
      <polygon points={`${rightStart - 8},${h / 2 - 3} ${rightStart - 2},${h / 2} ${rightStart - 8},${h / 2 + 3}`} fill="#14B8A6" opacity={0.85} />
      {/* After: normalized grid (per-column normalized) */}
      {normedOpacities.map((row, ri) =>
        row.map((op, ci) => (
          <rect
            key={`a-${ri}-${ci}`}
            x={rightStart + ci * (cellW + gap)}
            y={8 + ri * (cellH + gap)}
            width={cellW}
            height={cellH}
            rx={2}
            fill="#14B8A6"
            opacity={0.5 + op * 0.5}
          />
        ))
      )}
      <text x={leftStart + gridW / 2} y={h - 2} textAnchor="middle" fontSize="6" fill="#14B8A6" opacity={0.8}>batch × feat</text>
      <text x={rightStart + gridW / 2} y={h - 2} textAnchor="middle" fontSize="6" fill="#14B8A6" opacity={0.9}>per batch</text>
    </svg>
  );
}

// LayerNorm
function LayerNormBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType={"LayerNorm" as BlockType} params={data?.params ?? {}} selected={!!selected} data={data}>
      <LayerNormViz />
    </BaseBlock>
  );
}

export const LayerNormBlock = memo(LayerNormBlockComponent);

// BatchNorm
function BatchNormBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  return (
    <BaseBlock id={id} blockType={"BatchNorm" as BlockType} params={data?.params ?? {}} selected={!!selected} data={data}>
      <BatchNormViz />
    </BaseBlock>
  );
}

export const BatchNormBlock = memo(BatchNormBlockComponent);
