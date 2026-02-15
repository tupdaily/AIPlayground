"use client";

import { memo, useMemo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

/** Dots where some randomly fade out to show "dropping" neurons */
function DropoutViz({ dropRate }: { dropRate: number }) {
  const w = 160, h = 40;
  const cols = 10, rows = 3;
  const dotR = 4;
  const keepPct = Math.round((1 - dropRate) * 100);

  // Stable random pattern (seeded by dropRate)
  const dropped = useMemo(() => {
    const set = new Set<number>();
    const total = cols * rows;
    const numDrop = Math.round(total * dropRate);
    let seed = Math.round(dropRate * 1000);
    while (set.size < numDrop && set.size < total) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      set.add(seed % total);
    }
    return set;
  }, [dropRate]);

  return (
    <div>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        {Array.from({ length: rows }).flatMap((_, row) =>
          Array.from({ length: cols }).map((_, col) => {
            const idx = row * cols + col;
            const isDrop = dropped.has(idx);
            const cx = 12 + col * ((w - 24) / (cols - 1));
            const cy = 8 + row * ((h - 16) / Math.max(rows - 1, 1));
            return (
              <circle
                key={idx}
                cx={cx} cy={cy} r={dotR}
                fill={isDrop ? "#E5E7EB" : "#8B5CF6"}
                opacity={isDrop ? 0.6 : 0.9}
                stroke={isDrop ? "#D1D5DB" : "none"}
                strokeWidth={isDrop ? 1 : 0}
                strokeDasharray={isDrop ? "2 1" : ""}
              />
            );
          })
        )}
      </svg>
      <div className="flex items-center gap-2 mt-1">
        <div className="flex-1 h-1.5 rounded-full bg-[var(--surface-elevated)] overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{ width: `${keepPct}%`, backgroundColor: "#8B5CF6", opacity: 0.85 }}
          />
        </div>
        <span className="text-[10px] text-[var(--foreground-muted)] font-medium">{keepPct}% keep</span>
      </div>
    </div>
  );
}

function DropoutBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const p = Number(data?.params?.p ?? 0.5);

  return (
    <BaseBlock id={id} blockType="Dropout" params={data?.params ?? {}} selected={!!selected}>
      <DropoutViz dropRate={p} />
    </BaseBlock>
  );
}

export const DropoutBlock = memo(DropoutBlockComponent);
