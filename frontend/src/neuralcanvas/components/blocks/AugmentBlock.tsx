"use client";

/**
 * Augment block — image augmentations (rotate, flip, brightness, etc.).
 * Connect after Input. Eye button opens Augment preview modal (handled in PeepInsideOverlay).
 */

import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";

const AUGMENT_COLOR = "#EA580C";

/** Simple: clean image → noised/augmented image */
function AugmentViz() {
  const w = 140;
  const h = 36;
  const box = 24;
  const y = (h - box) / 2;
  const x1 = 16;
  const arrowGap = 18;
  const x2 = x1 + box + arrowGap;

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {/* Clean image — full orange */}
      <rect x={x1} y={y} width={box} height={box} rx={2} fill={AUGMENT_COLOR} stroke={AUGMENT_COLOR} strokeWidth="1" />
      <line x1={x1 + 4} y1={y + 6} x2={x1 + box - 4} y2={y + 6} stroke="white" strokeWidth="0.8" opacity={0.5} />
      <line x1={x1 + 4} y1={y + 12} x2={x1 + box - 4} y2={y + 12} stroke="white" strokeWidth="0.8" opacity={0.5} />
      <line x1={x1 + 4} y1={y + 18} x2={x1 + box - 4} y2={y + 18} stroke="white" strokeWidth="0.8" opacity={0.5} />
      {/* Short arrow */}
      <line x1={x1 + box + 2} y1={h / 2} x2={x2 - 2} y2={h / 2} stroke={AUGMENT_COLOR} strokeWidth="1" />
      <polygon points={`${x2 - 2},${h / 2 - 2} ${x2},${h / 2} ${x2 - 2},${h / 2 + 2}`} fill={AUGMENT_COLOR} />
      {/* Noised image — full orange + white noise dots (x2 = left edge of second box) */}
      <rect x={x2} y={y} width={box} height={box} rx={2} fill={AUGMENT_COLOR} stroke={AUGMENT_COLOR} strokeWidth="1" />
      <line x1={x2 + 4} y1={y + 6} x2={x2 + box - 4} y2={y + 6} stroke="white" strokeWidth="0.8" opacity={0.5} />
      <line x1={x2 + 4} y1={y + 12} x2={x2 + box - 4} y2={y + 12} stroke="white" strokeWidth="0.8" opacity={0.5} />
      <line x1={x2 + 4} y1={y + 18} x2={x2 + box - 4} y2={y + 18} stroke="white" strokeWidth="0.8" opacity={0.5} />
      {[2, 6, 10, 14, 18, 22].map((px) =>
        [2, 8, 14, 20].map((py) => (
          <circle key={`${px}-${py}`} cx={x2 + px} cy={y + py} r={0.8} fill="white" opacity={0.7} />
        ))
      )}
    </svg>
  );
}

interface BlockData extends Record<string, unknown> {
  params?: Record<string, number | string>;
}

function AugmentBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const params = data?.params ?? {};

  return (
    <BaseBlock id={id} blockType="Augment" params={params} selected={!!selected} data={data}>
      <AugmentViz />
      <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] p-2 mt-1">
        <p className="text-[11px] text-[var(--foreground-muted)]">
          {(() => {
            try {
              const aug = JSON.parse((params.augmentations as string) || "[]");
              const count = Array.isArray(aug) ? aug.filter((a: { enabled?: boolean }) => a.enabled !== false).length : 0;
              return count > 0 ? `${count} augmentation${count !== 1 ? "s" : ""} enabled` : "Click the eye to add augmentations";
            } catch {
              return "Click the eye to add augmentations";
            }
          })()}
        </p>
      </div>
    </BaseBlock>
  );
}

export const AugmentBlock = memo(AugmentBlockComponent);
