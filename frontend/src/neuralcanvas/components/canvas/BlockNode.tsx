"use client";

// ---------------------------------------------------------------------------
// BlockNode — v3 light theme: fallback renderer for any block type
// ---------------------------------------------------------------------------
// This is the generic fallback. The specific block components (LinearBlock,
// Conv2DBlock, etc.) are used when registered in nodeTypes. This component
// renders when a block type doesn't have a dedicated component.
// ---------------------------------------------------------------------------

import { memo, useMemo } from "react";
import { motion } from "framer-motion";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import {
  BLOCK_REGISTRY,
  type BlockType,
  type BlockDefinition,
} from "@/neuralcanvas/lib/blockRegistry";
import { getShapeLabel, getShapeLabelTooltip } from "@/neuralcanvas/lib/shapeEngine";
import { useShapes } from "./ShapeContext";
import { BLOCK_BASE_WIDTH } from "@/neuralcanvas/lib/canvasConstants";
import {
  Inbox,
  Target,
  Rows3,
  Grid3X3,
  RefreshCw,
  Focus,
  SlidersHorizontal,
  BarChart3,
  Zap,
  Shuffle,
  FoldHorizontal,
  Hash,
  Percent,
  type LucideIcon,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Icon lookup
// ---------------------------------------------------------------------------

const ICON_MAP: Record<string, LucideIcon> = {
  inbox: Inbox,
  target: Target,
  "rows-3": Rows3,
  "grid-3x3": Grid3X3,
  "refresh-cw": RefreshCw,
  focus: Focus,
  "sliders-horizontal": SlidersHorizontal,
  "bar-chart-3": BarChart3,
  zap: Zap,
  shuffle: Shuffle,
  "fold-horizontal": FoldHorizontal,
  hash: Hash,
  percent: Percent,
};

const FRIENDLY_NAMES: Record<string, string> = {
  in_features: "in",
  out_features: "out",
  in_channels: "in ch",
  out_channels: "out ch",
  kernel_size: "filter",
  embed_dim: "dim",
  num_heads: "heads",
  activation: "fn",
  p: "drop",
  hidden_size: "hidden",
  num_layers: "layers",
};

function paramSummary(
  def: BlockDefinition,
  params: Record<string, number | string>,
): string {
  if (def.paramSchema.length === 0) return "";
  return def.paramSchema
    .map((s) => {
      const val = params[s.name] ?? def.defaultParams[s.name] ?? "?";
      const name = FRIENDLY_NAMES[s.name] ?? s.name;
      return `${name}: ${val}`;
    })
    .join(" · ");
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface BlockNodeData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

function BlockNodeComponent({ id, type, data, selected }: NodeProps<Node<BlockNodeData>>) {
  const blockType = type as BlockType;
  const def = BLOCK_REGISTRY[blockType];
  const { shapes } = useShapes();
  const result = shapes.get(id);

  const params = data?.params ?? {};
  const Icon = def ? ICON_MAP[def.icon] : null;
  const color = def?.color ?? "#6366F1";
  const animateFromPalette = !!(data as { animateFromPalette?: boolean })?.animateFromPalette;

  const summary = useMemo(
    () => (def ? paramSummary(def, params) : ""),
    [def, params],
  );

  const outLabel = getShapeLabel(result?.outputShape ?? null);
  const hasError = !!result?.error;

  if (!def) {
    return (
      <div className="px-4 py-3 rounded-xl bg-[var(--danger-muted)] border border-[var(--danger)] text-[var(--danger)] text-sm">
        Unknown block: {type}
      </div>
    );
  }

  const blockContent = (
    <div
      className={`
        group relative
        rounded-2xl border bg-[var(--block-surface)]
        transition-all duration-200
        ${selected ? "ring-2 ring-[var(--accent-strong)] shadow-lg scale-[1.01]" : "shadow-[var(--shadow-card)]"}
        ${hasError ? "border-[var(--danger)]" : selected ? "border-[var(--accent)]" : "border-[var(--border)] hover:shadow-[var(--shadow-card-hover)]"}
      `}
      style={{ width: BLOCK_REGISTRY[blockType]?.width ?? BLOCK_BASE_WIDTH }}
    >
      {/* Colored left accent bar */}
      <div
        className="absolute left-0 top-3 bottom-3 w-[3px] rounded-r-full"
        style={{ backgroundColor: color }}
      />

      {/* Header */}
      <div className="flex items-center gap-2.5 px-4 pt-3 pb-2">
        {Icon && (
          <div
            className="flex items-center justify-center w-7 h-7 rounded-lg shrink-0"
            style={{ backgroundColor: `${color}12` }}
          >
            <Icon size={14} style={{ color }} />
          </div>
        )}
        <span className="text-[13px] font-bold text-[var(--foreground)] truncate flex-1" style={{ color }}>
          {def.label}
        </span>
        <span
          className={`text-[10px] font-mono px-2 py-0.5 rounded-md ${
            hasError ? "bg-[var(--danger-muted)] text-[var(--danger)]" : "bg-[var(--surface-elevated)] text-[var(--foreground-muted)]"
          }`}
          title={getShapeLabelTooltip(result?.outputShape ?? null) || undefined}
        >
          {outLabel}
        </span>
      </div>

      {/* Body */}
      <div className="px-4 pb-3 space-y-1">
        {summary && (
          <p className="text-[11px] text-[var(--foreground-muted)] font-mono leading-relaxed truncate">
            {summary}
          </p>
        )}
        {hasError && (
          <p className="text-[11px] text-[var(--danger)] leading-snug line-clamp-2">
            {result?.error}
          </p>
        )}
      </div>

      {/* Input handles */}
      {def.inputPorts.map((port, i) => {
        const topPct = def.inputPorts.length === 1
          ? 50
          : 25 + (i / Math.max(def.inputPorts.length - 1, 1)) * 50;
        return (
          <Handle
            key={port.id}
            id={port.id}
            type="target"
            position={Position.Left}
            className="!transition-all !duration-200"
            style={{
              top: `${topPct}%`,
              width: 10,
              height: 10,
              background: hasError ? "var(--danger)" : "var(--block-surface)",
              border: `2.5px solid ${hasError ? "var(--danger)" : color}`,
              boxShadow: `0 0 0 2px var(--block-surface), 0 1px 3px rgba(0,0,0,0.1)`,
            }}
          />
        );
      })}

      {/* Output handles */}
      {def.outputPorts.map((port, i) => {
        const topPct = def.outputPorts.length === 1
          ? 50
          : 25 + (i / Math.max(def.outputPorts.length - 1, 1)) * 50;
        return (
          <Handle
            key={port.id}
            id={port.id}
            type="source"
            position={Position.Right}
            className="!transition-all !duration-200"
            style={{
              top: `${topPct}%`,
              width: 10,
              height: 10,
              background: hasError ? "var(--danger)" : color,
              border: `2.5px solid var(--block-surface)`,
              boxShadow: `0 0 0 2px var(--block-surface), 0 1px 3px rgba(0,0,0,0.1)`,
            }}
          />
        );
      })}
    </div>
  );

  if (animateFromPalette) {
    return (
      <motion.div
        initial={{ opacity: 0, x: -120 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
        style={{ width: "100%" }}
      >
        {blockContent}
      </motion.div>
    );
  }
  return blockContent;
}

export const BlockNode = memo(BlockNodeComponent);
