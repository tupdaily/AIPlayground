"use client";

// ---------------------------------------------------------------------------
// ConnectionWire — custom React Flow edge with animated bezier, shape badge,
// three-state colour coding, and rich hover tooltip.
// ---------------------------------------------------------------------------

import { memo, useState, useMemo } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useNodes,
  type EdgeProps,
} from "@xyflow/react";
import { useShapes } from "./ShapeContext";
import {
  getShapeLabel,
  type Shape,
  type Dim,
} from "@/neuralcanvas/lib/shapeEngine";
import { BLOCK_REGISTRY, type BlockType } from "@/neuralcanvas/lib/blockRegistry";
import { CANVAS_UI_SCALE, SHAPE_LABEL_SCALE } from "@/neuralcanvas/lib/canvasConstants";

// ---------------------------------------------------------------------------
// Colour constants for the three states
// ---------------------------------------------------------------------------

const COLORS = {
  valid: { stroke: "#4b5563", glow: "#22c55e20", bg: "rgba(22,101,52,0.85)", border: "rgba(34,197,94,0.5)", text: "#bbf7d0" },
  error: { stroke: "#ef4444", glow: "#ef444420", bg: "rgba(69,10,10,0.90)", border: "rgba(239,68,68,0.5)", text: "#fecaca" },
  unknown: { stroke: "#4b5563", glow: "#6b728020", bg: "rgba(31,41,55,0.85)", border: "rgba(107,114,128,0.4)", text: "#9ca3af" },
} as const;

type WireState = keyof typeof COLORS;

// ---------------------------------------------------------------------------
// Human-readable shape description
// ---------------------------------------------------------------------------

function describeShape(shape: Shape | null): string {
  if (!shape || shape.length === 0) return "Unknown shape";

  const rank = shape.length;
  const dimStr = (d: Dim) => (typeof d === "number" ? d.toString() : d);

  // Skip batch dim for the description.
  const inner = shape.slice(1);

  if (rank === 1) return `Scalar values`;
  if (rank === 2) {
    const last = inner[0];
    if (last === "seq") return "Variable-length sequences";
    return `Batch of ${dimStr(last)}-dimensional vectors`;
  }
  if (rank === 3) {
    const [d1, d2] = inner;
    if (d1 === "seq") return `Sequences of ${dimStr(d2)}-dimensional embeddings`;
    return `Batch of ${dimStr(d1)}×${dimStr(d2)} matrices`;
  }
  if (rank === 4) {
    const [c, h, w] = inner;
    return `${dimStr(c)}-channel feature maps (${dimStr(h)}×${dimStr(w)})`;
  }
  return `${rank}D tensor`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function ConnectionWireComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  source,
  target,
  data,
  style = {},
  markerEnd,
  selected,
}: EdgeProps) {
  const { shapes } = useShapes();
  const nodes = useNodes();
  const [hovered, setHovered] = useState(false);

  // ── Resolve shape data ──
  const sourceResult = shapes.get(source);
  const targetResult = shapes.get(target);
  const outputShape = sourceResult?.outputShape ?? null;
  const shapeLabel = getShapeLabel(outputShape);

  // Error from validation at connection time OR from live propagation.
  const edgeError = (data?.error as string) || targetResult?.error || "";
  const hasError = !!edgeError;
  const hasShape = outputShape !== null && outputShape.length > 0;

  // Determine wire state.
  const wireState: WireState = hasError ? "error" : hasShape ? "valid" : "unknown";
  const colors = COLORS[wireState];

  // ── Node labels for the tooltip ──
  const sourceNode = nodes.find((n) => n.id === source);
  const targetNode = nodes.find((n) => n.id === target);
  const sourceLabel =
    BLOCK_REGISTRY[sourceNode?.type as BlockType]?.label ?? sourceNode?.type ?? "?";
  const targetLabel =
    BLOCK_REGISTRY[targetNode?.type as BlockType]?.label ?? targetNode?.type ?? "?";

  // ── Bezier path ──
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  // ── Human-readable description for tooltip ──
  const description = useMemo(() => {
    if (!hasShape) return "Shape not yet resolved — connect upstream blocks.";
    const shapeDesc = describeShape(outputShape);
    return `${shapeDesc} flowing from ${sourceLabel} to ${targetLabel}`;
  }, [hasShape, outputShape, sourceLabel, targetLabel]);

  return (
    <>
      {/* ── Solid connection line (no animation) ── */}
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          stroke: colors.stroke,
          strokeWidth: selected ? 2 : 1.5,
          strokeLinecap: "round",
          transition: "stroke 0.2s ease, stroke-width 0.2s ease",
        }}
      />

      {/* ── Invisible hit area for hover (wider than visible stroke) ── */}
      <path
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={20}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{ cursor: "pointer" }}
      />

      {/* ── Shape pill badge at midpoint ── */}
      <EdgeLabelRenderer>
        <div
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            pointerEvents: "all",
            transition: "transform 0.2s ease",
          }}
          className="nodrag nopan"
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
        >
          {/* Pill badge — scaled smaller than blocks */}
          <div
            className="
              relative rounded-full
              font-mono font-medium leading-none
              backdrop-blur-md shadow-lg
              border
              transition-all duration-300 ease-out
              select-none cursor-default
            "
            style={{
              padding: `${4 * CANVAS_UI_SCALE * SHAPE_LABEL_SCALE}px ${8 * CANVAS_UI_SCALE * SHAPE_LABEL_SCALE}px`,
              fontSize: `${9 * CANVAS_UI_SCALE * SHAPE_LABEL_SCALE}px`,
              backgroundColor: colors.bg,
              borderColor: colors.border,
              color: colors.text,
              boxShadow: `0 0 8px ${colors.glow}, 0 2px 8px rgba(0,0,0,0.3)`,
              transform: hovered ? "scale(1.08)" : "scale(1)",
            }}
          >
            {/* Dot indicator */}
            <span
              className="inline-block rounded-full align-middle"
              style={{
                width: 3 * CANVAS_UI_SCALE * SHAPE_LABEL_SCALE,
                height: 3 * CANVAS_UI_SCALE * SHAPE_LABEL_SCALE,
                marginRight: 4 * CANVAS_UI_SCALE * SHAPE_LABEL_SCALE,
                backgroundColor: colors.stroke,
                boxShadow: `0 0 4px ${colors.stroke}`,
              }}
            />
            {wireState === "error" ? "⚠ mismatch" : shapeLabel}
          </div>

          {/* ── Compact hover tooltip ── */}
          {hovered && (
            <div
              className="absolute left-1/2 -translate-x-1/2 mt-1.5 z-50 rounded-md border shadow-lg backdrop-blur-sm animate-fade-in max-w-[200px] truncate"
              style={{
                padding: "4px 8px",
                backgroundColor: colors.bg,
                borderColor: colors.border,
                fontSize: "10px",
                color: colors.text,
              }}
              title={description}
            >
              <span className="font-mono font-medium">{shapeLabel}</span>
              <span className="text-neutral-500 mx-1.5">·</span>
              <span className="text-neutral-500 font-mono text-[9px]">
                {sourceLabel} → {targetLabel}
              </span>
              {hasError && (
                <>
                  <span className="text-neutral-600 mx-1.5">·</span>
                  <span className="text-red-400 text-[9px] truncate">{edgeError}</span>
                </>
              )}
            </div>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

export const ConnectionWire = memo(ConnectionWireComponent);

// ---------------------------------------------------------------------------
// Global keyframes (injected once)
// ---------------------------------------------------------------------------

if (typeof document !== "undefined") {
  const STYLE_ID = "neural-canvas-connection-wire-anim";
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      @keyframes connectionFlowDash {
        to { stroke-dashoffset: -10; }
      }
      @keyframes connectionDotFlow {
        0%   { stroke-dashoffset: 0; }
        100% { stroke-dashoffset: -60; }
      }
    `;
    document.head.appendChild(style);
  }
}
