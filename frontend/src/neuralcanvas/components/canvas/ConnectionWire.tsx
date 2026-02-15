"use client";

// ---------------------------------------------------------------------------
// ConnectionWire — v3 Light Theme: clean white pills, colored flow dots
// ---------------------------------------------------------------------------

import { memo, useState, useMemo } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  useNodes,
  Position,
  type EdgeProps,
} from "@xyflow/react";
import { useShapes } from "./ShapeContext";
import {
  getShapeLabel,
  getShapeLabelTooltip,
  type Shape,
  type Dim,
} from "@/neuralcanvas/lib/shapeEngine";
import { BLOCK_REGISTRY, type BlockType } from "@/neuralcanvas/lib/blockRegistry";
import { CANVAS_UI_SCALE, SHAPE_LABEL_SCALE } from "@/neuralcanvas/lib/canvasConstants";

// ---------------------------------------------------------------------------
// Connector dot animation — one source of truth so keyframes match and loop is seamless
// ---------------------------------------------------------------------------
const CONNECTOR_DOT_DASH = 4;
const CONNECTOR_DOT_GAP = 24;
const CONNECTOR_DOT_PERIOD = CONNECTOR_DOT_DASH + CONNECTOR_DOT_GAP; // 28 — keyframes must use this for seamless loop

// ---------------------------------------------------------------------------
// Colors for the three states — light theme
// ---------------------------------------------------------------------------

// Colors now use CSS variables for theme awareness.
// Since these are used inline in SVGs, we read computed values at render time
// via a helper, but for the most part we reference CSS vars directly in JSX.
const COLORS = {
  valid: {
    stroke: "var(--connector-stroke)",
    dotStroke: "var(--accent)",
    bg: "var(--surface)",
    border: "var(--border)",
    text: "var(--foreground)",
    dot: "var(--success)",
  },
  error: {
    stroke: "var(--danger)",
    dotStroke: "var(--danger)",
    bg: "var(--danger-muted)",
    border: "var(--danger)",
    text: "var(--danger)",
    dot: "var(--danger)",
  },
  unknown: {
    stroke: "var(--connector-stroke)",
    dotStroke: "var(--foreground-muted)",
    bg: "var(--surface-elevated)",
    border: "var(--border)",
    text: "var(--foreground-muted)",
    dot: "var(--foreground-muted)",
  },
} as const;

type WireState = keyof typeof COLORS;

// ---------------------------------------------------------------------------
// Human-readable shape description
// ---------------------------------------------------------------------------

function describeShape(shape: Shape | null): string {
  if (!shape || shape.length === 0) return "Shape not yet resolved";

  const rank = shape.length;
  const dimStr = (d: Dim) => (typeof d === "number" ? d.toString() : d);
  const inner = shape.slice(1);

  if (rank === 1) return "Scalar values";
  if (rank === 2) {
    const last = inner[0];
    if (last === "seq") return "Variable-length sequences";
    return `${dimStr(last)} values per sample`;
  }
  if (rank === 3) {
    const [d1, d2] = inner;
    if (d1 === "seq") return `Sequences of ${dimStr(d2)}-dim embeddings`;
    return `${dimStr(d1)} x ${dimStr(d2)} matrix per sample`;
  }
  if (rank === 4) {
    const [c, h, w] = inner;
    return `${dimStr(c)}-channel ${dimStr(h)}x${dimStr(w)} feature maps`;
  }
  return `${rank}D tensor`;
}

// ---------------------------------------------------------------------------
// Get source block accent color for flow dots
// ---------------------------------------------------------------------------

function getSourceColor(sourceType: string | undefined): string {
  if (!sourceType) return "#6366F1";
  const def = BLOCK_REGISTRY[sourceType as BlockType];
  return def?.color ?? "#6366F1";
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

  const sourceResult = shapes.get(source);
  const targetResult = shapes.get(target);
  const outputShape = sourceResult?.outputShape ?? null;
  const shapeLabel = getShapeLabel(outputShape);

  const edgeError = (data?.error as string) || targetResult?.error || "";
  const hasError = !!edgeError;
  const hasShape = outputShape !== null && outputShape.length > 0;

  const wireState: WireState = hasError ? "error" : hasShape ? "valid" : "unknown";
  const colors = COLORS[wireState];

  const sourceNode = nodes.find((n) => n.id === source);
  const targetNode = nodes.find((n) => n.id === target);
  const sourceLabel =
    BLOCK_REGISTRY[sourceNode?.type as BlockType]?.label ?? sourceNode?.type ?? "?";
  const targetLabel =
    BLOCK_REGISTRY[targetNode?.type as BlockType]?.label ?? targetNode?.type ?? "?";
  const accentColor = getSourceColor(sourceNode?.type);

  // Extend path into handles so lines reach the center of the visible circles
  // Handles: 10px box + 2.5px border + 2px shadow each side → radius ~9px
  const HANDLE_INSET = 10;
  const srcPos = (sourcePosition ?? Position.Right) as string;
  const tgtPos = (targetPosition ?? Position.Left) as string;
  const dx = (p: string) => (p === "left" ? -HANDLE_INSET : p === "right" ? HANDLE_INSET : 0);
  const dy = (p: string) => (p === "top" ? -HANDLE_INSET : p === "bottom" ? HANDLE_INSET : 0);
  const insetSourceX = sourceX + dx(srcPos);
  const insetSourceY = sourceY + dy(srcPos);
  const insetTargetX = targetX + dx(tgtPos);
  const insetTargetY = targetY + dy(tgtPos);

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX: insetSourceX,
    sourceY: insetSourceY,
    sourcePosition: sourcePosition ?? Position.Right,
    targetX: insetTargetX,
    targetY: insetTargetY,
    targetPosition: targetPosition ?? Position.Left,
  });

  const description = useMemo(() => {
    if (!hasShape) return "Connect upstream blocks to see the data shape here.";
    return describeShape(outputShape);
  }, [hasShape, outputShape]);

  const s = CANVAS_UI_SCALE * SHAPE_LABEL_SCALE;

  return (
    <>
      {/* Main connection line */}
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          stroke: colors.stroke,
          strokeWidth: selected || hovered ? 2.5 : 1.5,
          strokeLinecap: "round",
          transition: "stroke 0.2s ease, stroke-width 0.2s ease",
        }}
      />

      {/* Animated flow dots on valid connections — short dash + round cap = round dots */}
      {wireState === "valid" && (
        <path
          d={edgePath}
          fill="none"
          stroke={accentColor}
          strokeWidth={4}
          strokeDasharray={`${CONNECTOR_DOT_DASH} ${CONNECTOR_DOT_GAP}`}
          strokeLinecap="round"
          opacity={0.72}
          style={{
            animation: "connectionDotFlow 2s linear infinite",
            animationDelay: `${-(id.split("").reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0) % 2000) / 1000}s`,
          }}
        />
      )}

      {/* Invisible hit area */}
      <path
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={24}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{ cursor: "pointer" }}
      />

      {/* Shape pill badge */}
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
          <div
            className="
              relative rounded-full
              font-mono font-medium leading-none
              border
              transition-all duration-200 ease-out
              select-none cursor-default
            "
            title={getShapeLabelTooltip(outputShape)}
            style={{
              padding: `${4 * s}px ${10 * s}px`,
              fontSize: `${Math.max(11, 11 * s)}px`,
              backgroundColor: colors.bg,
              borderColor: colors.border,
              color: colors.text,
              boxShadow: "0 1px 4px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
              transform: hovered ? "scale(1.06)" : "scale(1)",
            }}
          >
            {/* Status dot */}
            <span
              className="inline-block rounded-full align-middle"
              style={{
                width: 5,
                height: 5,
                marginRight: 5,
                backgroundColor: colors.dot,
                boxShadow: wireState === "valid"
                  ? `0 0 4px ${colors.dot}`
                  : wireState === "error"
                    ? `0 0 4px ${colors.dot}`
                    : "none",
              }}
            />
            {wireState === "error" ? "Mismatch" : shapeLabel}
          </div>

          {/* Hover tooltip */}
          {hovered && (
            <div
              className="absolute left-1/2 -translate-x-1/2 mt-2 z-50 rounded-xl border shadow-lg animate-fade-in"
              style={{
                padding: "10px 14px",
                backgroundColor: "var(--surface)",
                borderColor: "var(--border)",
                minWidth: 170,
                maxWidth: 280,
              }}
            >
              <p className="text-[12px] leading-relaxed mb-1 text-[var(--foreground-secondary)]">
                {description}
              </p>
              <p className="text-[11px] text-[var(--foreground-muted)] font-mono">
                {sourceLabel} → {targetLabel}
              </p>
              {hasError && (
                <p className="text-[11px] text-[var(--danger)] mt-1 leading-snug">
                  {edgeError}
                </p>
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
// Global keyframes
// ---------------------------------------------------------------------------

if (typeof document !== "undefined") {
  const STYLE_ID = "neural-canvas-connection-wire-anim";
  const PERIOD = CONNECTOR_DOT_PERIOD;
  let styleEl = document.getElementById(STYLE_ID) as HTMLStyleElement | null;
  if (!styleEl) {
    styleEl = document.createElement("style");
    styleEl.id = STYLE_ID;
    document.head.appendChild(styleEl);
  }
  // Always sync keyframes so all connectors use the same animation and loop is seamless
  styleEl.textContent = `
    @keyframes connectionDotFlow {
      0%   { stroke-dashoffset: 0; }
      100% { stroke-dashoffset: -${PERIOD}; }
    }
  `;
}
