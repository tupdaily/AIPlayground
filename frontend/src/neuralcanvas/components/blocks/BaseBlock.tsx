"use client";

// ---------------------------------------------------------------------------
// BaseBlock — v3 Light Theme: white card, auto-height, colored accent bar,
// unique visual content per block type, params shown as friendly pills
// ---------------------------------------------------------------------------

import {
  memo,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
  type ChangeEvent,
} from "react";
import { motion } from "framer-motion";
import { Handle, Position, useReactFlow } from "@xyflow/react";
import {
  BLOCK_REGISTRY,
  type BlockType,
  type BlockDefinition,
  type ParamSchema,
} from "@/neuralcanvas/lib/blockRegistry";
import { CANVAS_UI_SCALE, BLOCK_BASE_WIDTH } from "@/neuralcanvas/lib/canvasConstants";
import { getShapeLabel, getShapeLabelTooltip } from "@/neuralcanvas/lib/shapeEngine";
import { useShapes } from "@/neuralcanvas/components/canvas/ShapeContext";
import { usePeepInsideContext } from "@/neuralcanvas/components/peep-inside/PeepInsideContext";
import {
  Inbox,
  Target,
  Type,
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
  MapPin,
  Percent,
  Plus,
  Combine,
  Eye,
  AlertCircle,
  ChevronUp,
  ChevronDown,
  Upload,
  Monitor,
  PenTool,
  type LucideIcon,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Icon map
// ---------------------------------------------------------------------------

export const ICON_MAP: Record<string, LucideIcon> = {
  inbox: Inbox,
  target: Target,
  type: Type,
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
  "map-pin": MapPin,
  percent: Percent,
  plus: Plus,
  merge: Combine,
  upload: Upload,
  monitor: Monitor,
  "pen-tool": PenTool,
};

// ---------------------------------------------------------------------------
// Human-friendly parameter names
// ---------------------------------------------------------------------------

const FRIENDLY_PARAM_NAMES: Record<string, string> = {
  in_features: "Input size",
  out_features: "Output size",
  in_channels: "Input channels",
  out_channels: "Output channels",
  kernel_size: "Filter size",
  stride: "Step size",
  padding: "Padding",
  embed_dim: "Embedding size",
  num_heads: "Attention heads",
  activation: "Function",
  p: "Drop rate",
  hidden_size: "Hidden size",
  num_layers: "Layers",
  num_embeddings: "Vocabulary size",
  embedding_dim: "Vector size",
  max_len: "Max length",
  dim: "Dimension",
  normalized_shape: "Norm size",
  num_features: "Features",
  vocab_size: "Vocabulary",
  input_size: "Input size",
  d_model: "Model dim",
  batch_size: "Batch size",
  seq_len: "Sequence length",
};

// ---------------------------------------------------------------------------
// Number input with steppers
// ---------------------------------------------------------------------------

interface NumberParamProps {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  color: string;
  onChange: (v: number) => void;
}

function NumberParam({ value, min, max, step = 1, color, onChange }: NumberParamProps) {
  const clamp = useCallback(
    (v: number) => {
      let n = v;
      if (min !== undefined) n = Math.max(min, n);
      if (max !== undefined) n = Math.min(max, n);
      return n;
    },
    [min, max],
  );

  const [localValue, setLocalValue] = useState<string>(() => String(value));
  const isFocusedRef = useRef(false);
  useEffect(() => {
    if (!isFocusedRef.current) setLocalValue(String(value));
  }, [value]);

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const raw = e.target.value;
      setLocalValue(raw);
      if (raw === "" || raw === "-") return;
      const parsed = step < 1 ? parseFloat(raw) : parseInt(raw, 10);
      if (Number.isFinite(parsed)) onChange(clamp(parsed));
    },
    [onChange, clamp, step],
  );

  const handleFocus = useCallback(() => {
    isFocusedRef.current = true;
  }, []);

  const handleBlur = useCallback(() => {
    isFocusedRef.current = false;
    const trimmed = localValue.trim();
    if (trimmed === "" || trimmed === "-") {
      const fallback = min !== undefined ? min : 0;
      onChange(clamp(fallback));
      setLocalValue(String(fallback));
      return;
    }
    const parsed = step < 1 ? parseFloat(trimmed) : parseInt(trimmed, 10);
    if (Number.isFinite(parsed)) {
      const clamped = clamp(parsed);
      onChange(clamped);
      setLocalValue(String(clamped));
    } else {
      setLocalValue(String(value));
    }
  }, [localValue, min, step, value, onChange, clamp]);

  return (
    <div className="flex items-center gap-0.5">
      <input
        type="number"
        value={localValue}
        onChange={handleChange}
        onFocus={handleFocus}
        onBlur={handleBlur}
        min={min}
        max={max}
        step={step}
        className="
          nodrag nopan
          w-[52px] px-1.5 py-0.5 rounded-md text-[12px] font-mono text-center
          bg-[var(--surface-elevated)] border border-[var(--border)]
          text-[var(--foreground)]
          outline-none focus:border-indigo-300 focus:ring-1 focus:ring-indigo-100
          transition-colors duration-100
          [appearance:textfield]
          [&::-webkit-outer-spin-button]:appearance-none
          [&::-webkit-inner-spin-button]:appearance-none
        "
      />
      <div className="flex flex-col -space-y-px">
        <button
          className="nodrag nopan p-0 text-[var(--foreground-muted)] hover:text-[var(--foreground)] transition-colors"
          onClick={() => onChange(clamp(value + step))}
          tabIndex={-1}
        >
          <ChevronUp size={10} />
        </button>
        <button
          className="nodrag nopan p-0 text-[var(--foreground-muted)] hover:text-[var(--foreground)] transition-colors"
          onClick={() => onChange(clamp(value - step))}
          tabIndex={-1}
        >
          <ChevronDown size={10} />
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Select input
// ---------------------------------------------------------------------------

interface SelectParamProps {
  value: string;
  options: string[];
  color: string;
  onChange: (v: string) => void;
}

function SelectParam({ value, options, onChange }: SelectParamProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="
        nodrag nopan
        w-[72px] px-1.5 py-0.5 rounded-md text-[12px] font-mono
        bg-[var(--surface-elevated)] border border-[var(--border)]
        text-[var(--foreground)]
        outline-none focus:border-indigo-300 focus:ring-1 focus:ring-indigo-100
        transition-colors duration-100 cursor-pointer
      "
    >
      {options.map((opt) => (
        <option key={opt} value={opt}>{opt}</option>
      ))}
    </select>
  );
}

// ---------------------------------------------------------------------------
// Param row
// ---------------------------------------------------------------------------

interface ParamRowProps {
  schema: ParamSchema;
  value: number | string;
  color: string;
  onUpdate: (name: string, value: number | string) => void;
}

function ParamRow({ schema, value, color, onUpdate }: ParamRowProps) {
  const friendlyName = FRIENDLY_PARAM_NAMES[schema.name] ?? schema.name;

  const renderInput = () => {
    if (schema.type === "select" && schema.options) {
      return (
        <SelectParam
          value={String(value)}
          options={schema.options}
          color={color}
          onChange={(v) => onUpdate(schema.name, v)}
        />
      );
    }
    const numVal = typeof value === "number" ? value : parseFloat(String(value)) || 0;
    const step = schema.type === "float" ? 0.05 : 1;
    return (
      <NumberParam
        value={numVal}
        min={schema.min}
        max={schema.max}
        step={step}
        color={color}
        onChange={(v) => onUpdate(schema.name, v)}
      />
    );
  };

  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-[11px] text-[var(--foreground-muted)] font-medium truncate" title={`${friendlyName} (${schema.name})`}>
        {friendlyName}
      </span>
      {renderInput()}
    </div>
  );
}

// ---------------------------------------------------------------------------
// BaseBlock props
// ---------------------------------------------------------------------------

export interface BaseBlockProps {
  id: string;
  blockType: BlockType;
  params: Record<string, number | string>;
  selected: boolean;
  /** Node data (e.g. for animateFromPalette entrance animation) */
  data?: Record<string, unknown>;
  /** Unique visual content (SVG illustration) injected by each block type */
  children?: ReactNode;
}

// ---------------------------------------------------------------------------
// BaseBlock component — v3 white card design
// ---------------------------------------------------------------------------

function BaseBlockComponent({
  id,
  blockType,
  params,
  selected,
  data,
  children,
}: BaseBlockProps) {
  const def: BlockDefinition | undefined = BLOCK_REGISTRY[blockType];
  const { shapes } = useShapes();
  const result = shapes.get(id);
  const { setNodes } = useReactFlow();
  const [errorTooltip, setErrorTooltip] = useState(false);
  const [showShape, setShowShape] = useState(false);
  const { open: openPeep } = usePeepInsideContext();

  const handlePeepInside = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      openPeep({
        blockId: id,
        blockType,
        anchorX: rect.right,
        anchorY: rect.top,
        activationType:
          blockType === "Activation"
            ? String(params.activation ?? "")
            : undefined,
        params: { ...params },
      });
    },
    [id, blockType, params, openPeep],
  );

  const color = def?.color ?? "#6366F1";
  const Icon = def ? ICON_MAP[def.icon] : null;
  const hasError = !!result?.error;
  const inLabel = getShapeLabel(result?.inputShape ?? null);
  const outLabel = getShapeLabel(result?.outputShape ?? null);

  const onParamUpdate = useCallback(
    (name: string, value: number | string) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prevParams = (n.data?.params && typeof n.data.params === "object") ? n.data.params as Record<string, number | string> : {};
          return {
            ...n,
            data: {
              ...n.data,
              params: { ...prevParams, [name]: value },
            },
          };
        }),
      );
    },
    [id, setNodes],
  );

  if (!def) {
    return (
      <div className="px-4 py-3 rounded-xl bg-[var(--danger-muted)] border border-[var(--danger)] text-[var(--danger)] text-sm">
        Unknown block: {blockType}
      </div>
    );
  }

  const animateFromPalette = !!(data as { animateFromPalette?: boolean } | undefined)?.animateFromPalette;
  const blockContent = (
    <div
      className={`
        group/block relative flex flex-col
        bg-[var(--block-surface)] rounded-2xl
        border transition-all duration-200
        ${selected ? "ring-2 shadow-lg scale-[1.01]" : "shadow-[var(--shadow-card)]"}
        ${hasError ? "border-[var(--danger)] ring-[var(--danger-strong)]" : selected ? "border-[var(--accent)] ring-[var(--accent-strong)]" : "border-[var(--border)] hover:shadow-[var(--shadow-card-hover)]"}
      `}
      style={{ width: def.width ?? BLOCK_BASE_WIDTH }}
    >
      {/* ── Colored left accent bar ── */}
      <div
        className="absolute left-0 top-3 bottom-3 w-[3px] rounded-r-full"
        style={{ backgroundColor: color }}
      />

      {/* ═══ Header ═══ */}
      <div className="flex items-center gap-2 px-4 pt-3 pb-2">
        {Icon && (
          <div
            className="flex items-center justify-center w-7 h-7 rounded-lg shrink-0"
            style={{ backgroundColor: `${color}12` }}
          >
            <Icon size={14} style={{ color }} />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <span className="text-[13px] font-bold text-[var(--foreground)] truncate block leading-tight">
            {def.label}
          </span>
          <span className="text-[10px] font-medium uppercase tracking-wider block leading-tight" style={{ color }}>
            {def.category}
          </span>
        </div>
        {/* Peep inside button — visible on hover */}
        <button
          className="
            nodrag nopan
            flex items-center justify-center w-6 h-6 rounded-lg shrink-0
            text-[var(--foreground-muted)] hover:text-[var(--accent)]
            bg-transparent hover:bg-[var(--accent-muted)]
            transition-all duration-150
            opacity-0 group-hover/block:opacity-100
          "
          title="Peep inside this block"
          onClick={handlePeepInside}
        >
          <Eye size={13} />
        </button>
      </div>

      {/* ═══ Visual content area (unique per block type) ═══ */}
      {children && (
        <div className="px-4 pb-2">
          {children}
        </div>
      )}

      {/* ═══ Parameters ═══ */}
      {def.paramSchema.length > 0 && (
        <div className="px-4 py-2 space-y-1.5 border-t border-[var(--border)]">
          {def.paramSchema.map((schema) => (
            <ParamRow
              key={schema.name}
              schema={schema}
              value={params[schema.name] ?? def.defaultParams[schema.name] ?? 0}
              color={color}
              onUpdate={onParamUpdate}
            />
          ))}
        </div>
      )}

      {/* ═══ Shape bar — shown on hover ═══ */}
      <div
        className={`
          px-4 py-1.5 text-[11px] font-mono border-t transition-all duration-200
          ${hasError ? "border-[var(--danger)] bg-[var(--danger-muted)]" : "border-[var(--border)] bg-[var(--surface-elevated)]"}
          ${!hasError && !showShape ? "opacity-60 group-hover/block:opacity-100" : ""}
          rounded-b-2xl
        `}
        onMouseEnter={() => setShowShape(true)}
        onMouseLeave={() => setShowShape(false)}
      >
        {hasError ? (
          <div className="flex items-center gap-1.5">
            <div className="relative">
              <button
                className="nodrag nopan text-[var(--danger)] hover:opacity-80 transition-colors"
                onClick={() => setErrorTooltip((v) => !v)}
              >
                <AlertCircle size={12} />
              </button>
              {errorTooltip && (
                <div className="absolute bottom-full left-0 mb-2 z-50 w-64 p-3 rounded-xl bg-[var(--surface)] border border-[var(--danger)] text-[12px] text-[var(--danger)] leading-relaxed shadow-lg">
                  {result?.error}
                </div>
              )}
            </div>
            <span className="text-[var(--danger)] truncate flex-1">{result?.error}</span>
          </div>
        ) : (
          <div
            className="flex items-center gap-1.5"
            title={getShapeLabelTooltip(result?.inputShape ?? null) || getShapeLabelTooltip(result?.outputShape ?? null) || undefined}
          >
            <span className="text-[var(--foreground-muted)]">{inLabel}</span>
            <span className="text-[var(--foreground-muted)]">→</span>
            <span className="font-medium" style={{ color }}>{outLabel}</span>
          </div>
        )}
      </div>

      {/* ═══ Handles ═══ */}
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

  // Opacity-only animation: avoids translateX/translateY which would shift handles
  // during animation and break React Flow's cached handle positions (connections
  // would stop short of the visible handle circles).
  if (animateFromPalette) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        style={{ width: "100%" }}
      >
        {blockContent}
      </motion.div>
    );
  }
  return blockContent;
}

export const BaseBlock = memo(BaseBlockComponent);
