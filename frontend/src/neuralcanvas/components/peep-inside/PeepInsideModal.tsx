"use client";

// ---------------------------------------------------------------------------
// PeepInsideModal — v2: beginner-friendly X-ray view inside any block
// ---------------------------------------------------------------------------
// Changes:
// - "What am I looking at?" section at top of each tab
// - Larger font sizes (min 12px)
// - Cleaner styling with new design system
// - Better empty states with actionable guidance
// ---------------------------------------------------------------------------

import { memo, useMemo, useRef, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  RefreshCw,
  Radio,
  Eye,
  Layers,
  Activity,
  BarChart3,
  Grid3X3,
  ScanEye,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Database,
  ImageIcon,
  Type,
  Info,
} from "lucide-react";
import { BLOCK_REGISTRY, type BlockType } from "@/neuralcanvas/lib/blockRegistry";
import { getShapeLabel, getShapeLabelTooltip } from "@/neuralcanvas/lib/shapeEngine";
import { useShapes } from "@/neuralcanvas/components/canvas/ShapeContext";
import { usePeepInside } from "@/neuralcanvas/hooks/usePeepInside";
import { ICON_MAP } from "@/neuralcanvas/components/blocks/BaseBlock";
import { HeatmapViz } from "./HeatmapViz";
import { WeightHeatmap } from "./WeightHeatmap";
import { BarChartViz } from "./BarChartViz";
import { GradientFlowViz } from "./GradientFlowViz";
import { useGradientFlow } from "./GradientFlowContext";
import { ActivationHistogram } from "./ActivationHistogram";
import { AttentionHeatmap } from "./AttentionHeatmap";
import { FilterGrid } from "./FilterGrid";

// ---------------------------------------------------------------------------
// Tab system
// ---------------------------------------------------------------------------

interface TabDef {
  id: string;
  label: string;
  icon: typeof Layers;
  /** Beginner-friendly explanation of what this tab shows */
  explanation: string;
}

/** Blocks that only deal with data (no trainable weights). Peep shows a Data tab only. */
const DATA_SOURCE_BLOCKS: BlockType[] = ["Input", "InputSpace", "Board", "TextInput"];

/** Sink/endpoint blocks: single About tab only (no weights, activations, or gradients). */
const INFO_ONLY_BLOCKS: BlockType[] = ["Display", "Output"];

/** Pure reshape: no weights, no activation; activations/gradients just pass through in a new shape. Single About tab only. */
const PASSTHROUGH_ONLY_BLOCKS: BlockType[] = ["Flatten"];

/** Blocks that have no trainable weights; show Activations and Gradients only (not Display/Output). Excludes passthrough-only (e.g. Flatten). */
const NO_WEIGHTS_BLOCKS: BlockType[] = [
  "Add", "Concat", "Dropout", "Activation", "LayerNorm", "BatchNorm",
  "PositionalEncoding", "PositionalEmbedding", "Softmax",
];

/** Gradient tendency: shown only in Peep Inside (eye) for this block, not on the canvas. */
type GradientTendency = "vanishing" | "exploding" | "healthy" | "stabilizing";

function getGradientProneness(
  blockType: BlockType,
  activationType?: string,
): { tendency: GradientTendency; message: string } | undefined {
  if (blockType === "Activation") {
    const act = (activationType ?? "").toLowerCase();
    if (act === "sigmoid" || act === "tanh") {
      return {
        tendency: "vanishing",
        message: "This activation is prone to vanishing gradients: its slope is small for large |x|, so the learning signal gets squashed and can die out in deep networks.",
      };
    }
    if (act === "relu" || act === "gelu" || act === "leaky_relu") {
      return {
        tendency: "healthy",
        message: "ReLU (and GELU, Leaky ReLU) does not cause vanishing gradients: the gradient is 0 or 1 (or close), so it doesn’t shrink the signal like sigmoid/tanh. Exploding gradients usually come from weight scale and depth, not from this activation.",
      };
    }
    return {
      tendency: "healthy",
      message: "Gradient flow here depends on the exact activation. Vanishing is common with sigmoid/tanh; ReLU-style activations don’t cause it. Exploding is usually from weight scale and depth, not the activation.",
    };
  }
  const byBlock: Partial<Record<BlockType, { tendency: GradientTendency; message: string }>> = {
    LayerNorm: {
      tendency: "stabilizing",
      message: "This block helps stabilize gradients and can reduce both vanishing and exploding gradients, which is why it’s common in deep and transformer models.",
    },
    BatchNorm: {
      tendency: "stabilizing",
      message: "Batch normalization recenters and rescales activations, which helps keep gradients in a healthy range and can reduce exploding gradients.",
    },
    Add: {
      tendency: "healthy",
      message: "Residual (skip) connections pass the gradient through directly, which helps avoid vanishing gradients in very deep networks.",
    },
    LSTM: {
      tendency: "vanishing",
      message: "LSTMs can still suffer from vanishing gradients on long sequences, but gating helps compared to plain RNNs.",
    },
    Dropout: {
      tendency: "healthy",
      message: "Dropout scales activations at training time; gradients still flow through the kept units. It doesn’t cause vanishing by itself.",
    },
    Softmax: {
      tendency: "healthy",
      message: "At the output, softmax gradients can be small when the model is confident; that’s normal. Deeper layers are where vanishing usually matters more.",
    },
    Attention: {
      tendency: "healthy",
      message: "Attention uses skip-like connections (query, key, value paths), which helps gradients flow and reduces vanishing in deep transformers.",
    },
  };
  if (byBlock[blockType]) return byBlock[blockType];
  // Blocks with a Gradients tab but no specific note: short default.
  const hasGradientsTab =
    NO_WEIGHTS_BLOCKS.includes(blockType) ||
    ["Linear", "Conv2D", "Embedding", "TextEmbedding"].includes(blockType);
  if (hasGradientsTab) {
    return {
      tendency: "healthy" as const,
      message: "Gradient flow at this layer is usually in a normal range. Run training to see real gradient values here.",
    };
  }
  return undefined;
}

function getTabsForBlockType(blockType: BlockType): TabDef[] {
  // Info-only blocks (Display, Output): single About tab
  if (INFO_ONLY_BLOCKS.includes(blockType)) {
    const aboutExplanations: Record<string, string> = {
      Display: "This block shows the model’s prediction as a class label when connected to an Output block. It has no learnable parameters and no internal activations to inspect — it just displays whatever the Output passes through.",
      Output: "This block is the end of the model. It passes through the tensor (e.g. logits) from the previous layer. Connect it to Display to see the predicted class, or use it as the model’s final output. No weights or gradients to inspect here.",
    };
    return [
      {
        id: "about",
        label: "About",
        icon: Info,
        explanation: aboutExplanations[blockType] ?? BLOCK_REGISTRY[blockType]?.description ?? "About this block.",
      },
    ];
  }

  // Passthrough/reshape-only blocks (e.g. Flatten): no weights, no activation; values just pass through. Single About tab.
  if (PASSTHROUGH_ONLY_BLOCKS.includes(blockType)) {
    const passthroughExplanations: Record<string, string> = {
      Flatten:
        "This block only reshapes the tensor (e.g. from a 2D grid to a 1D vector). It has no weights and no activation function — the numbers are unchanged, just rearranged. Activations and gradients simply pass through in a different shape, so there's nothing extra to inspect here.",
    };
    return [
      {
        id: "about",
        label: "About",
        icon: Info,
        explanation: passthroughExplanations[blockType] ?? BLOCK_REGISTRY[blockType]?.description ?? "About this block.",
      },
    ];
  }

  // Data-source blocks: only Data tab (shape, dataset, uploaded image, etc.)
  if (DATA_SOURCE_BLOCKS.includes(blockType)) {
    const dataExplanations: Record<string, string> = {
      Input: "This block selects which dataset feeds the model. When an Input Space or Board block is connected, it uses the uploaded, drawn, or captured data instead. The shape below is what the next layer will receive.",
      InputSpace: "Custom data you upload or capture (image, CSV, or text). The preview shows exactly what will be sent into the model. Use this to try your own inputs without changing the dataset.",
      Board: "The drawing canvas. Whatever you draw is resized to the width×height below and sent as a single-channel image when this block is connected to Input. Great for testing digit or shape classifiers.",
      TextInput: "Token IDs for text sequences. The shape [batch, seq_len] is fixed here; the next block (e.g. Text Embedding) turns these integers into vectors.",
    };
    return [
      {
        id: "data",
        label: "Data",
        icon: Database,
        explanation: dataExplanations[blockType] ?? "Data and shape for this block.",
      },
    ];
  }

  // No-weights blocks: Activations and Gradients only (no Weights tab)
  if (NO_WEIGHTS_BLOCKS.includes(blockType)) {
    return [
      {
        id: "activations",
        label: "Activations",
        icon: Activity,
        explanation: "Activations show what this layer outputs when data passes through it. This block has no learnable weights — it only transforms or combines its inputs.",
      },
      {
        id: "gradients",
        label: "Gradients",
        icon: BarChart3,
        explanation: "Gradient flow is the learning signal that moves backward through the model during training. This block has no weights, but the signal still passes through. Checking it here helps you see if the signal is healthy or dying (vanishing) / blowing up (exploding) somewhere in the network.",
      },
    ];
  }

  // Weight blocks: Weights, Activations, Gradients; + optional Attention / Filters
  const base: TabDef[] = [
    {
      id: "weights",
      label: "Weights",
      icon: Layers,
      explanation: "Weights are the numbers the model learns during training. They determine how strongly each input affects the output. Before training, these are random — after training, they encode patterns the model has discovered.",
    },
    {
      id: "activations",
      label: "Activations",
      icon: Activity,
      explanation: "Activations show what this layer outputs when data passes through it. They reveal which features the model is detecting. Healthy activations have a spread of values — if most are zero or all the same, something may be wrong.",
    },
    {
      id: "gradients",
      label: "Gradients",
      icon: BarChart3,
      explanation: "Gradient flow is the “learning signal” that travels backward through the network during training. It tells each weight how much to change. If this signal is too weak (vanishing), later layers barely learn; if it’s too strong (exploding), training blows up. This tab shows whether gradients at this block are in a healthy range.",
    },
  ];

  if (blockType === "Attention") {
    base.push({
      id: "attention",
      label: "Attention",
      icon: ScanEye,
      explanation: "The attention map shows which parts of the input each position is paying attention to. Brighter spots mean stronger attention. Different heads often learn to focus on different patterns.",
    });
  }
  if (blockType === "Conv2D") {
    base.push({
      id: "filters",
      label: "Filters",
      icon: Grid3X3,
      explanation: "Filters are small patterns the convolutional layer has learned to detect — like edges, textures, or shapes. After training, each filter specializes in recognizing a specific visual feature.",
    });
  }

  return base;
}

// ---------------------------------------------------------------------------
// Data tab content (Input / InputSpace / TextInput)
// ---------------------------------------------------------------------------

function DataTabContent({
  blockType,
  params,
  outLabel,
  color,
}: {
  blockType: BlockType;
  params: Record<string, number | string>;
  outLabel: string;
  color: string;
}) {
  if (blockType === "Input") {
    const datasetId = (params.dataset_id as string) ?? "";
    const shape = (params.input_shape as string) ?? "—";
    const isCustom = datasetId === "__custom__" || !datasetId;
    return (
      <div className="space-y-4">
        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] p-4 space-y-2">
          <div className="flex items-center gap-2">
            <Database size={16} style={{ color }} />
            <span className="text-[12px] font-medium text-[var(--foreground)]">Dataset</span>
          </div>
          <p className="text-[12px] text-[var(--foreground-muted)]">
            {isCustom ? "Custom (from Input Space when connected)" : `Selected: ${datasetId}`}
          </p>
          <div className="flex flex-wrap gap-2 items-center">
            <span className="text-[11px] text-[var(--foreground-faint)]">Output shape</span>
            <span className="font-mono text-[12px] px-2 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)]" style={{ color }}>
              {outLabel || shape}
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (blockType === "InputSpace") {
    const dataType = (params.data_type as string) ?? "image";
    const filename = (params.custom_data_filename as string) ?? "";
    const payload = (params.custom_data_payload as string) ?? "";
    const shape = (params.input_shape as string) ?? "—";
    const isImage = dataType === "image" || dataType === "webcam";
    const hasImage = isImage && payload.startsWith("data:");
    const textPreview = !isImage && payload ? String(payload).slice(0, 400) : "";

    return (
      <div className="space-y-4">
        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] p-4 space-y-3">
          <div className="flex items-center gap-2">
            <ImageIcon size={16} style={{ color }} />
            <span className="text-[12px] font-medium text-[var(--foreground)] capitalize">{dataType}</span>
          </div>
          {hasImage && (
            <div className="rounded-lg overflow-hidden border border-[var(--border)] bg-[var(--surface)] flex items-center justify-center min-h-[140px] max-h-[220px]">
              <img src={payload} alt="Uploaded" className="max-w-full max-h-[200px] object-contain" />
            </div>
          )}
          {filename && (
            <p className="text-[11px] text-[var(--foreground-muted)] truncate" title={filename}>
              File: {filename}
            </p>
          )}
          {textPreview && (
            <pre className="text-[11px] text-[var(--foreground-muted)] whitespace-pre-wrap break-words max-h-32 overflow-y-auto p-2 rounded bg-[var(--surface)] border border-[var(--border)]">
              {textPreview}
              {String(payload).length > 400 ? "…" : ""}
            </pre>
          )}
          {!hasImage && !textPreview && !filename && (
            <p className="text-[12px] text-[var(--foreground-faint)]">Upload or capture data in the block to see a preview here.</p>
          )}
          <div className="flex flex-wrap gap-2 items-center">
            <span className="text-[11px] text-[var(--foreground-faint)]">Output shape</span>
            <span className="font-mono text-[12px] px-2 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)]" style={{ color }}>
              {outLabel || shape}
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (blockType === "Board") {
    const w = Number(params.width) ?? 28;
    const h = Number(params.height) ?? 28;
    const payload = (params.custom_data_payload as string) ?? "";
    const hasDrawing = payload.startsWith("data:");

    return (
      <div className="space-y-4">
        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] p-4 space-y-3">
          <div className="flex items-center gap-2">
            <ImageIcon size={16} style={{ color }} />
            <span className="text-[12px] font-medium text-[var(--foreground)]">Drawing output</span>
          </div>
          {hasDrawing && (
            <div className="rounded-lg overflow-hidden border border-[var(--border)] bg-[var(--surface)] flex items-center justify-center min-h-[100px] max-h-[200px]">
              <img src={payload} alt="Drawn" className="max-w-full max-h-[180px] object-contain" />
            </div>
          )}
          {!hasDrawing && (
            <p className="text-[12px] text-[var(--foreground-faint)]">Draw in the block to see a preview here.</p>
          )}
          <div className="grid grid-cols-2 gap-2 text-[12px]">
            <div>
              <span className="text-[var(--foreground-faint)]">Width</span>
              <span className="font-mono ml-2" style={{ color }}>{w}</span>
            </div>
            <div>
              <span className="text-[var(--foreground-faint)]">Height</span>
              <span className="font-mono ml-2" style={{ color }}>{h}</span>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 items-center">
            <span className="text-[11px] text-[var(--foreground-faint)]">Output shape</span>
            <span className="font-mono text-[12px] px-2 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)]" style={{ color }}>
              {outLabel || `[1, 1, ${h}, ${w}]`}
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (blockType === "TextInput") {
    const batch = params.batch_size ?? 1;
    const seqLen = params.seq_len ?? 128;
    return (
      <div className="space-y-4">
        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Type size={16} style={{ color }} />
            <span className="text-[12px] font-medium text-[var(--foreground)]">Token sequence</span>
          </div>
          <p className="text-[12px] text-[var(--foreground-muted)]">
            Shape: batch = {String(batch)}, sequence length = {String(seqLen)}. The next block (e.g. Text Embedding) will turn token IDs into vectors.
          </p>
          <div className="flex flex-wrap gap-2 items-center">
            <span className="text-[11px] text-[var(--foreground-faint)]">Output shape</span>
            <span className="font-mono text-[12px] px-2 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)]" style={{ color }}>
              {outLabel || `[${batch}, ${seqLen}]`}
            </span>
          </div>
        </div>
      </div>
    );
  }

  return null;
}

// ---------------------------------------------------------------------------
// About tab content (Display, Output)
// ---------------------------------------------------------------------------

function AboutTabContent({
  blockType,
  inLabel,
  outLabel,
  color,
}: {
  blockType: BlockType;
  inLabel: string;
  outLabel: string;
  color: string;
}) {
  const def = BLOCK_REGISTRY[blockType];
  const description = def?.description ?? "";

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Info size={16} style={{ color }} />
          <span className="text-[12px] font-medium text-[var(--foreground)]">What this block does</span>
        </div>
        <p className="text-[12px] text-[var(--foreground-muted)] leading-relaxed">
          {description}
        </p>
        <div className="flex flex-wrap gap-2 items-center pt-1">
          <span className="text-[11px] text-[var(--foreground-faint)]">Shape</span>
          <span className="font-mono text-[12px] px-2 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)]" style={{ color }}>
            {inLabel} → {outLabel}
          </span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface PeepInsideModalProps {
  blockId: string;
  blockType: BlockType;
  anchorX: number;
  anchorY: number;
  activationType?: string;
  /** Block params (for Data tab: dataset, shape, uploaded image, etc.). */
  params?: Record<string, number | string>;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function PeepInsideModalComponent({
  blockId,
  blockType,
  anchorX,
  anchorY,
  activationType,
  params = {},
  onClose,
}: PeepInsideModalProps) {
  const def = BLOCK_REGISTRY[blockType];
  const color = def?.color ?? "#6366f1";
  const Icon = def ? ICON_MAP[def.icon] : null;
  const { shapes } = useShapes();
  const result = shapes.get(blockId);
  const inLabel = getShapeLabel(result?.inputShape ?? null);
  const outLabel = getShapeLabel(result?.outputShape ?? null);

  const isDataSource = DATA_SOURCE_BLOCKS.includes(blockType);
  const isInfoOnly = INFO_ONLY_BLOCKS.includes(blockType);
  const isPassthroughOnly = PASSTHROUGH_ONLY_BLOCKS.includes(blockType);
  const isAboutOnly = isInfoOnly || isPassthroughOnly;
  const skipPeepData = isDataSource || isAboutOnly;
  const { data, loading, trained, live, refresh } = usePeepInside(skipPeepData ? null : blockId, skipPeepData ? null : blockType);
  const { gradients: gradMap } = useGradientFlow();
  const gradientInfo = gradMap.get(blockId) ?? null;

  const tabs = useMemo(() => getTabsForBlockType(blockType), [blockType]);
  const defaultTab = isDataSource ? "data" : isAboutOnly ? "about" : (tabs[0]?.id ?? "weights");
  const [activeTab, setActiveTab] = useState(defaultTab);
  const [showExplanation, setShowExplanation] = useState(true);

  useEffect(() => {
    setActiveTab(defaultTab);
  }, [blockId, blockType, defaultTab]);

  const activeTabDef = tabs.find((t) => t.id === activeTab);

  const prevWeightsRef = useRef(data?.weights ?? null);
  if (data?.weights && data.weights !== prevWeightsRef.current) {
    prevWeightsRef.current = data.weights;
  }

  // Panel position and size: fit within viewport so no page scroll is needed
  const panelStyle = useMemo(() => {
    const w = 440;
    const padding = 24;
    const maxH = typeof window !== "undefined" ? window.innerHeight - padding * 2 : 560;
    const preferredH = Math.min(520, maxH);
    let x = anchorX + 20;
    let y = anchorY - 40;
    if (typeof window !== "undefined") {
      if (x + w > window.innerWidth - padding) x = anchorX - w - 20;
      if (x < padding) x = padding;
      if (y + preferredH > window.innerHeight - padding) y = window.innerHeight - preferredH - padding;
      if (y < padding) y = padding;
    }
    return {
      left: x,
      top: y,
      width: w,
      height: preferredH,
      maxHeight: maxH,
    };
  }, [anchorX, anchorY]);

  return (
    <AnimatePresence>
      {/* Backdrop — light semi-transparent */}
      <motion.div
        key="peep-backdrop"
        className="fixed inset-0 z-[100]"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        onClick={onClose}
        style={{ backgroundColor: "rgba(0,0,0,0.15)" }}
      />

      {/* Panel */}
      <motion.div
        key="peep-panel"
        className="fixed z-[101] flex flex-col rounded-2xl overflow-hidden border border-[var(--border)] shadow-xl min-h-0"
        style={{
          ...panelStyle,
          backgroundColor: "var(--surface)",
          backdropFilter: "blur(24px)",
          boxShadow: `0 0 40px ${color}08, 0 20px 60px rgba(0,0,0,0.1)`,
        }}
        initial={{
          opacity: 0,
          scale: 0.85,
          x: anchorX - (panelStyle.left ?? 0),
          y: anchorY - (panelStyle.top ?? 0),
        }}
        animate={{ opacity: 1, scale: 1, x: 0, y: 0 }}
        exit={{ opacity: 0, scale: 0.9 }}
        transition={{ type: "spring", stiffness: 350, damping: 30, mass: 0.8 }}
      >
        {/* ══════════ Title bar ══════════ */}
        <div
          className="flex items-center gap-3 px-5 py-3.5 shrink-0"
          style={{ background: `linear-gradient(135deg, ${color}15 0%, transparent 60%)` }}
        >
          {Icon && (
            <div
              className="flex items-center justify-center w-8 h-8 rounded-lg shrink-0"
              style={{ backgroundColor: `${color}18` }}
            >
              <Icon size={16} style={{ color }} />
            </div>
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-[14px] font-bold" style={{ color }}>{def?.label ?? blockType}</span>
              <span className="text-[10px] uppercase tracking-widest text-[var(--foreground-faint)] font-mono">{def?.category}</span>
            </div>
            <div
              className="text-[12px] text-[var(--foreground-muted)] font-mono mt-0.5"
              title={getShapeLabelTooltip(result?.inputShape ?? null) || getShapeLabelTooltip(result?.outputShape ?? null) || undefined}
            >
              {inLabel} → {outLabel}
            </div>
          </div>

          {live && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[var(--success-muted)] border border-[var(--success)]/20">
              <Radio size={10} className="text-[var(--success)] animate-pulse" />
              <span className="text-[10px] text-[var(--success)] font-medium">LIVE</span>
            </div>
          )}

          {!skipPeepData && (
            <button
              onClick={refresh}
              className="p-2 rounded-lg text-[var(--foreground-muted)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)] transition-colors"
              title="Refresh data"
            >
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            </button>
          )}

          <button
            onClick={onClose}
            className="p-2 rounded-lg text-[var(--foreground-muted)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)] transition-colors"
          >
            <X size={14} />
          </button>
        </div>

        {/* ══════════ Tab bar ══════════ */}
        <div className="flex px-5 gap-1 border-b border-[var(--border)] shrink-0">
          {tabs.map((tab) => {
            const TabIcon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-1.5 px-3 py-2.5 text-[12px] font-medium
                  border-b-2 -mb-px transition-all duration-150
                  ${active
                    ? "border-current text-[var(--foreground)]"
                    : "border-transparent text-[var(--foreground-muted)] hover:text-[var(--foreground-secondary)]"
                  }
                `}
                style={active ? { color } : undefined}
              >
                <TabIcon size={13} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* ══════════ Content area ══════════ */}
        <div className="flex-1 overflow-y-auto p-5 min-h-[300px]">
          {/* "What am I looking at?" collapsible explainer */}
          {activeTabDef && (
            <button
              onClick={() => setShowExplanation((s) => !s)}
              className="w-full flex items-start gap-2 mb-4 p-3 rounded-xl bg-[var(--surface)] border border-[var(--border-muted)] text-left transition-colors hover:bg-[var(--surface-elevated)]"
            >
              <HelpCircle className="h-4 w-4 text-[var(--accent)] shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <span className="text-[12px] font-medium text-[var(--foreground-secondary)]">
                  What am I looking at?
                </span>
                {showExplanation && (
                  <p className="text-[12px] text-[var(--foreground-muted)] leading-relaxed mt-1">
                    {activeTabDef.explanation}
                  </p>
                )}
              </div>
              <span className="text-[var(--foreground-faint)] shrink-0 mt-0.5">
                {showExplanation ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </span>
            </button>
          )}

          {/* Data tab for Input / InputSpace / Board / TextInput */}
          {isDataSource && activeTab === "data" && (
            <DataTabContent blockType={blockType} params={params} outLabel={outLabel} color={color} />
          )}

          {/* About tab for Display / Output / Flatten (passthrough-only) */}
          {isAboutOnly && activeTab === "about" && (
            <AboutTabContent blockType={blockType} inLabel={inLabel} outLabel={outLabel} color={color} />
          )}

          {!skipPeepData && loading && (
            <div className="flex items-center justify-center h-48">
              <div className="flex flex-col items-center gap-3">
                <RefreshCw size={24} className="animate-spin text-[var(--foreground-muted)]" />
                <span className="text-[12px] text-[var(--foreground-muted)]">Loading block data...</span>
              </div>
            </div>
          )}

          {!skipPeepData && !loading && !data && (
            <div className="flex items-center justify-center h-48">
              <div className="flex flex-col items-center gap-3 text-center">
                <Eye size={28} className="text-[var(--foreground-faint)]" />
                <p className="text-[13px] text-[var(--foreground-muted)] font-medium">No data yet</p>
                <p className="text-[12px] text-[var(--foreground-faint)] max-w-[260px]">
                  Train the model to see what this block is learning. Click &quot;Start Training&quot; in the training panel.
                </p>
              </div>
            </div>
          )}

          {!skipPeepData && !loading && data && (
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15 }}
              >
                {/* Weights */}
                {activeTab === "weights" && (
                  <div className="space-y-4">
                    {!trained && <NotTrainedBanner color={color} />}
                    {data.weights && (
                      <WeightHeatmap
                        tensor={data.weights}
                        prevTensor={prevWeightsRef.current}
                        accentColor={color}
                        label={`Weight matrix (${data.weights.shape.join("×")})`}
                        width={398}
                        height={220}
                      />
                    )}
                    {!data.weights && <EmptyState message="This block type doesn't have trainable weights." />}
                  </div>
                )}

                {/* Activations */}
                {activeTab === "activations" && (
                  <div className="space-y-4">
                    {!trained && <NotTrainedBanner color={color} />}
                    {data.activations && (
                      <ActivationHistogram
                        tensor={data.activations}
                        accentColor={color}
                        label={`Activation distribution (${data.activations.data.length} values)`}
                        blockType={blockType}
                        activationType={activationType}
                      />
                    )}
                    {data.activations && (
                      <HeatmapViz
                        tensor={data.activations}
                        colorScheme="sequential"
                        accentColor={color}
                        label="Activation heatmap"
                        width={398}
                        height={60}
                      />
                    )}
                    {!data.activations && <EmptyState message="No activation data yet. Run a forward pass by starting training." />}
                  </div>
                )}

                {/* Gradients */}
                {activeTab === "gradients" && (
                  <div className="space-y-4">
                    {(() => {
                      const proneness = getGradientProneness(blockType, activationType);
                      const tendencyColors: Record<GradientTendency, string> = {
                        vanishing: "#ef4444",
                        exploding: "#3b82f6",
                        healthy: "#22c55e",
                        stabilizing: "#8b5cf6",
                      };
                      return proneness ? (
                        <div
                          className="rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] p-3 pl-4 border-l-4 text-[12px]"
                          style={{ borderLeftColor: tendencyColors[proneness.tendency] }}
                          role="status"
                        >
                          <span className="font-medium text-[var(--foreground)]">
                            Gradient info
                          </span>
                          <p className="text-[var(--foreground-muted)] mt-1 leading-relaxed">
                            {proneness.message}
                          </p>
                        </div>
                      ) : null;
                    })()}
                    {!trained && <NotTrainedBanner color={color} />}
                    {(data.gradients && data.gradients.length > 0) || gradientInfo ? (
                      <GradientFlowViz
                        gradientInfo={gradientInfo}
                        rawGradients={data.gradients}
                        accentColor={color}
                        label="Gradient health"
                      />
                    ) : (
                      <EmptyState message="No gradient data yet. Start training to see how gradients flow through this block." />
                    )}
                  </div>
                )}

                {/* Attention Map */}
                {activeTab === "attention" && (
                  <div className="space-y-4">
                    {!trained && <NotTrainedBanner color={color} />}
                    {data.attentionMap && (
                      <AttentionHeatmap
                        tensor={data.attentionMap}
                        accentColor={color}
                        label={`Attention weights (${data.attentionMap.shape.join("×")})`}
                        width={398}
                        height={260}
                      />
                    )}
                    {!data.attentionMap && <EmptyState message="No attention data available. Train the model first." />}
                  </div>
                )}

                {/* Filters */}
                {activeTab === "filters" && (
                  <div className="space-y-4">
                    {!trained && <NotTrainedBanner color={color} />}
                    {data.filters && (
                      <FilterGrid
                        tensor={data.filters}
                        featureMaps={data.activations}
                        accentColor={color}
                        label="Learned convolutional filters"
                      />
                    )}
                    {!data.filters && <EmptyState message="No filter data available. Train the model to see what patterns it learns." />}
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-2.5 border-t border-[var(--border)] flex items-center justify-between shrink-0">
          <span className="text-[11px] text-[var(--foreground-faint)] font-mono">{blockId}</span>
          {!skipPeepData && data && (
            <span className="text-[11px] text-[var(--foreground-faint)] font-mono">
              {new Date(data.timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function NotTrainedBanner({ color }: { color: string }) {
  return (
    <div
      className="flex items-start gap-2.5 px-4 py-3 rounded-xl border text-[12px] leading-relaxed"
      style={{
        backgroundColor: `${color}08`,
        borderColor: `${color}18`,
        color: `${color}bb`,
      }}
    >
      <Eye size={14} className="shrink-0 mt-0.5 opacity-60" />
      <span>
        <strong>Not trained yet.</strong> You&apos;re seeing initial random values.
        Train the model to see what it learns!
      </span>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex items-center justify-center py-12">
      <p className="text-[12px] text-[var(--foreground-faint)] text-center max-w-[280px] leading-relaxed">
        {message}
      </p>
    </div>
  );
}

export const PeepInsideModal = memo(PeepInsideModalComponent);
