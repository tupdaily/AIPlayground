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

import { memo, useMemo, useRef, useState } from "react";
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

function getTabsForBlockType(blockType: BlockType): TabDef[] {
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
      explanation: "Gradients tell us how much each weight should change during training. If they're too small (vanishing), the model stops learning. If they're too large (exploding), training becomes unstable.",
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
// Props
// ---------------------------------------------------------------------------

export interface PeepInsideModalProps {
  blockId: string;
  blockType: BlockType;
  anchorX: number;
  anchorY: number;
  activationType?: string;
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
  onClose,
}: PeepInsideModalProps) {
  const def = BLOCK_REGISTRY[blockType];
  const color = def?.color ?? "#6366f1";
  const Icon = def ? ICON_MAP[def.icon] : null;
  const { shapes } = useShapes();
  const result = shapes.get(blockId);
  const inLabel = getShapeLabel(result?.inputShape ?? null);
  const outLabel = getShapeLabel(result?.outputShape ?? null);

  const { data, loading, trained, live, refresh } = usePeepInside(blockId, blockType);
  const { gradients: gradMap } = useGradientFlow();
  const gradientInfo = gradMap.get(blockId) ?? null;

  const tabs = useMemo(() => getTabsForBlockType(blockType), [blockType]);
  const [activeTab, setActiveTab] = useState(tabs[0]?.id ?? "weights");
  const [showExplanation, setShowExplanation] = useState(true);

  const activeTabDef = tabs.find((t) => t.id === activeTab);

  const prevWeightsRef = useRef(data?.weights ?? null);
  if (data?.weights && data.weights !== prevWeightsRef.current) {
    prevWeightsRef.current = data.weights;
  }

  // Panel position
  const panelStyle = useMemo(() => {
    const w = 440;
    const h = 520;
    let x = anchorX + 20;
    let y = anchorY - 40;
    if (typeof window !== "undefined") {
      if (x + w > window.innerWidth - 16) x = anchorX - w - 20;
      if (y + h > window.innerHeight - 16) y = window.innerHeight - h - 16;
      if (y < 16) y = 16;
      if (x < 16) x = 16;
    }
    return { left: x, top: y, width: w };
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
        className="fixed z-[101] flex flex-col rounded-2xl overflow-hidden border border-[var(--border)] shadow-xl"
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

          <button
            onClick={refresh}
            className="p-2 rounded-lg text-[var(--foreground-muted)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)] transition-colors"
            title="Refresh data"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>

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

          {loading && (
            <div className="flex items-center justify-center h-48">
              <div className="flex flex-col items-center gap-3">
                <RefreshCw size={24} className="animate-spin text-[var(--foreground-muted)]" />
                <span className="text-[12px] text-[var(--foreground-muted)]">Loading block data...</span>
              </div>
            </div>
          )}

          {!loading && !data && (
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

          {!loading && data && (
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
          {data && (
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
