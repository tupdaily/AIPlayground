"use client";

import { memo, useEffect, useState, useMemo } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { useReactFlow } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";
import { useShapes } from "@/neuralcanvas/components/canvas/ShapeContext";
import { usePrediction } from "@/neuralcanvas/components/canvas/PredictionContext";
import { getShapeLabel } from "@/neuralcanvas/lib/shapeEngine";
import { getClassLabelsForDataset } from "@/neuralcanvas/lib/trainingApi";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

const DISPLAY_STYLE_ID = "neural-canvas-display-block-styles";

function injectDisplayStyles() {
  if (typeof document === "undefined") return;
  if (document.getElementById(DISPLAY_STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = DISPLAY_STYLE_ID;
  style.textContent = `
    @keyframes display-tv-static {
      0%, 100% { opacity: 1; transform: translate(0, 0); }
      10% { opacity: 0.97; transform: translate(-1px, 0); }
      20% { opacity: 1; transform: translate(0, 1px); }
      30% { opacity: 0.98; transform: translate(1px, -1px); }
      40% { opacity: 1; transform: translate(0, 0); }
      50% { opacity: 0.99; transform: translate(-1px, 1px); }
      60% { opacity: 1; transform: translate(1px, 0); }
      70% { opacity: 0.97; transform: translate(0, -1px); }
      80% { opacity: 1; transform: translate(-1px, -1px); }
      90% { opacity: 0.98; transform: translate(1px, 1px); }
    }
    @keyframes display-scanline {
      0% { transform: translateY(-100%); }
      100% { transform: translateY(100%); }
    }
    @keyframes display-flicker {
      0%, 92%, 100% { opacity: 1; }
      93% { opacity: 0.85; }
      95% { opacity: 1; }
      97% { opacity: 0.9; }
    }
    @keyframes display-glitch {
      0%, 90%, 100% { filter: none; }
      91% { filter: hue-rotate(0deg); }
      92% { filter: hue-rotate(5deg); }
      93% { filter: hue-rotate(-3deg); }
      94% { filter: hue-rotate(0deg); }
    }
  `;
  document.head.appendChild(style);
}

/** No-signal TV static: SVG feTurbulence noise + B&W */
function NoSignalScreen() {
  useEffect(() => injectDisplayStyles(), []);
  return (
    <div className="absolute inset-0 overflow-hidden rounded-lg bg-black">
      <svg className="absolute inset-0 h-full w-full" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="display-static-noise" x="-20%" y="-20%" width="140%" height="140%">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.7 0.7"
              numOctaves="3"
              result="noise"
            >
              <animate
                attributeName="baseFrequency"
                values="0.6 0.6;0.85 0.85;0.7 0.75;0.75 0.7"
                dur="0.2s"
                repeatCount="indefinite"
              />
            </feTurbulence>
            <feColorMatrix in="noise" type="saturate" values="0" result="bw" />
            <feComponentTransfer in="bw" result="contrast">
              <feFuncR type="linear" slope="2.2" intercept="-0.6" />
              <feFuncG type="linear" slope="2.2" intercept="-0.6" />
              <feFuncB type="linear" slope="2.2" intercept="-0.6" />
              <feFuncA type="discrete" tableValues="1" />
            </feComponentTransfer>
          </filter>
        </defs>
        <rect
          x="0"
          y="0"
          width="100%"
          height="100%"
          fill="#0a0a0a"
          filter="url(#display-static-noise)"
          style={{ animation: "display-tv-static 0.1s steps(1) infinite" }}
        />
      </svg>
      <div
        className="absolute inset-0 flex flex-col items-center justify-center gap-0.5 text-white/90"
        style={{ textShadow: "0 0 8px rgba(255,255,255,0.3)" }}
      >
        <span className="text-[10px] font-semibold tracking-widest uppercase">No signal</span>
        <span className="text-[8px] text-white/60 tracking-wider">Connect input</span>
      </div>
    </div>
  );
}

/** LCD-style screen when input is connected. Shows human-readable label, never raw class index. */
function DisplayScreen({
  shapeLabel,
  displayLabel,
  isClassOutput,
}: {
  shapeLabel: string;
  displayLabel: string;
  isClassOutput: boolean;
}) {
  useEffect(() => injectDisplayStyles(), []);
  return (
    <div
      className="absolute inset-0 overflow-hidden rounded-lg flex flex-col items-center justify-center gap-0.5"
      style={{
        background: "linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)",
        boxShadow: "inset 0 0 20px rgba(0,0,0,0.5)",
      }}
    >
      {/* Subtle scanline overlay */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.06]"
        style={{
          background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.4) 2px, rgba(0,0,0,0.4) 4px)",
        }}
      />
      {isClassOutput ? (
        <>
          <div className="font-mono text-[9px] text-slate-500 tracking-wide uppercase">
            Label
          </div>
          <div
            className="relative font-mono text-[12px] font-bold tracking-wider text-emerald-400/95 max-w-full truncate px-2 text-center"
            style={{
              animation: "display-flicker 4s ease-in-out infinite",
              textShadow: "0 0 10px rgba(52,211,153,0.4), 0 0 2px rgba(52,211,153,0.8)",
            }}
            title={displayLabel}
          >
            {displayLabel}
          </div>
        </>
      ) : (
        <div
          className="relative font-mono text-[11px] font-bold tracking-wider text-emerald-400/95"
          style={{
            animation: "display-flicker 4s ease-in-out infinite",
            textShadow: "0 0 10px rgba(52,211,153,0.4), 0 0 2px rgba(52,211,153,0.8)",
          }}
        >
          Output
        </div>
      )}
      <div className="font-mono text-[9px] text-slate-500 tracking-wide">
        {shapeLabel}
      </div>
      <div className="absolute bottom-1 right-1.5 font-mono text-[7px] text-slate-600">
        {displayLabel ? "READY" : "—"}
      </div>
    </div>
  );
}

function DisplayBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const { shapes } = useShapes();
  const { getNodes } = useReactFlow();
  const { predictedClassIndex } = usePrediction();
  const result = shapes.get(id);
  const hasInput = result?.inputShape != null && result.inputShape.length > 0 && !result?.error;
  const shapeLabel = getShapeLabel(result?.outputShape ?? result?.inputShape ?? null);
  const inputShape = result?.inputShape;

  const datasetId = useMemo(() => {
    const nodes = getNodes();
    const inputNode = nodes.find((n) => (n.type as string) === "Input");
    const params = inputNode?.data?.params;
    if (params && typeof params === "object" && "dataset_id" in params) {
      const did = (params as Record<string, unknown>).dataset_id;
      return typeof did === "string" && did && did !== "__custom__" ? did : null;
    }
    return null;
  }, [getNodes]);

  const [classLabels, setClassLabels] = useState<string[]>([]);
  useEffect(() => {
    if (!datasetId) {
      setClassLabels([]);
      return;
    }
    let cancelled = false;
    getClassLabelsForDataset(datasetId).then((labels) => {
      if (!cancelled) setClassLabels(labels);
    });
    return () => { cancelled = true; };
  }, [datasetId]);

  const isClassOutput = inputShape != null && inputShape.length === 2 && classLabels.length > 0;
  const displayLabel = useMemo(() => {
    if (!isClassOutput || predictedClassIndex == null) return "—";
    const idx = Math.floor(Number(predictedClassIndex));
    if (idx < 0 || idx >= classLabels.length) return "—";
    return classLabels[idx];
  }, [isClassOutput, predictedClassIndex, classLabels]);

  return (
    <BaseBlock id={id} blockType="Display" params={data?.params ?? {}} selected={!!selected}>
      <div className="w-full rounded-xl overflow-hidden" style={{ marginBottom: 2 }}>
        {/* Monitor bezel */}
        <div
          className="relative w-full rounded-xl border-2 px-1.5 pt-1.5 pb-2"
          style={{
            backgroundColor: "var(--surface-elevated)",
            borderColor: "var(--border)",
            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.1), 0 2px 8px rgba(0,0,0,0.08)",
          }}
        >
          {/* Screen area */}
          <div className="relative aspect-[16/10] min-h-[72px] w-full rounded-lg overflow-hidden border border-black/40">
            {hasInput ? (
              <DisplayScreen
                shapeLabel={shapeLabel}
                displayLabel={displayLabel}
                isClassOutput={isClassOutput}
              />
            ) : (
              <NoSignalScreen />
            )}
          </div>
          {/* Small "stand" */}
          <div
            className="mx-auto mt-1 h-1 w-8 rounded-b-full"
            style={{ backgroundColor: "var(--border)", opacity: 0.8 }}
          />
        </div>
      </div>
    </BaseBlock>
  );
}

export const DisplayBlock = memo(DisplayBlockComponent);
