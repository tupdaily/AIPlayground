"use client";

/**
 * AugmentPreviewModal — configure image augmentations and see live preview.
 * Used when clicking the eye on the Augment block.
 * Uses a sample image from the Input block's selected dataset when available.
 */

import { useCallback, useMemo, useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { X, RotateCw, FlipHorizontal, FlipVertical, Sun, Contrast, Palette, ScanEye, Sparkles } from "lucide-react";
import { getApiBase } from "@/neuralcanvas/lib/trainingApi";

/** Draws Gaussian-like noise on a canvas overlay so "Add noise" is clearly visible in the preview. */
function NoiseOverlay({ amount }: { amount: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const opacity = useMemo(() => Math.min(1, Math.max(0.15, amount * 3)), [amount]);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = 64;
    const h = 64;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imageData = ctx.createImageData(w * dpr, h * dpr);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const g = Math.floor(128 + 80 * (Math.random() - 0.5));
      data[i] = data[i + 1] = data[i + 2] = g;
      data[i + 3] = Math.floor(255 * opacity);
    }
    ctx.putImageData(imageData, 0, 0);
  }, [opacity]);
  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none rounded-lg mix-blend-overlay"
      style={{ imageRendering: "pixelated" }}
      width={64}
      height={64}
      aria-hidden
    />
  );
}

export interface AugmentSpec {
  id: string;
  enabled: boolean;
  /** e.g. { degrees: 15 }, { factor: 0.2 } */
  params: Record<string, number>;
}

const AVAILABLE_AUGMENTATIONS: {
  id: string;
  label: string;
  icon: "rotate" | "hflip" | "vflip" | "brightness" | "contrast" | "saturation" | "noise" | "blur";
  paramSchema: { name: string; min: number; max: number; step: number; default: number; label: string }[];
}[] = [
  {
    id: "rotation",
    label: "Random rotation",
    icon: "rotate",
    paramSchema: [
      { name: "degrees", min: 0, max: 45, step: 5, default: 15, label: "Degrees" },
    ],
  },
  {
    id: "hflip",
    label: "Horizontal flip",
    icon: "hflip",
    paramSchema: [],
  },
  {
    id: "vflip",
    label: "Vertical flip",
    icon: "vflip",
    paramSchema: [],
  },
  {
    id: "brightness",
    label: "Brightness",
    icon: "brightness",
    paramSchema: [
      { name: "factor", min: 0, max: 1, step: 0.1, default: 0.2, label: "Strength" },
    ],
  },
  {
    id: "contrast",
    label: "Contrast",
    icon: "contrast",
    paramSchema: [
      { name: "factor", min: 0, max: 1, step: 0.1, default: 0.2, label: "Strength" },
    ],
  },
  {
    id: "saturation",
    label: "Saturation",
    icon: "saturation",
    paramSchema: [
      { name: "factor", min: 0, max: 1, step: 0.1, default: 0.3, label: "Strength" },
    ],
  },
  {
    id: "noise",
    label: "Add noise",
    icon: "noise",
    paramSchema: [
      { name: "amount", min: 0.01, max: 0.3, step: 0.01, default: 0.08, label: "Amount" },
    ],
  },
  {
    id: "blur",
    label: "Gaussian blur",
    icon: "blur",
    paramSchema: [
      { name: "radius", min: 0.5, max: 3, step: 0.5, default: 1, label: "Radius" },
    ],
  },
];

function parseAugmentations(json: string): AugmentSpec[] {
  try {
    const raw = JSON.parse(json || "[]");
    if (!Array.isArray(raw)) return [];
    return raw.map((item: unknown) => {
      if (item && typeof item === "object" && "id" in item) {
        const o = item as { id: string; enabled?: boolean; params?: Record<string, number> };
        const def = AVAILABLE_AUGMENTATIONS.find((a) => a.id === o.id);
        const params: Record<string, number> = {};
        if (def) {
          def.paramSchema.forEach((p) => {
            params[p.name] = (o.params && typeof o.params[p.name] === "number")
              ? (o.params[p.name] as number)
              : p.default;
          });
        }
        return {
          id: o.id,
          enabled: o.enabled !== false,
          params: Object.keys(params).length ? params : (o.params ?? {}),
        };
      }
      return null;
    }).filter(Boolean) as AugmentSpec[];
  } catch {
    return [];
  }
}

function ensureAllAugmentations(specs: AugmentSpec[]): AugmentSpec[] {
  const byId = new Map(specs.map((s) => [s.id, s]));
  return AVAILABLE_AUGMENTATIONS.map((def) => {
    const existing = byId.get(def.id);
    if (existing) return existing;
    const params: Record<string, number> = {};
    def.paramSchema.forEach((p) => { params[p.name] = p.default; });
    return { id: def.id, enabled: false, params };
  });
}

// Fallback when dataset has no sample (e.g. custom)
const FALLBACK_SAMPLE_URL =
  "data:image/svg+xml," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><rect width="28" height="28" fill="#f0f0f0"/><path fill="#333" d="M8 4h3v8l8 12h-3l-6-9-2 3v6H5V4z"/></svg>`
  );

interface AugmentPreviewModalProps {
  open: boolean;
  onClose: () => void;
  blockId: string;
  initialAugmentations: string;
  /** Dataset selected in Input block (mnist, fashion_mnist, cifar10); used to show a real sample. */
  datasetId: string | null;
  onSave: (augmentationsJson: string) => void;
  color: string;
  anchorX: number;
  anchorY: number;
}

export function AugmentPreviewModal({
  open,
  onClose,
  blockId,
  initialAugmentations,
  datasetId,
  onSave,
  color,
  anchorX,
  anchorY,
}: AugmentPreviewModalProps) {
  const [specs, setSpecs] = useState<AugmentSpec[]>(() =>
    ensureAllAugmentations(parseAugmentations(initialAugmentations))
  );
  const [sampleImageUrl, setSampleImageUrl] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setSpecs(ensureAllAugmentations(parseAugmentations(initialAugmentations)));
    }
  }, [open, initialAugmentations]);

  // Fetch one sample image from the selected dataset for preview
  useEffect(() => {
    if (!open || !datasetId || datasetId === "__custom__") {
      setSampleImageUrl(null);
      return;
    }
    let revoked = false;
    let blobUrl: string | null = null;
    const base = getApiBase();
    fetch(`${base}/api/datasets/${encodeURIComponent(datasetId)}/sample?t=${Date.now()}`)
      .then((r) => {
        if (!r.ok) throw new Error("No sample");
        return r.blob();
      })
      .then((blob) => {
        if (revoked) return;
        blobUrl = URL.createObjectURL(blob);
        setSampleImageUrl(blobUrl);
      })
      .catch(() => {
        if (!revoked) setSampleImageUrl(null);
      });
    return () => {
      revoked = true;
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [open, datasetId]);

  const toggle = useCallback((id: string) => {
    setSpecs((prev) =>
      prev.map((s) => (s.id === id ? { ...s, enabled: !s.enabled } : s))
    );
  }, []);

  const setParam = useCallback((id: string, paramName: string, value: number) => {
    setSpecs((prev) =>
      prev.map((s) =>
        s.id === id ? { ...s, params: { ...s.params, [paramName]: value } } : s
      )
    );
  }, []);

  const handleSave = useCallback(() => {
    const enabledSpecs = specs.filter((s) => s.enabled);
    onSave(JSON.stringify(enabledSpecs));
    onClose();
  }, [specs, onSave, onClose]);

  const previewStyle = useMemo(() => {
    let transform = "";
    let filter = "";
    specs.forEach((s) => {
      if (!s.enabled) return;
      switch (s.id) {
        case "rotation":
          transform += ` rotate(${s.params.degrees ?? 15}deg)`;
          break;
        case "hflip":
          transform += " scaleX(-1)";
          break;
        case "vflip":
          transform += " scaleY(-1)";
          break;
        case "brightness": {
          const f = 1 + (s.params.factor ?? 0.2);
          filter += ` brightness(${f})`;
          break;
        }
        case "contrast": {
          const f = 1 + (s.params.factor ?? 0.2);
          filter += ` contrast(${f})`;
          break;
        }
        case "saturation": {
          const f = 1 + (s.params.factor ?? 0.3);
          filter += ` saturate(${f})`;
          break;
        }
        case "blur": {
          const r = s.params.radius ?? 1;
          filter += ` blur(${r}px)`;
          break;
        }
      }
    });
    return { transform: transform || "none", filter: filter || "none" };
  }, [specs]);

  if (!open) return null;

  const content = (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Augment preview"
    >
      <div
        className="
          rounded-2xl border border-[var(--border)] bg-[var(--surface)]
          shadow-xl max-w-lg w-full max-h-[90vh] overflow-hidden
          flex flex-col
        "
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]"
          style={{ borderLeftWidth: 4, borderLeftColor: color }}
        >
          <h2 className="text-base font-bold text-[var(--foreground)] flex items-center gap-2">
            <span style={{ color }}>Augment</span>
            <span className="text-[var(--foreground-muted)] font-normal text-sm">— Preview &amp; configure</span>
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg text-[var(--foreground-muted)] hover:text-[var(--foreground)] hover:bg-[var(--surface-elevated)] transition-colors"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex flex-col flex-1 min-h-0">
          {/* Preview area — fixed height, not scrollable */}
          <div className="shrink-0 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)] p-4 mx-4 mt-4">
            <p className="text-xs font-medium text-[var(--foreground-muted)] uppercase tracking-wider mb-3">
              Live preview
            </p>
            <div className="flex items-center justify-center gap-8">
              <div className="text-center">
                <p className="text-[11px] text-[var(--foreground-muted)] mb-2">Original</p>
                <div className="w-[88px] h-[88px] flex items-center justify-center">
                  <img
                    src={sampleImageUrl ?? FALLBACK_SAMPLE_URL}
                    alt="Sample original"
                    className="w-20 h-20 object-contain border border-[var(--border)] rounded-lg bg-white shrink-0"
                  />
                </div>
              </div>
              <div className="text-center">
                <p className="text-[11px] text-[var(--foreground-muted)] mb-2">Augmented</p>
                <div className="w-[100px] h-[100px] min-h-[100px] flex items-center justify-center overflow-hidden border border-[var(--border)] rounded-lg bg-white relative">
                  <img
                    src={sampleImageUrl ?? FALLBACK_SAMPLE_URL}
                    alt="Sample augmented"
                    className="w-16 h-16 object-contain transition-all duration-150 shrink-0"
                    style={previewStyle}
                  />
                  {specs.some((s) => s.enabled && s.id === "noise") && (
                    <NoiseOverlay amount={specs.find((s) => s.id === "noise")?.params?.amount ?? 0.08} />
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Augmentations list — only this section scrolls */}
          <div className="flex-1 min-h-0 overflow-y-auto px-4 py-4">
            <p className="text-xs font-medium text-[var(--foreground-muted)] uppercase tracking-wider mb-3">
              Add augmentations
            </p>
            <ul className="space-y-2">
              {AVAILABLE_AUGMENTATIONS.map((def) => {
                const spec = specs.find((s) => s.id === def.id)!;
                const Icon =
                  def.icon === "rotate"
                    ? RotateCw
                    : def.icon === "hflip"
                      ? FlipHorizontal
                      : def.icon === "vflip"
                        ? FlipVertical
                        : def.icon === "brightness"
                          ? Sun
                          : def.icon === "contrast"
                            ? Contrast
                            : def.icon === "saturation"
                              ? Palette
                              : def.icon === "noise"
                                ? Sparkles
                                : ScanEye;
                return (
                  <li
                    key={def.id}
                    className="flex items-center gap-3 p-3 rounded-xl border border-[var(--border)] bg-[var(--surface-elevated)]"
                  >
                    <button
                      type="button"
                      onClick={() => toggle(def.id)}
                      className={`
                        flex items-center justify-center w-9 h-9 rounded-lg shrink-0 transition-colors
                        ${spec.enabled ? "opacity-100" : "opacity-50"}
                      `}
                      style={{
                        backgroundColor: spec.enabled ? `${color}20` : "var(--surface)",
                        color: spec.enabled ? color : "var(--foreground-muted)",
                      }}
                      title={spec.enabled ? "Disable" : "Enable"}
                    >
                      <Icon size={18} />
                    </button>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-[var(--foreground)]">{def.label}</p>
                      {def.paramSchema.length > 0 && spec.enabled && (
                        <div className="mt-2 flex items-center gap-2">
                          {def.paramSchema.map((p) => (
                            <label key={p.name} className="flex items-center gap-2 text-xs">
                              <span className="text-[var(--foreground-muted)]">{p.label}</span>
                              <input
                                type="range"
                                min={p.min}
                                max={p.max}
                                step={p.step}
                                value={spec.params[p.name] ?? p.default}
                                onChange={(e) =>
                                  setParam(def.id, p.name, parseFloat(e.target.value))
                                }
                                className="w-24 h-1.5 rounded-full accent-[var(--accent)]"
                              />
                              <span className="font-mono text-[var(--foreground-muted)] w-8">
                                {spec.params[p.name] ?? p.default}
                              </span>
                            </label>
                          ))}
                        </div>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 px-4 py-3 border-t border-[var(--border)]">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-sm font-medium text-[var(--foreground-secondary)] hover:bg-[var(--surface-elevated)] transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            className="px-4 py-2 rounded-lg text-sm font-medium text-white transition-colors"
            style={{ backgroundColor: color }}
          >
            Save augmentations
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(content, document.body);
}
