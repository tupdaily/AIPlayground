"use client";

import { memo, useCallback, useEffect, useRef, useState } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { useReactFlow } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";
import { useShapes } from "@/neuralcanvas/components/canvas/ShapeContext";
import { getShapeLabel } from "@/neuralcanvas/lib/shapeEngine";
import { Upload, Camera, ImageIcon } from "lucide-react";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

type DataType = "image" | "table" | "text" | "webcam";

/** Input Space viz: upload/capture concept */
function InputSpaceViz({ dataType }: { dataType: DataType }) {
  const color = "#F59E0B";
  const w = 160;
  const h = 36;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <rect x={20} y={6} width={48} height={24} rx={6} fill={`${color}20`} stroke={color} strokeWidth="1" />
      <text x={44} y={22} fontSize="8" fill={color} fontWeight="600" textAnchor="middle">
        {dataType === "image" && "IMG"}
        {dataType === "table" && "CSV"}
        {dataType === "text" && "TXT"}
        {dataType === "webcam" && "CAM"}
      </text>
      <line x1={72} y1={18} x2={140} y2={18} stroke={color} strokeWidth="1.5" opacity={0.7} />
      <polygon points="136,14 144,18 136,22" fill={color} opacity={0.9} />
      <text x={88} y={12} fontSize="7" fill={color} opacity={0.9}>
        Custom data → Input
      </text>
    </svg>
  );
}

function InputSpaceBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const params = data?.params ?? {};
  const dataType = (params.data_type as DataType) ?? "image";
  const { shapes } = useShapes();
  const { setNodes } = useReactFlow();
  const result = shapes.get(id);
  const outLabel = getShapeLabel(result?.outputShape ?? null);

  const [webcamActive, setWebcamActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const uploadLabel = (params.custom_data_filename as string) || "";
  const storedPreview = (params.custom_data_payload as string) || null;
  const hasStoredImage = dataType === "image" || dataType === "webcam" ? !!storedPreview?.startsWith("data:") : !!storedPreview;

  const onDataTypeChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value as DataType;
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const prev = (n.data?.params && typeof n.data.params === "object") ? (n.data.params as Record<string, number | string>) : {};
          const shape = value === "image" || value === "webcam" ? "1,28,28" : value === "table" ? "1,100" : "1,128";
          return {
            ...n,
            data: {
              ...n.data,
              params: {
                ...prev,
                data_type: value,
                input_shape: shape,
                custom_data_payload: "",
                custom_data_filename: "",
              },
            },
          };
        }),
      );
    },
    [id, setNodes],
  );

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>, type: DataType) => {
      const file = e.target.files?.[0];
      if (!file) return;

      if (type === "image") {
        const reader = new FileReader();
        reader.onload = () => {
          const dataUrl = reader.result as string;
          const img = new Image();
          img.onload = () => {
            const c = Math.min(3, img.width * img.height > 224 * 224 ? 3 : 1);
            const h = Math.min(img.height, 224);
            const w = Math.min(img.width, 224);
            const shape = `${c},${h},${w}`;
            setNodes((nds) =>
              nds.map((n) => {
                if (n.id !== id) return n;
                const prev = (n.data?.params && typeof n.data.params === "object") ? (n.data.params as Record<string, number | string>) : {};
                return {
                  ...n,
                  data: {
                    ...n.data,
                    params: {
                      ...prev,
                      input_shape: shape,
                      custom_data_payload: dataUrl,
                      custom_data_filename: file.name,
                    },
                  },
                };
              }),
            );
          };
          img.src = dataUrl;
        };
        reader.readAsDataURL(file);
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          const text = (reader.result as string) ?? "";
          const shape = type === "table" ? "1,100" : "1,128";
          setNodes((nds) =>
            nds.map((n) => {
              if (n.id !== id) return n;
              const prev = (n.data?.params && typeof n.data.params === "object") ? (n.data.params as Record<string, number | string>) : {};
              return {
                ...n,
                data: {
                  ...n.data,
                  params: {
                    ...prev,
                    input_shape: shape,
                    custom_data_payload: text,
                    custom_data_filename: file.name,
                  },
                },
              };
            }),
          );
        };
        reader.readAsText(file);
      }
      e.target.value = "";
    },
    [id, setNodes],
  );

  const startWebcam = useCallback(() => {
    if (webcamActive) return;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 224, height: 224 } })
      .then((stream) => {
        streamRef.current = stream;
        setWebcamActive(true);
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(() => setWebcamActive(false));
  }, [webcamActive]);

  const stopWebcam = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setWebcamActive(false);
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !webcamActive) return;
    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0);
    const dataUrl = canvas.toDataURL("image/png");
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        const prev = (n.data?.params && typeof n.data.params === "object") ? (n.data.params as Record<string, number | string>) : {};
        return {
          ...n,
          data: {
            ...n.data,
            params: {
              ...prev,
              input_shape: "3,224,224",
              custom_data_payload: dataUrl,
              custom_data_filename: "webcam-capture.png",
            },
          },
        };
      }),
    );
  }, [id, setNodes, webcamActive]);

  useEffect(() => () => { stopWebcam(); }, [stopWebcam]);

  return (
    <BaseBlock id={id} blockType="InputSpace" params={params} selected={!!selected}>
      {/* When image is uploaded, show small thumbnail in block; otherwise show type viz */}
      {hasStoredImage && (dataType === "image" || dataType === "webcam") && storedPreview ? (
        <div className="rounded-xl overflow-hidden border border-[var(--border)] bg-[var(--surface-elevated)] flex items-center justify-center aspect-[16/10] max-h-20">
          <img src={storedPreview} alt="Uploaded" className="w-full h-full object-contain" />
        </div>
      ) : (
        <InputSpaceViz dataType={dataType} />
      )}
      <div className="space-y-2 pt-0.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] text-[var(--foreground-muted)] font-medium shrink-0">
            Data type
          </span>
          <span className="text-[10px] font-mono text-[var(--foreground-muted)] truncate min-w-0" title="Output shape">
            {outLabel}
          </span>
        </div>
        <select
          value={dataType}
          onChange={onDataTypeChange}
          className="nodrag nopan w-full px-2.5 py-1.5 rounded-lg text-[12px] font-medium bg-[var(--surface-elevated)] border border-[var(--border)] text-[var(--foreground)] outline-none focus:border-[var(--block-input)] focus:ring-1 focus:ring-[var(--block-input)]/20 cursor-pointer"
        >
          <option value="image">Image</option>
          <option value="table">Table (CSV)</option>
          <option value="text">Text</option>
          <option value="webcam">Webcam</option>
        </select>

        {(dataType === "image" || dataType === "table" || dataType === "text") && (
          <label className="nodrag nopan flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-[var(--surface-elevated)] border border-[var(--border)] cursor-pointer hover:border-[var(--block-input)]/50 transition-colors">
            <Upload size={14} className="text-[var(--foreground-muted)] shrink-0" />
            <span className="text-[11px] text-[var(--foreground-secondary)] truncate flex-1">
              {uploadLabel || (dataType === "image" ? "Upload image" : dataType === "table" ? "Upload CSV" : "Upload text")}
            </span>
            <input
              type="file"
              accept={dataType === "image" ? "image/*" : dataType === "table" ? ".csv" : "text/*,.txt"}
              onChange={(e) => onFileChange(e, dataType)}
              className="hidden"
            />
          </label>
        )}

        {dataType === "webcam" && (
          <div className="space-y-1.5 nodrag nopan">
            <div className="relative rounded-lg overflow-hidden bg-[var(--surface-elevated)] border border-[var(--border)] aspect-video max-h-24">
              {hasStoredImage ? (
                <img src={storedPreview!} alt="Captured" className="w-full h-full object-cover" />
              ) : (
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                  style={{ display: webcamActive ? "block" : "none" }}
                />
              )}
              {!webcamActive && !hasStoredImage && (
                <div className="absolute inset-0 flex items-center justify-center text-[var(--foreground-muted)] text-[10px]">
                  Camera off
                </div>
              )}
            </div>
            <div className="flex gap-1.5">
              {!webcamActive ? (
                <button
                  type="button"
                  onClick={startWebcam}
                  className="flex-1 px-2 py-1.5 rounded-lg text-[11px] font-medium bg-[var(--block-input)]/15 text-[var(--block-input)] border border-[var(--block-input)]/30 hover:bg-[var(--block-input)]/25 transition-colors flex items-center justify-center gap-1"
                >
                  <Camera size={12} /> Start camera
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={captureFrame}
                    className="flex-1 px-2 py-1.5 rounded-lg text-[11px] font-medium bg-[var(--block-input)]/15 text-[var(--block-input)] border border-[var(--block-input)]/30 hover:bg-[var(--block-input)]/25 transition-colors flex items-center justify-center gap-1"
                  >
                    <ImageIcon size={12} /> Capture
                  </button>
                  <button
                    type="button"
                    onClick={stopWebcam}
                    className="px-2 py-1.5 rounded-lg text-[11px] font-medium bg-[var(--surface-elevated)] border border-[var(--border)] text-[var(--foreground-muted)] hover:bg-[var(--danger-muted)] hover:text-[var(--danger)] hover:border-[var(--danger)] transition-colors"
                  >
                    Stop
                  </button>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </BaseBlock>
  );
}

export const InputSpaceBlock = memo(InputSpaceBlockComponent);
