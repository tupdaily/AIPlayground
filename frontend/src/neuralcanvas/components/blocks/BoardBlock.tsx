"use client";

import { memo, useCallback, useRef, useEffect, useState } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import { useReactFlow } from "@xyflow/react";
import { BaseBlock } from "./BaseBlock";
import { useShapes } from "@/neuralcanvas/components/canvas/ShapeContext";
import { getShapeLabel } from "@/neuralcanvas/lib/shapeEngine";
import { Eraser } from "lucide-react";

interface BlockData extends Record<string, unknown> {
  params: Record<string, number | string>;
}

const CANVAS_SIZE = 140;
const STROKE = 8;
const BG = "#1a1a1a";

function BoardBlockComponent({ id, data, selected }: NodeProps<Node<BlockData>>) {
  const params = data?.params ?? {};
  const { shapes } = useShapes();
  const { setNodes } = useReactFlow();
  const result = shapes.get(id);
  const outLabel = getShapeLabel(result?.outputShape ?? null);

  const width = Math.max(8, Math.min(224, Number(params.width) || 28));
  const height = Math.max(8, Math.min(224, Number(params.height) || 28));
  const storedData = (params.custom_data_payload as string) || null;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);
  const [hasDrawn, setHasDrawn] = useState(!!storedData);

  const captureAndStore = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const off = document.createElement("canvas");
    off.width = width;
    off.height = height;
    const offCtx = off.getContext("2d");
    if (!offCtx) return;
    offCtx.fillStyle = "#000";
    offCtx.fillRect(0, 0, width, height);
    offCtx.drawImage(canvas, 0, 0, CANVAS_SIZE, CANVAS_SIZE, 0, 0, width, height);
    const dataUrl = off.toDataURL("image/png");
    const shapeStr = `1,${height},${width}`;

    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        const prev = (n.data?.params && typeof n.data.params === "object")
          ? (n.data.params as Record<string, number | string>)
          : {};
        return {
          ...n,
          data: {
            ...n.data,
            params: {
              ...prev,
              input_shape: shapeStr,
              custom_data_payload: dataUrl,
            },
          },
        };
      }),
    );
  }, [id, width, height, setNodes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = STROKE;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (storedData && storedData.startsWith("data:")) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, width, height, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
      };
      img.src = storedData;
    }
  }, [storedData, width, height]);

  const getPos = useCallback((e: React.PointerEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }, []);

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      const pos = getPos(e);
      if (!pos) return;
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      isDrawingRef.current = true;
      setHasDrawn(true);
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y);
    },
    [getPos],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!isDrawingRef.current) return;
      const pos = getPos(e);
      if (!pos) return;
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
    },
    [getPos],
  );

  const onPointerUp = useCallback(() => {
    if (isDrawingRef.current) {
      isDrawingRef.current = false;
      captureAndStore();
    }
  }, [captureAndStore]);

  const onPointerLeave = useCallback(() => {
    if (isDrawingRef.current) {
      isDrawingRef.current = false;
      captureAndStore();
    }
  }, [captureAndStore]);

  const onClear = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setHasDrawn(false);
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        const prev = (n.data?.params && typeof n.data.params === "object")
          ? (n.data.params as Record<string, number | string>)
          : {};
        const { custom_data_payload: _, input_shape: __, ...rest } = prev;
        return {
          ...n,
          data: {
            ...n.data,
            params: { ...rest, width: prev.width ?? 28, height: prev.height ?? 28 },
          },
        };
      }),
    );
  }, [id, setNodes]);

  return (
    <BaseBlock id={id} blockType="Board" params={params} selected={!!selected}>
      <div className="space-y-2">
        <div className="relative rounded-xl overflow-hidden border border-[var(--border)] bg-[var(--surface-elevated)] flex items-center justify-center">
          <canvas
            ref={canvasRef}
            className="nodrag nopan w-full max-w-[140px] h-[140px] touch-none cursor-crosshair block"
            style={{ maxHeight: 140 }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerLeave}
            onPointerCancel={onPointerUp}
          />
        </div>
        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={onClear}
            className="nodrag nopan flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium bg-[var(--surface-elevated)] border border-[var(--border)] text-[var(--foreground-muted)] hover:text-[var(--foreground)] hover:border-[var(--border-strong)] transition-colors"
          >
            <Eraser size={12} />
            Clear
          </button>
          <span className="text-[10px] font-mono text-[var(--foreground-muted)]" title="Output shape">
            {outLabel || `${height}×${width}`}
          </span>
        </div>
      </div>
    </BaseBlock>
  );
}

export const BoardBlock = memo(BoardBlockComponent);
