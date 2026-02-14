"use client";

import { Handle, Position, NodeProps } from "@xyflow/react";
import { getBlockDef } from "@/lib/blockRegistry";
import { NodeData } from "@/store/graphStore";

export default function MLNode({ data, selected }: NodeProps) {
  const nodeData = data as unknown as NodeData;
  const def = getBlockDef(nodeData.blockType);
  if (!def) return null;

  const borderColor = selected ? "#2563eb" : def.color;
  const paramEntries = Object.entries(nodeData.params).filter(
    ([key]) => key !== "shape"
  );

  return (
    <div
      className="rounded-lg shadow-md bg-white min-w-[140px] border-2 text-xs"
      style={{ borderColor }}
    >
      {/* Target handles */}
      {def.inputs.map((inp, i) => (
        <Handle
          key={inp.id}
          type="target"
          position={Position.Top}
          id={inp.id}
          style={{
            left: def.inputs.length === 1 ? "50%" : `${((i + 1) / (def.inputs.length + 1)) * 100}%`,
            background: def.color,
            width: 10,
            height: 10,
          }}
          title={inp.label}
        />
      ))}

      {/* Header */}
      <div
        className="px-3 py-1.5 font-semibold text-white rounded-t-md text-center"
        style={{ backgroundColor: def.color }}
      >
        {def.label}
      </div>

      {/* Parameters summary */}
      {(paramEntries.length > 0 || nodeData.inferredShape) && (
        <div className="px-3 py-1.5 space-y-0.5">
          {paramEntries.map(([key, value]) => (
            <div key={key} className="flex justify-between gap-2 text-gray-600">
              <span className="truncate">{key}:</span>
              <span className="font-mono text-gray-900">
                {Array.isArray(value) ? `[${value.join(",")}]` : String(value)}
              </span>
            </div>
          ))}
          {nodeData.inferredShape && (
            <div className="flex justify-between gap-2 text-gray-400 border-t pt-0.5 mt-0.5">
              <span>shape:</span>
              <span className="font-mono">
                [{nodeData.inferredShape.join(", ")}]
              </span>
            </div>
          )}
        </div>
      )}

      {/* Source handles */}
      {def.outputs.map((out, i) => (
        <Handle
          key={out.id}
          type="source"
          position={Position.Bottom}
          id={out.id}
          style={{
            left: def.outputs.length === 1 ? "50%" : `${((i + 1) / (def.outputs.length + 1)) * 100}%`,
            background: def.color,
            width: 10,
            height: 10,
          }}
          title={out.label}
        />
      ))}
    </div>
  );
}
