"use client";

import { useGraphStore } from "@/store/graphStore";
import { getBlockDef } from "@/lib/blockRegistry";

export default function PropertiesPanel() {
  const { nodes, selectedNodeId, updateNodeParams } = useGraphStore();
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  if (!selectedNode) {
    return (
      <div className="w-64 bg-gray-50 border-l border-gray-200 p-4 flex-shrink-0">
        <p className="text-sm text-gray-400 text-center mt-8">
          Select a block to configure its properties
        </p>
      </div>
    );
  }

  const def = getBlockDef(selectedNode.data.blockType);
  if (!def) return null;

  const params = selectedNode.data.params;

  const handleChange = (key: string, value: unknown) => {
    updateNodeParams(selectedNode.id, { [key]: value });
  };

  return (
    <div className="w-64 bg-gray-50 border-l border-gray-200 overflow-y-auto flex-shrink-0">
      <div className="p-4">
        <div className="flex items-center gap-2 mb-4">
          <div
            className="w-3 h-3 rounded-sm"
            style={{ backgroundColor: def.color }}
          />
          <h2 className="font-bold text-sm text-gray-700">{def.label}</h2>
        </div>

        <div className="text-xs text-gray-400 mb-4">ID: {selectedNode.id}</div>

        {def.parameters.length === 0 && (
          <p className="text-sm text-gray-400">No configurable parameters</p>
        )}

        <div className="space-y-3">
          {def.parameters.map((param) => (
            <div key={param.key}>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                {param.label}
              </label>

              {param.type === "number" && (
                <input
                  type="number"
                  value={Number(params[param.key] ?? param.default)}
                  min={param.min}
                  max={param.max}
                  step={param.step ?? 1}
                  onChange={(e) =>
                    handleChange(param.key, parseFloat(e.target.value) || 0)
                  }
                  className="w-full px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              )}

              {param.type === "select" && (
                <select
                  value={String(params[param.key] ?? param.default)}
                  onChange={(e) => handleChange(param.key, e.target.value)}
                  className="w-full px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                  {param.options?.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              )}

              {param.type === "boolean" && (
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={Boolean(params[param.key] ?? param.default)}
                    onChange={(e) => handleChange(param.key, e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm text-gray-600">Enabled</span>
                </label>
              )}

              {param.type === "tuple" && (
                <input
                  type="text"
                  value={
                    Array.isArray(params[param.key])
                      ? (params[param.key] as number[]).join(", ")
                      : String(param.default)
                  }
                  onChange={(e) => {
                    const vals = e.target.value
                      .split(",")
                      .map((v) => parseInt(v.trim(), 10))
                      .filter((v) => !isNaN(v));
                    handleChange(param.key, vals);
                  }}
                  placeholder="e.g. 1, 28, 28"
                  className="w-full px-2 py-1 border border-gray-300 rounded text-sm font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              )}
            </div>
          ))}
        </div>

        {selectedNode.data.inferredShape && (
          <div className="mt-4 pt-3 border-t">
            <label className="block text-xs font-medium text-gray-400 mb-1">
              Inferred Output Shape
            </label>
            <div className="text-sm font-mono text-gray-600">
              [{selectedNode.data.inferredShape.join(", ")}]
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
