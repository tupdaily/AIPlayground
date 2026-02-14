"use client";

import { useGraphStore } from "@/store/graphStore";
import { useTrainingStore } from "@/store/trainingStore";
import { serializeGraph } from "@/lib/serialization";
import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Toolbar() {
  const { nodes, edges, clearGraph } = useGraphStore();
  const { status } = useTrainingStore();
  const [validating, setValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<{
    valid: boolean;
    message: string;
    total_params?: number;
  } | null>(null);

  const handleValidate = async () => {
    setValidating(true);
    setValidationResult(null);
    try {
      const graph = serializeGraph(nodes, edges);
      const res = await fetch(`${API_BASE}/api/graphs/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(graph),
      });
      const result = await res.json();
      setValidationResult(result);

      // Update shapes on nodes
      if (result.valid && result.shapes) {
        const { updateNodeShape } = useGraphStore.getState();
        for (const [nodeId, shape] of Object.entries(result.shapes)) {
          updateNodeShape(nodeId, shape as number[]);
        }
      }
    } catch {
      setValidationResult({
        valid: false,
        message: "Cannot connect to backend server",
      });
    } finally {
      setValidating(false);
    }
  };

  const handleExportJSON = () => {
    const graph = serializeGraph(nodes, edges);
    const blob = new Blob([JSON.stringify(graph, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "model-graph.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-12 bg-white border-b border-gray-200 flex items-center px-4 gap-3 flex-shrink-0">
      <h1 className="font-bold text-gray-800 mr-4">AIPlayground</h1>

      <button
        onClick={handleValidate}
        disabled={validating || nodes.length === 0}
        className="px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {validating ? "Validating..." : "Validate"}
      </button>

      <button
        onClick={handleExportJSON}
        disabled={nodes.length === 0}
        className="px-3 py-1 bg-gray-200 text-gray-700 text-sm rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Export JSON
      </button>

      <button
        onClick={clearGraph}
        disabled={nodes.length === 0}
        className="px-3 py-1 bg-gray-200 text-gray-700 text-sm rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Clear
      </button>

      <div className="flex-1" />

      {validationResult && (
        <div
          className={`text-xs px-2 py-1 rounded ${
            validationResult.valid
              ? "bg-green-100 text-green-700"
              : "bg-red-100 text-red-700"
          }`}
        >
          {validationResult.valid
            ? `Valid - ${validationResult.total_params?.toLocaleString()} params`
            : validationResult.message}
        </div>
      )}

      <div className="text-xs text-gray-400">
        {nodes.length} blocks, {edges.length} connections
      </div>

      {status === "running" && (
        <div className="text-xs text-blue-600 font-medium animate-pulse">
          Training...
        </div>
      )}
    </div>
  );
}
