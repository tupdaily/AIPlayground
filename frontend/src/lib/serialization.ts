import { Node, Edge } from "@xyflow/react";
import { GraphSchema } from "@/types/graph";
import { NodeData } from "@/store/graphStore";

export function serializeGraph(
  nodes: Node<NodeData>[],
  edges: Edge[]
): GraphSchema {
  return {
    version: "1.0",
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type!,
      params: n.data.params,
      position: n.position,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      sourceHandle: e.sourceHandle || "out",
      target: e.target,
      targetHandle: e.targetHandle || "in",
    })),
    metadata: {
      name: "Untitled Model",
      created_at: new Date().toISOString(),
    },
  };
}
