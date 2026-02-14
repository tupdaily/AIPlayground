import { Node, Edge } from "@xyflow/react";
import { GraphSchema } from "@/types/graph";
import { NodeData } from "@/store/graphStore";
import { getBlockDef } from "@/lib/blockRegistry";

export function serializeGraph(
  nodes: Node<NodeData>[],
  edges: Edge[],
  metadata?: { name?: string; created_at?: string }
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
      name: metadata?.name ?? "Untitled Model",
      created_at: metadata?.created_at ?? new Date().toISOString(),
    },
  };
}

/** Deserialize a saved GraphSchema into React Flow nodes and edges for loadGraph(). */
export function deserializeGraph(schema: GraphSchema): {
  nodes: Node<NodeData>[];
  edges: Edge[];
} {
  const nodes: Node<NodeData>[] = schema.nodes.map((n) => {
    const def = getBlockDef(n.type);
    return {
      id: n.id,
      type: n.type,
      position: n.position ?? { x: 0, y: 0 },
      data: {
        label: def?.label ?? n.type,
        blockType: n.type,
        params: { ...def?.defaultParams, ...n.params },
      },
    };
  });
  const edges: Edge[] = schema.edges.map((e) => ({
    id: e.id,
    source: e.source,
    sourceHandle: e.sourceHandle ?? "out",
    target: e.target,
    targetHandle: e.targetHandle ?? "in",
    type: "smoothstep",
    animated: true,
  }));
  return { nodes, edges };
}
