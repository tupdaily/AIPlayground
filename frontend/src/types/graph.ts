export interface GraphSchema {
  version: "1.0";
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: {
    name: string;
    created_at: string;
    description?: string;
  };
}

export interface GraphNode {
  id: string;
  type: string;
  params: Record<string, unknown>;
  position: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}
