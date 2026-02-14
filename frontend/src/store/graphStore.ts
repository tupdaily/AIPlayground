import { create } from "zustand";
import {
  Node,
  Edge,
  OnNodesChange,
  OnEdgesChange,
  OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  Connection,
} from "@xyflow/react";
import { getBlockDef } from "@/lib/blockRegistry";

export interface NodeData {
  label: string;
  blockType: string;
  params: Record<string, unknown>;
  inferredShape?: number[];
  [key: string]: unknown;
}

interface GraphState {
  nodes: Node<NodeData>[];
  edges: Edge[];
  selectedNodeId: string | null;
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  setSelectedNode: (id: string | null) => void;
  addNode: (type: string, position: { x: number; y: number }) => void;
  updateNodeParams: (nodeId: string, params: Record<string, unknown>) => void;
  updateNodeShape: (nodeId: string, shape: number[]) => void;
  clearGraph: () => void;
  loadGraph: (nodes: Node<NodeData>[], edges: Edge[]) => void;
}

let nodeIdCounter = 0;

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,

  onNodesChange: (changes) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes) as Node<NodeData>[],
    });
  },

  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },

  onConnect: (connection: Connection) => {
    const { edges, nodes } = get();
    const targetNode = nodes.find((n) => n.id === connection.target);

    // Prevent duplicate connections to the same target handle (unless merge node)
    const existingToHandle = edges.find(
      (e) =>
        e.target === connection.target &&
        e.targetHandle === connection.targetHandle
    );
    if (existingToHandle) {
      return;
    }

    // Prevent self-connections
    if (connection.source === connection.target) return;

    // Prevent connecting to non-existent nodes
    if (!targetNode) return;

    set({
      edges: addEdge(
        { ...connection, type: "smoothstep", animated: true },
        edges
      ),
    });
  },

  setSelectedNode: (id) => set({ selectedNodeId: id }),

  addNode: (type, position) => {
    const def = getBlockDef(type);
    if (!def) return;

    const id = `node_${++nodeIdCounter}_${Date.now()}`;
    const newNode: Node<NodeData> = {
      id,
      type,
      position,
      data: {
        label: def.label,
        blockType: type,
        params: { ...def.defaultParams },
      },
    };

    set({ nodes: [...get().nodes, newNode] });
  },

  updateNodeParams: (nodeId, params) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...n.data.params, ...params } } }
          : n
      ),
    });
  },

  updateNodeShape: (nodeId, shape) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, inferredShape: shape } }
          : n
      ),
    });
  },

  clearGraph: () => {
    set({ nodes: [], edges: [], selectedNodeId: null });
  },

  loadGraph: (nodes, edges) => {
    set({ nodes, edges, selectedNodeId: null });
  },
}));
