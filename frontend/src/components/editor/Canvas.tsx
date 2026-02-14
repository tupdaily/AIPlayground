"use client";

import { useCallback, useRef, DragEvent } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useGraphStore, NodeData } from "@/store/graphStore";
import { BLOCK_REGISTRY } from "@/lib/blockRegistry";
import MLNode from "@/components/nodes/MLNode";

type AppNode = Node<NodeData>;

// Register all block types to use the generic MLNode component
const nodeTypes = Object.fromEntries(
  BLOCK_REGISTRY.map((b) => [b.type, MLNode])
);

export default function Canvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const screenToFlowPosition = useRef<
    ((pos: { x: number; y: number }) => { x: number; y: number }) | null
  >(null);

  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    setSelectedNode,
    addNode,
  } = useGraphStore();

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: AppNode) => {
      setSelectedNode(node.id);
    },
    [setSelectedNode]
  );

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();

      const blockType = event.dataTransfer.getData("application/reactflow");
      if (!blockType || !screenToFlowPosition.current) return;

      const position = screenToFlowPosition.current({
        x: event.clientX,
        y: event.clientY,
      });

      addNode(blockType, position);
    },
    [addNode]
  );

  return (
    <div ref={reactFlowWrapper} className="flex-1 h-full">
      <ReactFlow<AppNode, Edge>
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={(instance) => {
          screenToFlowPosition.current = instance.screenToFlowPosition;
        }}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
        defaultEdgeOptions={{
          type: "smoothstep",
          animated: true,
        }}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={15} size={1} />
        <Controls />
        <MiniMap
          nodeStrokeWidth={3}
          zoomable
          pannable
          className="!bg-gray-100"
        />
      </ReactFlow>
    </div>
  );
}
