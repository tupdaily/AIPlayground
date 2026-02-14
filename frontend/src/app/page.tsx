"use client";

import { ReactFlowProvider } from "@xyflow/react";
import Canvas from "@/components/editor/Canvas";
import Sidebar from "@/components/editor/Sidebar";
import Toolbar from "@/components/editor/Toolbar";
import PropertiesPanel from "@/components/editor/PropertiesPanel";
import TrainingDashboard from "@/components/training/TrainingDashboard";

export default function Home() {
  return (
    <ReactFlowProvider>
      <div className="h-screen w-screen flex flex-col overflow-hidden">
        <Toolbar />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <Canvas />
          <PropertiesPanel />
          <TrainingDashboard />
        </div>
      </div>
    </ReactFlowProvider>
  );
}
