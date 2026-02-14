"use client";

import { BLOCK_REGISTRY, BLOCK_CATEGORIES } from "@/lib/blockRegistry";
import { DragEvent } from "react";

export default function Sidebar() {
  const onDragStart = (event: DragEvent, blockType: string) => {
    event.dataTransfer.setData("application/reactflow", blockType);
    event.dataTransfer.effectAllowed = "move";
  };

  return (
    <div className="w-56 bg-gray-50 border-r border-gray-200 overflow-y-auto flex-shrink-0">
      <div className="p-3">
        <h2 className="font-bold text-sm text-gray-700 mb-3">Blocks</h2>
        {BLOCK_CATEGORIES.map((cat) => {
          const blocks = BLOCK_REGISTRY.filter((b) => b.category === cat.key);
          if (blocks.length === 0) return null;
          return (
            <div key={cat.key} className="mb-3">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                {cat.label}
              </h3>
              <div className="space-y-1">
                {blocks.map((block) => (
                  <div
                    key={block.type}
                    draggable
                    onDragStart={(e) => onDragStart(e, block.type)}
                    className="flex items-center gap-2 px-2 py-1.5 rounded cursor-grab hover:bg-gray-100 active:cursor-grabbing border border-transparent hover:border-gray-200 transition-colors"
                  >
                    <div
                      className="w-3 h-3 rounded-sm flex-shrink-0"
                      style={{ backgroundColor: block.color }}
                    />
                    <span className="text-sm text-gray-700">{block.label}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
