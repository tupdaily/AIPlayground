"use client";

import dynamic from "next/dynamic";

// React Flow accesses `window` on import, so we must disable SSR.
const NeuralCanvas = dynamic(
  () => import("@/components/canvas/NeuralCanvas"),
  { ssr: false },
);

export default function Home() {
  return <NeuralCanvas />;
}
