"use client";

import { createContext, useContext, type ReactNode } from "react";

/** Context for current playground ID so Model block can list/save models without prop drilling. */
const PlaygroundIdContext = createContext<string | undefined>(undefined);

export function PlaygroundIdProvider({
  playgroundId,
  children,
}: {
  playgroundId: string | undefined;
  children: ReactNode;
}) {
  return (
    <PlaygroundIdContext.Provider value={playgroundId}>
      {children}
    </PlaygroundIdContext.Provider>
  );
}

export function usePlaygroundId(): string | undefined {
  return useContext(PlaygroundIdContext);
}
