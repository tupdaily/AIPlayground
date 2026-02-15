"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";

/**
 * Compact sun/moon toggle for light/dark mode.
 * Uses next-themes under the hood, with `data-theme` attribute on <html>.
 */
export function ThemeToggle({ className = "" }: { className?: string }) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch — render nothing until mounted
  useEffect(() => setMounted(true), []);

  if (!mounted) {
    return (
      <div
        className={`w-8 h-8 rounded-full border border-[var(--border)] bg-[var(--surface)] ${className}`}
      />
    );
  }

  const isDark = theme === "dark";

  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className={`
        flex items-center justify-center w-8 h-8 rounded-full
        border transition-all duration-200 select-none shadow-sm
        ${
          isDark
            ? "bg-[var(--surface)] border-[var(--border)] text-amber-400 hover:bg-[var(--surface-hover)] hover:border-amber-500/40 hover:shadow-[0_0_12px_rgba(251,191,36,0.15)]"
            : "bg-white border-[var(--border)] text-indigo-500 hover:bg-indigo-50 hover:border-indigo-200 hover:shadow-md"
        }
        ${className}
      `}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {isDark ? <Sun size={14} /> : <Moon size={14} />}
    </button>
  );
}
