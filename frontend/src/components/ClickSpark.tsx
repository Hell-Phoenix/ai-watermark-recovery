import { useRef, type MouseEvent, type ReactNode } from "react";

/**
 * ClickSpark — spawns small animated spark particles on every click.
 * Inspired by react-bits ClickSpark. Pure CSS + DOM, no canvas needed.
 */

interface ClickSparkProps {
  children: ReactNode;
  /** Number of sparks per click (default: 8) */
  sparkCount?: number;
  /** Spark travel distance in px (default: 30) */
  sparkRadius?: number;
  /** Duration in ms (default: 400) */
  duration?: number;
  /** Spark colours — cycles through list */
  colors?: string[];
  /** Spark size in px (default: 3) */
  sparkSize?: number;
}

const DEFAULT_COLORS = [
  "#6366f1", // indigo
  "#06b6d4", // cyan
  "#8b5cf6", // violet
  "#f59e0b", // amber
  "#10b981", // emerald
  "#f43f5e", // rose
];

export default function ClickSpark({
  children,
  sparkCount = 8,
  sparkRadius = 30,
  duration = 400,
  colors = DEFAULT_COLORS,
  sparkSize = 3,
}: ClickSparkProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const handleClick = (e: MouseEvent<HTMLDivElement>) => {
    const container = containerRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    for (let i = 0; i < sparkCount; i++) {
      const spark = document.createElement("div");
      const angle =
        (2 * Math.PI * i) / sparkCount + (Math.random() - 0.5) * 0.5;
      const dist = sparkRadius * (0.6 + Math.random() * 0.4);
      const dx = Math.cos(angle) * dist;
      const dy = Math.sin(angle) * dist;
      const color = colors[i % colors.length];
      const size = sparkSize + Math.random() * 2;

      Object.assign(spark.style, {
        position: "absolute",
        left: `${x}px`,
        top: `${y}px`,
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: "50%",
        background: color,
        boxShadow: `0 0 6px ${color}, 0 0 10px ${color}`,
        pointerEvents: "none",
        zIndex: "9999",
        transform: "translate(-50%, -50%) scale(1)",
        opacity: "1",
        transition: `transform ${duration}ms cubic-bezier(0.22, 1, 0.36, 1), opacity ${duration}ms ease-out`,
      });

      container.appendChild(spark);

      // Trigger reflow then animate
      spark.getBoundingClientRect();
      requestAnimationFrame(() => {
        spark.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px)) scale(0)`;
        spark.style.opacity = "0";
      });

      setTimeout(() => spark.remove(), duration + 50);
    }
  };

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      style={{ position: "relative", display: "contents" }}
    >
      <div style={{ position: "relative" }}>{children}</div>
    </div>
  );
}
