import { useEffect, useRef, useState, type CSSProperties } from "react";

/* ── Floating Orbs (ambient background) ───────────────────────── */

interface OrbProps {
  size: number;
  color: string;
  top: string;
  left: string;
  delay: number;
}

function Orb({ size, color, top, left, delay }: OrbProps) {
  const style: CSSProperties = {
    position: "absolute",
    width: size,
    height: size,
    borderRadius: "50%",
    background: `radial-gradient(circle, ${color}33, transparent 70%)`,
    top,
    left,
    animation: `float 8s ease-in-out ${delay}s infinite`,
    filter: `blur(${size / 3}px)`,
    pointerEvents: "none",
  };
  return <div style={style} />;
}

/* ── Parallax Container ───────────────────────────────────────── */

interface ParallaxProps {
  children: React.ReactNode;
  speed?: number;
  className?: string;
}

export function ParallaxLayer({
  children,
  speed = 0.5,
  className,
}: ParallaxProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      if (!ref.current) return;
      const scrollY = window.scrollY;
      ref.current.style.transform = `translateY(${scrollY * speed}px)`;
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, [speed]);

  return (
    <div ref={ref} className={className} style={{ willChange: "transform" }}>
      {children}
    </div>
  );
}

/* ── Scroll-Reveal Animation ──────────────────────────────────── */

interface RevealProps {
  children: React.ReactNode;
  delay?: number;
  direction?: "up" | "left" | "right";
  className?: string;
}

export function ScrollReveal({
  children,
  delay = 0,
  direction = "up",
  className,
}: RevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.15 },
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  const transforms: Record<string, string> = {
    up: "translateY(60px)",
    left: "translateX(-60px)",
    right: "translateX(60px)",
  };

  const style: CSSProperties = {
    opacity: visible ? 1 : 0,
    transform: visible ? "translate(0)" : transforms[direction],
    transition: `opacity 0.8s var(--ease-out-expo) ${delay}s, transform 0.8s var(--ease-out-expo) ${delay}s`,
    willChange: "opacity, transform",
  };

  return (
    <div ref={ref} style={style} className={className}>
      {children}
    </div>
  );
}

/* ── Animated Background ──────────────────────────────────────── */

export function AnimatedBackground() {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        overflow: "hidden",
        pointerEvents: "none",
        zIndex: 0,
      }}
    >
      <Orb size={600} color="#6366f1" top="-10%" left="-5%" delay={0} />
      <Orb size={400} color="#06b6d4" top="30%" left="80%" delay={2} />
      <Orb size={500} color="#8b5cf6" top="70%" left="10%" delay={4} />
      <Orb size={350} color="#f43f5e" top="60%" left="60%" delay={1} />
    </div>
  );
}

/* ── Noise Grain Overlay ──────────────────────────────────────── */

export function GrainOverlay() {
  return (
    <div
      style={{
        position: "fixed",
        inset: "-50%",
        width: "200%",
        height: "200%",
        background: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E")`,
        animation: "grain 0.5s steps(1) infinite",
        pointerEvents: "none",
        zIndex: 9999,
        opacity: 0.4,
      }}
    />
  );
}

/* ── Magnetic Cursor Glow (for cards) ─────────────────────────── */

export function useMouseGlow() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const handleMove = (e: MouseEvent) => {
      const rect = el.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      el.style.setProperty("--mouse-x", `${x}px`);
      el.style.setProperty("--mouse-y", `${y}px`);
    };

    el.addEventListener("mousemove", handleMove);
    return () => el.removeEventListener("mousemove", handleMove);
  }, []);

  return ref;
}

/* ── Typing Effect ────────────────────────────────────────────── */

export function TypeWriter({
  text,
  speed = 50,
}: {
  text: string;
  speed?: number;
}) {
  const [displayed, setDisplayed] = useState("");
  const [cursorVisible, setCursorVisible] = useState(true);

  useEffect(() => {
    let i = 0;
    const timer = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1));
        i++;
      } else {
        clearInterval(timer);
      }
    }, speed);
    return () => clearInterval(timer);
  }, [text, speed]);

  useEffect(() => {
    const blink = setInterval(() => setCursorVisible((v) => !v), 530);
    return () => clearInterval(blink);
  }, []);

  return (
    <span style={{ fontFamily: "var(--font-mono)" }}>
      {displayed}
      <span
        style={{ opacity: cursorVisible ? 1 : 0, color: "var(--accent-cyan)" }}
      >
        |
      </span>
    </span>
  );
}
