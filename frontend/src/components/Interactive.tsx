import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
} from "react";

/* ═══════════════════════════════════════════════════════════════
   1. SCROLL PROGRESS BAR — thin gradient line at the top of viewport
   ═══════════════════════════════════════════════════════════════ */

export function ScrollProgress() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const update = () => {
      if (!ref.current) return;
      const scrollTop = window.scrollY;
      const docHeight =
        document.documentElement.scrollHeight - window.innerHeight;
      const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      ref.current.style.width = `${pct}%`;
    };
    window.addEventListener("scroll", update, { passive: true });
    update();
    return () => window.removeEventListener("scroll", update);
  }, []);

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: 3,
        zIndex: 10001,
        pointerEvents: "none",
      }}
    >
      <div
        ref={ref}
        style={{
          height: "100%",
          background: "var(--gradient-primary)",
          boxShadow: "0 0 12px rgba(99,102,241,0.6)",
          transition: "width 60ms linear",
          willChange: "width",
        }}
      />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   2. ANIMATED COUNTER — counts from 0 to target on scroll into view
   ═══════════════════════════════════════════════════════════════ */

interface CountUpProps {
  target: string; // e.g. "99.9%", ">95%", "<0.05"
  duration?: number;
  className?: string;
  style?: CSSProperties;
}

export function CountUp({
  target,
  duration = 1800,
  className,
  style,
}: CountUpProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const [displayed, setDisplayed] = useState(target);
  const hasAnimated = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true;
          observer.disconnect();

          // Extract numeric part
          const prefix = target.match(/^[^0-9.]*/)?.[0] || "";
          const suffix = target.match(/[^0-9.]*$/)?.[0] || "";
          const numStr = target
            .replace(/^[^0-9.]*/, "")
            .replace(/[^0-9.]*$/, "");
          const numVal = parseFloat(numStr);

          if (isNaN(numVal)) {
            setDisplayed(target);
            return;
          }

          const decimals = numStr.includes(".")
            ? numStr.split(".")[1].length
            : 0;
          const start = performance.now();

          const tick = (now: number) => {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = numVal * eased;
            setDisplayed(`${prefix}${current.toFixed(decimals)}${suffix}`);
            if (progress < 1) requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }
      },
      { threshold: 0.5 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [target, duration]);

  return (
    <span ref={ref} className={className} style={style}>
      {displayed}
    </span>
  );
}

/* ═══════════════════════════════════════════════════════════════
   3. TILT CARD — 3D perspective tilt following the cursor
   ═══════════════════════════════════════════════════════════════ */

interface TiltCardProps {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  maxTilt?: number; // degrees, default 12
  glare?: boolean;
}

export function TiltCard({
  children,
  className,
  style,
  maxTilt = 12,
  glare = true,
}: TiltCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const glareRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      const tiltX = (0.5 - y) * maxTilt;
      const tiltY = (x - 0.5) * maxTilt;
      el.style.transform = `perspective(600px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) scale3d(1.02,1.02,1.02)`;

      if (glare && glareRef.current) {
        const angle = Math.atan2(y - 0.5, x - 0.5) * (180 / Math.PI) + 90;
        glareRef.current.style.background = `linear-gradient(${angle}deg, rgba(255,255,255,0.12) 0%, transparent 80%)`;
        glareRef.current.style.opacity = "1";
      }
    },
    [maxTilt, glare],
  );

  const handleLeave = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.transform =
      "perspective(600px) rotateX(0deg) rotateY(0deg) scale3d(1,1,1)";
    if (glareRef.current) glareRef.current.style.opacity = "0";
  }, []);

  return (
    <div
      ref={ref}
      className={className}
      style={{
        ...style,
        transition: "transform 0.35s var(--ease-out-expo)",
        willChange: "transform",
        position: "relative",
        transformStyle: "preserve-3d",
      }}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
    >
      {children}
      {glare && (
        <div
          ref={glareRef}
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "inherit",
            pointerEvents: "none",
            opacity: 0,
            transition: "opacity 0.35s ease",
            zIndex: 5,
          }}
        />
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   4. RIPPLE BUTTON — material-design ripple on click
   ═══════════════════════════════════════════════════════════════ */

interface RippleButtonProps {
  children: ReactNode;
  onClick?: () => void;
  style?: CSSProperties;
  className?: string;
  disabled?: boolean;
  onMouseEnter?: (e: ReactMouseEvent<HTMLButtonElement>) => void;
  onMouseLeave?: (e: ReactMouseEvent<HTMLButtonElement>) => void;
}

export function RippleButton({
  children,
  onClick,
  style,
  className,
  disabled,
  onMouseEnter,
  onMouseLeave,
}: RippleButtonProps) {
  const btnRef = useRef<HTMLButtonElement>(null);

  const handleClick = (e: ReactMouseEvent<HTMLButtonElement>) => {
    const btn = btnRef.current;
    if (!btn) return;
    const rect = btn.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const size = Math.max(rect.width, rect.height) * 2;

    const ripple = document.createElement("span");
    Object.assign(ripple.style, {
      position: "absolute",
      left: `${x - size / 2}px`,
      top: `${y - size / 2}px`,
      width: `${size}px`,
      height: `${size}px`,
      borderRadius: "50%",
      background: "rgba(255, 255, 255, 0.25)",
      transform: "scale(0)",
      opacity: "1",
      pointerEvents: "none",
      zIndex: "10",
      transition: "transform 0.6s ease-out, opacity 0.6s ease-out",
    });
    btn.appendChild(ripple);

    requestAnimationFrame(() => {
      ripple.style.transform = "scale(1)";
      ripple.style.opacity = "0";
    });

    setTimeout(() => {
      try {
        btn.removeChild(ripple);
      } catch {
        // already removed
      }
    }, 650);

    onClick?.();
  };

  return (
    <button
      ref={btnRef}
      style={{ ...style, position: "relative", overflow: "hidden" }}
      className={className}
      onClick={handleClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

/* ═══════════════════════════════════════════════════════════════
   5. MAGNETIC BUTTON — subtly pulls toward cursor on hover
   ═══════════════════════════════════════════════════════════════ */

interface MagneticProps {
  children: ReactNode;
  strength?: number; // pixels pull, default 12
  className?: string;
  style?: CSSProperties;
}

export function Magnetic({
  children,
  strength = 12,
  className,
  style,
}: MagneticProps) {
  const ref = useRef<HTMLDivElement>(null);

  const handleMove = (e: ReactMouseEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = (e.clientX - cx) / (rect.width / 2);
    const dy = (e.clientY - cy) / (rect.height / 2);
    el.style.transform = `translate(${dx * strength}px, ${dy * strength}px)`;
  };

  const handleLeave = () => {
    if (ref.current) ref.current.style.transform = "translate(0, 0)";
  };

  return (
    <div
      ref={ref}
      className={className}
      style={{
        ...style,
        display: "inline-block",
        transition: "transform 0.35s var(--ease-out-expo)",
        willChange: "transform",
      }}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
    >
      {children}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   6. PAGE TRANSITION WRAPPER — fade + slide when page changes
   ═══════════════════════════════════════════════════════════════ */

interface PageTransitionProps {
  children: ReactNode;
  pageKey: string;
}

export function PageTransition({ children, pageKey }: PageTransitionProps) {
  const [show, setShow] = useState(false);
  const [currentKey, setCurrentKey] = useState(pageKey);
  const [renderChildren, setRenderChildren] = useState(children);

  useEffect(() => {
    if (pageKey !== currentKey) {
      // Fade out
      setShow(false);
      const timer = setTimeout(() => {
        setRenderChildren(children);
        setCurrentKey(pageKey);
        // Fade in
        requestAnimationFrame(() => setShow(true));
      }, 350);
      return () => clearTimeout(timer);
    } else {
      // Initial mount
      requestAnimationFrame(() => setShow(true));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageKey, children]);

  return (
    <div
      style={{
        opacity: show ? 1 : 0,
        transform: show ? "translateY(0)" : "translateY(24px)",
        transition:
          "opacity 0.45s var(--ease-out-expo), transform 0.45s var(--ease-out-expo)",
        willChange: "opacity, transform",
      }}
    >
      {renderChildren}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   7. CURSOR TRAIL — softly glowing dots trailing the mouse
   ═══════════════════════════════════════════════════════════════ */

export function CursorTrail({
  dotCount = 8,
  color = "rgba(99, 102, 241, 0.5)",
}: {
  dotCount?: number;
  color?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouse = useRef({ x: -100, y: -100 });
  const dots = useRef<{ x: number; y: number }[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Initialize dots offscreen
    dots.current = Array.from({ length: dotCount }, () => ({
      x: -100,
      y: -100,
    }));

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const move = (e: MouseEvent) => {
      mouse.current.x = e.clientX;
      mouse.current.y = e.clientY;
    };
    window.addEventListener("mousemove", move, { passive: true });

    let raf: number;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Leader follows mouse, others follow previous dot
      const pts = dots.current;
      pts[0].x += (mouse.current.x - pts[0].x) * 0.35;
      pts[0].y += (mouse.current.y - pts[0].y) * 0.35;
      for (let i = 1; i < pts.length; i++) {
        pts[i].x += (pts[i - 1].x - pts[i].x) * 0.3;
        pts[i].y += (pts[i - 1].y - pts[i].y) * 0.3;
      }

      // Draw dots with decreasing size + opacity
      for (let i = pts.length - 1; i >= 0; i--) {
        const t = 1 - i / pts.length;
        const radius = 4 * t + 1;
        ctx.beginPath();
        ctx.arc(pts[i].x, pts[i].y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color.replace(/[\d.]+\)$/, `${(0.5 * t).toFixed(2)})`);
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", move);
    };
  }, [dotCount, color]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        inset: 0,
        pointerEvents: "none",
        zIndex: 9998,
      }}
    />
  );
}

/* ═══════════════════════════════════════════════════════════════
   8. FLOATING PARTICLES — ambient dots drifting through background
   ═══════════════════════════════════════════════════════════════ */

export function FloatingParticles({ count = 40 }: { count?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const particles = Array.from({ length: count }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3 - 0.15, // slight upward drift
      r: Math.random() * 2 + 0.5,
      alpha: Math.random() * 0.4 + 0.1,
      color: ["#6366f1", "#06b6d4", "#8b5cf6", "#10b981"][
        Math.floor(Math.random() * 4)
      ],
    }));

    let raf: number;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;

        // Wrap around
        if (p.x < -10) p.x = canvas.width + 10;
        if (p.x > canvas.width + 10) p.x = -10;
        if (p.y < -10) p.y = canvas.height + 10;
        if (p.y > canvas.height + 10) p.y = -10;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Draw faint connections between nearby particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(99, 102, 241, ${0.06 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [count]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        inset: 0,
        pointerEvents: "none",
        zIndex: 0,
      }}
    />
  );
}

/* ═══════════════════════════════════════════════════════════════
   9. TOOLTIP — animated tooltip on hover
   ═══════════════════════════════════════════════════════════════ */

interface TooltipProps {
  children: ReactNode;
  text: string;
  position?: "top" | "bottom";
}

export function Tooltip({ children, text, position = "top" }: TooltipProps) {
  const [show, setShow] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const ref = useRef<HTMLDivElement>(null);

  const handleEnter = () => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    setCoords({
      x: rect.left + rect.width / 2,
      y: position === "top" ? rect.top - 8 : rect.bottom + 8,
    });
    setShow(true);
  };

  return (
    <div
      ref={ref}
      style={{ display: "inline-block", position: "relative" }}
      onMouseEnter={handleEnter}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div
          style={{
            position: "fixed",
            left: coords.x,
            top: coords.y,
            transform: `translate(-50%, ${position === "top" ? "-100%" : "0"})`,
            padding: "6px 12px",
            borderRadius: 8,
            background: "rgba(22,22,30,0.95)",
            border: "1px solid var(--border-subtle)",
            color: "var(--text-primary)",
            fontSize: 12,
            fontWeight: 500,
            whiteSpace: "nowrap",
            zIndex: 10002,
            pointerEvents: "none",
            opacity: show ? 1 : 0,
            animation: "tooltip-in 0.2s var(--ease-out-expo)",
          }}
        >
          {text}
        </div>
      )}
    </div>
  );
}
