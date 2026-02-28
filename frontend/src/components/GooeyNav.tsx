import { useCallback, useEffect, useRef, useState } from "react";
import { sfxClick } from "../utils/sfx";
import "./GooeyNav.css";

/* ── Types ── */

interface GooeyNavItem {
  label: string;
  href?: string;
  onClick?: () => void;
}

interface GooeyNavProps {
  items: GooeyNavItem[];
  animationTime?: number;
  particleCount?: number;
  particleDistances?: [number, number];
  particleR?: number;
  timeVariance?: number;
  colors?: number[];
  initialActiveIndex?: number;
  onActiveChange?: (index: number) => void;
}

interface Particle {
  start: [number, number];
  end: [number, number];
  time: number;
  scale: number;
  color: number;
  rotate: number;
}

const noise = (n = 1) => n / 2 - Math.random() * n;

const getXY = (
  distance: number,
  pointIndex: number,
  totalPoints: number,
): [number, number] => {
  const angle = ((360 + noise(8)) / totalPoints) * pointIndex * (Math.PI / 180);
  return [distance * Math.cos(angle), distance * Math.sin(angle)];
};

/* ── Component ── */

export default function GooeyNav({
  items,
  animationTime = 600,
  particleCount = 15,
  particleDistances = [90, 10],
  particleR = 100,
  timeVariance = 300,
  colors = [1, 2, 3, 1, 2, 3, 1, 4],
  initialActiveIndex = 0,
  onActiveChange,
}: GooeyNavProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const navRef = useRef<HTMLUListElement>(null);
  const filterRef = useRef<HTMLSpanElement>(null);
  const textRef = useRef<HTMLSpanElement>(null);
  const [activeIndex, setActiveIndex] = useState(initialActiveIndex);

  const createParticle = useCallback(
    (i: number, t: number, d: [number, number], r: number): Particle => {
      const rotate = noise(r / 10);
      return {
        start: getXY(d[0], particleCount - i, particleCount),
        end: getXY(d[1] + noise(7), particleCount - i, particleCount),
        time: t,
        scale: 1 + noise(0.2),
        color: colors[Math.floor(Math.random() * colors.length)],
        rotate: rotate > 0 ? (rotate + r / 20) * 10 : (rotate - r / 20) * 10,
      };
    },
    [particleCount, colors],
  );

  const makeParticles = useCallback(
    (element: HTMLElement) => {
      const d = particleDistances;
      const r = particleR;
      const bubbleTime = animationTime * 2 + timeVariance;
      element.style.setProperty("--time", `${bubbleTime}ms`);

      for (let i = 0; i < particleCount; i++) {
        const t = animationTime * 2 + noise(timeVariance * 2);
        const p = createParticle(i, t, d, r);
        element.classList.remove("active");

        setTimeout(() => {
          const particle = document.createElement("span");
          const point = document.createElement("span");
          particle.classList.add("particle");
          particle.style.setProperty("--start-x", `${p.start[0]}px`);
          particle.style.setProperty("--start-y", `${p.start[1]}px`);
          particle.style.setProperty("--end-x", `${p.end[0]}px`);
          particle.style.setProperty("--end-y", `${p.end[1]}px`);
          particle.style.setProperty("--time", `${p.time}ms`);
          particle.style.setProperty("--scale", `${p.scale}`);
          particle.style.setProperty(
            "--color",
            `var(--color-${p.color}, white)`,
          );
          particle.style.setProperty("--rotate", `${p.rotate}deg`);

          point.classList.add("point");
          particle.appendChild(point);
          element.appendChild(particle);
          requestAnimationFrame(() => {
            element.classList.add("active");
          });
          setTimeout(() => {
            try {
              element.removeChild(particle);
            } catch {
              // already removed
            }
          }, t);
        }, 30);
      }
    },
    [
      animationTime,
      particleCount,
      particleDistances,
      particleR,
      timeVariance,
      createParticle,
    ],
  );

  const updateEffectPosition = useCallback((element: HTMLElement) => {
    if (!containerRef.current || !filterRef.current || !textRef.current) return;
    const containerRect = containerRef.current.getBoundingClientRect();
    const pos = element.getBoundingClientRect();

    const s = {
      left: `${pos.x - containerRect.x}px`,
      top: `${pos.y - containerRect.y}px`,
      width: `${pos.width}px`,
      height: `${pos.height}px`,
    };
    Object.assign(filterRef.current.style, s);
    Object.assign(textRef.current.style, s);
    textRef.current.innerText = element.innerText;
  }, []);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLLIElement>, index: number) => {
      e.preventDefault();
      const liEl = e.currentTarget;
      if (activeIndex === index) return;

      sfxClick();
      setActiveIndex(index);
      onActiveChange?.(index);
      updateEffectPosition(liEl);

      if (filterRef.current) {
        filterRef.current
          .querySelectorAll(".particle")
          .forEach((p) => p.remove());
      }

      if (textRef.current) {
        textRef.current.classList.remove("active");
        void textRef.current.offsetWidth; // reflow
        textRef.current.classList.add("active");
      }

      if (filterRef.current) {
        makeParticles(filterRef.current);
      }

      items[index]?.onClick?.();
    },
    [activeIndex, items, makeParticles, onActiveChange, updateEffectPosition],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLAnchorElement>, index: number) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        const liEl = e.currentTarget.parentElement as HTMLLIElement | null;
        if (liEl) {
          handleClick(
            {
              currentTarget: liEl,
              preventDefault: () => {},
            } as unknown as React.MouseEvent<HTMLLIElement>,
            index,
          );
        }
      }
    },
    [handleClick],
  );

  // Sync external activeIndex changes
  useEffect(() => {
    setActiveIndex(initialActiveIndex);
  }, [initialActiveIndex]);

  // Position the effect on mount & resize
  useEffect(() => {
    if (!navRef.current || !containerRef.current) return;
    const activeLi = navRef.current.querySelectorAll("li")[activeIndex];
    if (activeLi) {
      updateEffectPosition(activeLi);
      textRef.current?.classList.add("active");
    }

    const resizeObserver = new ResizeObserver(() => {
      const currentActiveLi =
        navRef.current?.querySelectorAll("li")[activeIndex];
      if (currentActiveLi) {
        updateEffectPosition(currentActiveLi);
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [activeIndex, updateEffectPosition]);

  return (
    <div className="gooey-nav-container" ref={containerRef}>
      {/* SVG gooey filter */}
      <svg
        style={{ position: "absolute", width: 0, height: 0 }}
        aria-hidden="true"
      >
        <defs>
          <filter id="gooey-filter">
            <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur" />
            <feColorMatrix
              in="blur"
              mode="matrix"
              values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7"
              result="gooey"
            />
            <feComposite in="SourceGraphic" in2="gooey" operator="atop" />
          </filter>
        </defs>
      </svg>

      <nav>
        <ul ref={navRef}>
          {items.map((item, index) => (
            <li
              key={index}
              className={activeIndex === index ? "active" : ""}
              onClick={(e) => handleClick(e, index)}
            >
              <a
                href={item.href || "#"}
                onClick={(e) => e.preventDefault()}
                onKeyDown={(e) => handleKeyDown(e, index)}
              >
                {item.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>
      <span className="effect filter" ref={filterRef} />
      <span className="effect text" ref={textRef} />
    </div>
  );
}
