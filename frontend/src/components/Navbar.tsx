import { useEffect, useState, type CSSProperties } from "react";
import { sfxClick } from "../utils/sfx";
import GooeyNav from "./GooeyNav";

/* ── Navbar Styles ── */

const nav: CSSProperties = {
  position: "fixed",
  top: 0,
  left: 0,
  right: 0,
  zIndex: 100,
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "0 32px",
  height: 64,
  transition:
    "background 0.35s ease, border-color 0.35s ease, backdrop-filter 0.35s ease",
};

const logoStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  fontWeight: 800,
  fontSize: 18,
  letterSpacing: "-0.02em",
  cursor: "pointer",
};

const logoIcon: CSSProperties = {
  width: 32,
  height: 32,
  borderRadius: 10,
  background: "var(--gradient-primary)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 16,
};

/* ── Component ── */

type Page = "home" | "embed" | "detect";

const PAGES: Page[] = ["home", "embed", "detect"];

interface NavbarProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
}

export default function Navbar({ currentPage, onNavigate }: NavbarProps) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const gooeyItems = PAGES.map((page) => ({
    label: page.charAt(0).toUpperCase() + page.slice(1),
    onClick: () => onNavigate(page),
  }));

  return (
    <nav
      style={{
        ...nav,
        background: scrolled ? "rgba(10,10,15,0.82)" : "transparent",
        backdropFilter: scrolled ? "blur(16px) saturate(1.4)" : "none",
        borderBottom: scrolled
          ? "1px solid var(--border-subtle)"
          : "1px solid transparent",
      }}
    >
      {/* Logo */}
      <div
        style={logoStyle}
        onClick={() => {
          sfxClick();
          onNavigate("home");
        }}
      >
        <div style={logoIcon}>🔏</div>
        <span>
          <span className="gradient-text">Water</span>mark AI
        </span>
      </div>

      {/* Gooey particle nav */}
      <GooeyNav
        items={gooeyItems}
        initialActiveIndex={PAGES.indexOf(currentPage)}
        particleCount={18}
        particleDistances={[90, 10]}
        particleR={100}
        animationTime={1200}
        timeVariance={500}
        colors={[1, 2, 3, 1, 2, 3, 1, 4]}
        onActiveChange={(i) => onNavigate(PAGES[i])}
      />
    </nav>
  );
}
