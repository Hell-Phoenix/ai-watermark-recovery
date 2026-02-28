import { useState, type CSSProperties } from "react";
import { sfxHover, sfxPress } from "../utils/sfx";
import { ParallaxLayer, ScrollReveal, TypeWriter } from "./Effects";
import {
  CountUp,
  Magnetic,
  RippleButton,
  TiltCard,
  Tooltip,
} from "./Interactive";
import SpotlightCard from "./SpotlightCard";

/* ── CSS module-in-JS (inline styles for zero-config) ─────────── */

const section: CSSProperties = {
  position: "relative",
  minHeight: "100vh",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  padding: "120px 24px 80px",
  overflow: "hidden",
  zIndex: 1,
};

const badge: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  padding: "6px 16px",
  borderRadius: 100,
  background: "rgba(99,102,241,0.12)",
  border: "1px solid rgba(99,102,241,0.25)",
  fontSize: 13,
  fontWeight: 500,
  color: "var(--accent-indigo)",
  letterSpacing: 1.5,
  textTransform: "uppercase",
  marginBottom: 24,
};

const h1Style: CSSProperties = {
  fontSize: "clamp(2.8rem, 7vw, 5.5rem)",
  fontWeight: 900,
  lineHeight: 1.05,
  letterSpacing: "-0.03em",
  textAlign: "center",
  maxWidth: 900,
  marginBottom: 24,
};

const subtitle: CSSProperties = {
  fontSize: "clamp(1.05rem, 2vw, 1.25rem)",
  color: "var(--text-secondary)",
  textAlign: "center",
  maxWidth: 640,
  lineHeight: 1.7,
  marginBottom: 48,
};

const statsRow: CSSProperties = {
  display: "flex",
  gap: 48,
  flexWrap: "wrap",
  justifyContent: "center",
  marginTop: 40,
};

const stat: CSSProperties = {
  textAlign: "center" as const,
};

const statNumber: CSSProperties = {
  fontSize: "2.2rem",
  fontWeight: 800,
  lineHeight: 1,
};

const statLabel: CSSProperties = {
  fontSize: 13,
  color: "var(--text-muted)",
  marginTop: 6,
  textTransform: "uppercase" as const,
  letterSpacing: 1,
};

const ctaRow: CSSProperties = {
  display: "flex",
  gap: 16,
  flexWrap: "wrap",
  justifyContent: "center",
};

const btnPrimary: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 10,
  padding: "16px 36px",
  borderRadius: 14,
  background: "var(--gradient-primary)",
  color: "#fff",
  fontWeight: 600,
  fontSize: 16,
  border: "none",
  cursor: "pointer",
  transition: "transform 0.3s var(--ease-spring), box-shadow 0.3s ease",
  boxShadow: "0 4px 24px rgba(99,102,241,0.3)",
};

const btnSecondary: CSSProperties = {
  ...btnPrimary,
  background: "transparent",
  border: "1px solid var(--border-glow)",
  boxShadow: "none",
  color: "var(--text-primary)",
};

/* ── Features data ── */

const features = [
  {
    icon: "🛡️",
    title: "Dual-Domain Embedding",
    desc: "Latent (WIND) + pixel (iIWN) watermarking survives even generative regeneration attacks.",
    gradient: "var(--gradient-primary)",
  },
  {
    icon: "🔬",
    title: "IGRM Restoration",
    desc: "Inverse Generative Restoration rebuilds watermark bit structure from heavily degraded images.",
    gradient: "var(--gradient-secondary)",
  },
  {
    icon: "⚡",
    title: "Attack Simulation",
    desc: "Differentiable JPEG, crop, rotation, and diffusion attack layers for end-to-end training.",
    gradient: "var(--gradient-success)",
  },
  {
    icon: "🔐",
    title: "MetaSeal Binding",
    desc: "DINOv2 perceptual hash + ECDSA signatures prevent transplantation forgery attacks.",
    gradient: "var(--gradient-warm)",
  },
];

const featureCardInner: CSSProperties = {
  padding: 32,
  cursor: "default",
};

/* ── Component ────────────────────────────────────────────────── */

interface HeroProps {
  onNavigate: (page: "embed" | "detect") => void;
}

export default function Hero({ onNavigate }: HeroProps) {
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

  return (
    <>
      {/* ── Hero Section ── */}
      <section style={section}>
        <ParallaxLayer speed={-0.15}>
          {/* Giant blurred badge behind title */}
          <div
            style={{
              position: "absolute",
              width: 500,
              height: 500,
              borderRadius: "50%",
              background:
                "radial-gradient(circle, rgba(99,102,241,0.08), transparent 70%)",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              filter: "blur(80px)",
              pointerEvents: "none",
            }}
          />
        </ParallaxLayer>

        <ScrollReveal>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <div style={badge}>
              <span style={{ fontSize: 10 }}>●</span>
              Forensic AI System
            </div>

            <h1 style={h1Style}>
              <span className="gradient-text">Watermark Recovery</span>
              <br />
              <span style={{ color: "var(--text-primary)" }}>
                Beyond Destruction
              </span>
            </h1>

            <p style={subtitle}>
              Deep learning-powered watermark embedding, attack simulation, and{" "}
              <strong style={{ color: "var(--accent-cyan)" }}>
                blind recovery
              </strong>{" "}
              from severely degraded images — surviving JPEG QF=5, 95% crops,
              and generative regeneration attacks.
            </p>

            <div style={ctaRow}>
              <Magnetic strength={10}>
                <RippleButton
                  style={btnPrimary}
                  onClick={() => {
                    sfxPress();
                    onNavigate("embed");
                  }}
                  onMouseEnter={(e) => {
                    sfxHover();
                    e.currentTarget.style.transform =
                      "translateY(-3px) scale(1.02)";
                    e.currentTarget.style.boxShadow =
                      "0 8px 40px rgba(99,102,241,0.45)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0) scale(1)";
                    e.currentTarget.style.boxShadow =
                      "0 4px 24px rgba(99,102,241,0.3)";
                  }}
                >
                  <span>🔏</span>
                  Embed Watermark
                </RippleButton>
              </Magnetic>
              <Magnetic strength={10}>
                <RippleButton
                  style={btnSecondary}
                  onClick={() => {
                    sfxPress();
                    onNavigate("detect");
                  }}
                  onMouseEnter={(e) => {
                    sfxHover();
                    e.currentTarget.style.transform = "translateY(-3px)";
                    e.currentTarget.style.borderColor = "rgba(6,182,212,0.5)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.borderColor = "var(--border-glow)";
                  }}
                >
                  <span>🔍</span>
                  Detect & Recover
                </RippleButton>
              </Magnetic>
            </div>

            {/* Terminal-style output */}
            <div
              style={{
                marginTop: 56,
                padding: "20px 28px",
                borderRadius: "var(--radius-md)",
                background: "rgba(0,0,0,0.4)",
                border: "1px solid var(--border-subtle)",
                fontFamily: "var(--font-mono)",
                fontSize: 14,
                color: "var(--accent-emerald)",
                maxWidth: 620,
                width: "100%",
              }}
            >
              <span style={{ color: "var(--text-muted)" }}>$</span>{" "}
              <TypeWriter
                text='POST /api/v1/detect {"confidence": 0.97, "attack_type": "D2RA"}'
                speed={35}
              />
            </div>
          </div>
        </ScrollReveal>

        <ScrollReveal delay={0.3}>
          <div style={statsRow}>
            {[
              { value: "99.9%", label: "Clean TPR" },
              { value: ">95%", label: "Regen Survival" },
              { value: "<0.05", label: "BER @ QF=10" },
              { value: "0%", label: "Forgery Success" },
            ].map((s, i) => (
              <div key={i} style={stat}>
                <CountUp
                  target={s.value}
                  className="gradient-text"
                  style={statNumber}
                  duration={2000}
                />
                <div style={statLabel}>{s.label}</div>
              </div>
            ))}
          </div>
        </ScrollReveal>
      </section>

      {/* ── Features Section ── */}
      <section
        style={{
          ...section,
          minHeight: "auto",
          padding: "80px 24px 120px",
        }}
      >
        <ScrollReveal>
          <h2
            style={{
              fontSize: "clamp(2rem, 4vw, 3rem)",
              fontWeight: 800,
              textAlign: "center",
              marginBottom: 16,
            }}
          >
            How It <span className="gradient-text">Works</span>
          </h2>
          <p
            style={{
              ...subtitle,
              marginBottom: 64,
            }}
          >
            A 5-layer ML pipeline implementing the full research blueprint —
            from embedding to forensic recovery.
          </p>
        </ScrollReveal>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 24,
            maxWidth: 1200,
            width: "100%",
          }}
        >
          {features.map((f, i) => (
            <ScrollReveal
              key={i}
              delay={i * 0.12}
              direction={i % 2 === 0 ? "left" : "right"}
            >
              <TiltCard maxTilt={8}>
                <SpotlightCard
                  spotlightColor={
                    f.gradient.includes("cyan")
                      ? "rgba(6,182,212,0.55)"
                      : "rgba(99,102,241,0.55)"
                  }
                >
                  <div
                    style={{
                      ...featureCardInner,
                      transform:
                        hoveredCard === i
                          ? "translateY(-4px)"
                          : "translateY(0)",
                      transition: "transform 0.4s var(--ease-out-expo)",
                    }}
                    onMouseEnter={() => setHoveredCard(i)}
                    onMouseLeave={() => setHoveredCard(null)}
                  >
                    {/* Gradient bar at top */}
                    <div
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        right: 0,
                        height: 3,
                        background: f.gradient,
                        opacity: hoveredCard === i ? 1 : 0,
                        transition: "opacity 0.4s ease",
                        zIndex: 3,
                      }}
                    />

                    <div style={{ fontSize: 36, marginBottom: 16 }}>
                      {f.icon}
                    </div>
                    <h3
                      style={{
                        fontSize: 20,
                        fontWeight: 700,
                        marginBottom: 10,
                        letterSpacing: "-0.01em",
                      }}
                    >
                      {f.title}
                    </h3>
                    <p
                      style={{
                        color: "var(--text-secondary)",
                        fontSize: 15,
                        lineHeight: 1.7,
                      }}
                    >
                      {f.desc}
                    </p>
                  </div>
                </SpotlightCard>
              </TiltCard>
            </ScrollReveal>
          ))}
        </div>
      </section>

      {/* ── Architecture Section ── */}
      <section
        style={{ ...section, minHeight: "auto", padding: "40px 24px 120px" }}
      >
        <ScrollReveal>
          <div
            className="glass"
            style={{
              maxWidth: 900,
              width: "100%",
              borderRadius: "var(--radius-xl)",
              padding: "48px 40px",
              textAlign: "center",
            }}
          >
            <h2
              style={{ fontSize: "1.8rem", fontWeight: 800, marginBottom: 32 }}
            >
              Pipeline <span className="gradient-text">Architecture</span>
            </h2>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 12,
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              {[
                {
                  name: "Encoder (HiDDeN)",
                  tip: "Embeds watermark into latent space",
                },
                { name: "→", tip: "" },
                {
                  name: "ASL Attack Sim",
                  tip: "Differentiable JPEG, crop & noise attacks",
                },
                { name: "→", tip: "" },
                {
                  name: "IGRM Restore",
                  tip: "Inverse generative restoration module",
                },
                { name: "→", tip: "" },
                { name: "Swin Decoder", tip: "Transformer-based bit recovery" },
                { name: "→", tip: "" },
                {
                  name: "MetaSeal Verify",
                  tip: "ECDSA signature + perceptual hash",
                },
              ].map((item, i) =>
                item.name === "→" ? (
                  <span
                    key={i}
                    style={{
                      color: "var(--accent-cyan)",
                      fontSize: 24,
                      fontWeight: 300,
                    }}
                  >
                    →
                  </span>
                ) : (
                  <Tooltip key={i} text={item.tip}>
                    <div
                      style={{
                        padding: "12px 20px",
                        borderRadius: "var(--radius-sm)",
                        background: "var(--bg-card)",
                        border: "1px solid var(--border-subtle)",
                        fontFamily: "var(--font-mono)",
                        fontSize: 13,
                        fontWeight: 500,
                        transition:
                          "border-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s var(--ease-spring)",
                        cursor: "default",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor =
                          "var(--border-glow)";
                        e.currentTarget.style.boxShadow = "var(--shadow-glow)";
                        e.currentTarget.style.transform = "translateY(-3px)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor =
                          "var(--border-subtle)";
                        e.currentTarget.style.boxShadow = "none";
                        e.currentTarget.style.transform = "translateY(0)";
                      }}
                    >
                      {item.name}
                    </div>
                  </Tooltip>
                ),
              )}
            </div>

            <div
              style={{
                marginTop: 32,
                display: "flex",
                gap: 32,
                justifyContent: "center",
                flexWrap: "wrap",
              }}
            >
              {[
                { label: "ML Modules", value: "13" },
                { label: "Tests Passing", value: "246" },
                { label: "API Endpoints", value: "6" },
                { label: "Phases Done", value: "6" },
              ].map((s, i) => (
                <div key={i} style={{ textAlign: "center" }}>
                  <CountUp
                    target={s.value}
                    className="gradient-text"
                    style={{
                      fontSize: "1.5rem",
                      fontWeight: 800,
                      display: "block",
                    }}
                    duration={1500}
                  />
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--text-muted)",
                      marginTop: 4,
                    }}
                  >
                    {s.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </ScrollReveal>
      </section>
    </>
  );
}
