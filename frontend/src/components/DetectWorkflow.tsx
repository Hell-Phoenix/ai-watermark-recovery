import { useState, type CSSProperties } from "react";
import {
  sfxClick,
  sfxError,
  sfxPress,
  sfxSuccess,
  sfxToggle,
} from "../utils/sfx";
import { ScrollReveal } from "./Effects";
import ImageUploader from "./ImageUploader";
import { Magnetic, RippleButton } from "./Interactive";
import SpotlightCard from "./SpotlightCard";

/* ── Styles ── */

const page: CSSProperties = {
  minHeight: "100vh",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  padding: "100px 24px 80px",
  position: "relative",
  zIndex: 1,
};

const heading: CSSProperties = {
  fontSize: "clamp(2rem, 4vw, 3rem)",
  fontWeight: 800,
  textAlign: "center",
  marginBottom: 8,
};

const sub: CSSProperties = {
  fontSize: 15,
  color: "var(--text-secondary)",
  textAlign: "center",
  marginBottom: 48,
  maxWidth: 520,
};

const card: CSSProperties = {
  width: "100%",
  maxWidth: 640,
  padding: 36,
};

const submitBtn: CSSProperties = {
  width: "100%",
  padding: "16px",
  borderRadius: "var(--radius-md)",
  background: "var(--gradient-secondary)",
  border: "none",
  color: "#fff",
  fontSize: 16,
  fontWeight: 600,
  cursor: "pointer",
  transition: "box-shadow 0.3s ease, transform 0.2s ease, opacity 0.3s ease",
  boxShadow: "0 4px 20px rgba(6,182,212,0.3)",
  marginTop: 24,
};

/* ── Confidence Ring ── */

function ConfidenceRing({
  value,
  size = 140,
}: {
  value: number;
  size?: number;
}) {
  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - value);
  const color =
    value >= 0.9
      ? "var(--accent-emerald)"
      : value >= 0.7
        ? "var(--accent-amber)"
        : "var(--accent-rose)";

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={8}
        />
        {/* Progress */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{
            transition:
              "stroke-dashoffset 1.2s var(--ease-out-expo), stroke 0.5s ease",
          }}
        />
      </svg>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span style={{ fontSize: 28, fontWeight: 800, color }}>
          {(value * 100).toFixed(1)}%
        </span>
        <span
          style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}
        >
          Confidence
        </span>
      </div>
    </div>
  );
}

/* ── Attack Badge ── */

function AttackBadge({ attack }: { attack: string }) {
  const attackColors: Record<string, string> = {
    clean: "var(--accent-emerald)",
    jpeg: "var(--accent-amber)",
    crop: "var(--accent-rose)",
    rotation: "var(--accent-violet)",
    D2RA: "var(--accent-rose)",
    unknown: "var(--text-muted)",
  };
  const color = attackColors[attack] || attackColors.unknown;
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 14px",
        borderRadius: 100,
        background: `${color}15`,
        border: `1px solid ${color}40`,
        color,
        fontSize: 13,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: 0.8,
      }}
    >
      <span style={{ fontSize: 8 }}>●</span>
      {attack}
    </div>
  );
}

/* ── Tamper Visualization ── */

function TamperOverlay({
  imageUrl,
  tamperMask,
}: {
  imageUrl: string;
  tamperMask: string | null;
}) {
  const [showMask, setShowMask] = useState(true);

  return (
    <div
      style={{
        position: "relative",
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
      }}
    >
      <img
        src={imageUrl}
        alt="Analyzed"
        style={{
          width: "100%",
          display: "block",
          maxHeight: 300,
          objectFit: "cover",
        }}
      />
      {tamperMask && showMask && (
        <img
          src={`data:image/png;base64,${tamperMask}`}
          alt="Tamper map"
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
            opacity: 0.55,
            mixBlendMode: "screen",
            pointerEvents: "none",
          }}
        />
      )}
      {tamperMask && (
        <button
          onClick={() => {
            sfxToggle(!showMask);
            setShowMask((v) => !v);
          }}
          style={{
            position: "absolute",
            top: 12,
            right: 12,
            padding: "6px 12px",
            borderRadius: "var(--radius-sm)",
            background: "rgba(0,0,0,0.6)",
            backdropFilter: "blur(8px)",
            border: "1px solid rgba(255,255,255,0.1)",
            color: "#fff",
            fontSize: 12,
            fontWeight: 500,
            cursor: "pointer",
          }}
        >
          {showMask ? "Hide" : "Show"} Tamper Map
        </button>
      )}
    </div>
  );
}

/* ── Spinner ── */
function Spinner() {
  return (
    <div
      style={{
        width: 20,
        height: 20,
        border: "2px solid rgba(255,255,255,0.2)",
        borderTopColor: "#fff",
        borderRadius: "50%",
        animation: "spin 0.6s linear infinite",
        display: "inline-block",
      }}
    />
  );
}

/* ── Main Component ── */

interface DetectWorkflowProps {
  onBack: () => void;
}

export default function DetectWorkflow({ onBack }: DetectWorkflowProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<
    "idle" | "uploading" | "processing" | "success" | "error"
  >("idle");
  const [result, setResult] = useState<{
    confidence: number;
    attack_type: string;
    payload_hex: string;
    ber: number;
    tamper_mask: string | null;
    integrity: { latent: boolean; pixel: boolean; metaseal: boolean };
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = file && status === "idle";

  const handleDetect = async () => {
    if (!canSubmit) return;

    setStatus("uploading");
    setError(null);

    try {
      await new Promise((r) => setTimeout(r, 900));
      setStatus("processing");
      await new Promise((r) => setTimeout(r, 2800));

      // Demo result
      setResult({
        confidence: 0.94 + Math.random() * 0.06,
        attack_type: ["clean", "jpeg", "D2RA", "crop"][
          Math.floor(Math.random() * 4)
        ],
        payload_hex: "48656c6c6f",
        ber: Math.random() * 0.05,
        tamper_mask: null,
        integrity: {
          latent: true,
          pixel: Math.random() > 0.3,
          metaseal: Math.random() > 0.2,
        },
      });
      setStatus("success");
      sfxSuccess();
    } catch {
      setError("Detection failed.");
      setStatus("error");
      sfxError();
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setStatus("idle");
    setResult(null);
    setError(null);
  };

  return (
    <div style={page}>
      <ScrollReveal>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <button
            onClick={() => {
              sfxClick();
              onBack();
            }}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-secondary)",
              fontSize: 14,
              cursor: "pointer",
              marginBottom: 24,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            ← Back to Home
          </button>

          <h1 style={heading}>
            <span className="gradient-text">Detect</span> & Recover
          </h1>
          <p style={sub}>
            Upload a potentially attacked image to recover watermark payload and
            assess integrity.
          </p>
        </div>
      </ScrollReveal>

      <ScrollReveal delay={0.1}>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <ImageUploader
            onFileSelected={(f, p) => {
              setFile(f);
              setPreview(p);
            }}
            onClear={() => {
              setFile(null);
              setPreview(null);
            }}
            preview={preview}
            uploading={status === "uploading"}
          />
        </div>
      </ScrollReveal>

      {!result && (
        <ScrollReveal delay={0.15}>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              marginTop: 24,
            }}
          >
            <div style={{ width: "100%", maxWidth: 540 }}>
              {error && (
                <div
                  style={{
                    padding: "12px 16px",
                    borderRadius: "var(--radius-sm)",
                    background: "rgba(244,63,94,0.08)",
                    border: "1px solid rgba(244,63,94,0.2)",
                    color: "var(--accent-rose)",
                    fontSize: 14,
                    marginBottom: 16,
                  }}
                >
                  {error}
                </div>
              )}

              <Magnetic strength={6}>
                <RippleButton
                  style={{
                    ...submitBtn,
                    opacity: canSubmit ? 1 : 0.5,
                    cursor: canSubmit ? "pointer" : "not-allowed",
                  }}
                  disabled={!canSubmit}
                  onClick={() => {
                    sfxPress();
                    handleDetect();
                  }}
                  onMouseEnter={(e) => {
                    if (canSubmit) {
                      e.currentTarget.style.boxShadow =
                        "0 8px 32px rgba(6,182,212,0.45)";
                      e.currentTarget.style.transform = "translateY(-2px)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow =
                      "0 4px 20px rgba(6,182,212,0.3)";
                    e.currentTarget.style.transform = "translateY(0)";
                  }}
                >
                  {status === "idle" && "🔍  Detect & Recover"}
                  {status === "uploading" && (
                    <span
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        justifyContent: "center",
                      }}
                    >
                      <Spinner /> Uploading…
                    </span>
                  )}
                  {status === "processing" && (
                    <span
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        justifyContent: "center",
                      }}
                    >
                      <Spinner /> Analyzing…
                    </span>
                  )}
                  {status === "error" && "Retry"}
                </RippleButton>
              </Magnetic>
            </div>
          </div>
        </ScrollReveal>
      )}

      {/* ── Results ── */}
      {result && (
        <ScrollReveal>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              marginTop: 32,
            }}
          >
            <SpotlightCard spotlightColor="rgba(6, 182, 212, 0.55)">
              <div style={card}>
                {/* Header: Confidence + Attack */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 32,
                    flexWrap: "wrap",
                    marginBottom: 28,
                  }}
                >
                  <ConfidenceRing value={result.confidence} />
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <div style={{ marginBottom: 12 }}>
                      <span
                        style={{
                          fontSize: 13,
                          color: "var(--text-muted)",
                          display: "block",
                          marginBottom: 6,
                        }}
                      >
                        Detected Attack
                      </span>
                      <AttackBadge attack={result.attack_type} />
                    </div>
                    <div>
                      <span
                        style={{
                          fontSize: 13,
                          color: "var(--text-muted)",
                          display: "block",
                          marginBottom: 6,
                        }}
                      >
                        Bit Error Rate
                      </span>
                      <span
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 20,
                          fontWeight: 700,
                          color:
                            result.ber < 0.01
                              ? "var(--accent-emerald)"
                              : result.ber < 0.05
                                ? "var(--accent-amber)"
                                : "var(--accent-rose)",
                        }}
                      >
                        {result.ber.toFixed(4)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Recovered Payload */}
                <div
                  style={{
                    padding: "16px 20px",
                    borderRadius: "var(--radius-md)",
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border-subtle)",
                    marginBottom: 24,
                  }}
                >
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--text-muted)",
                      marginBottom: 8,
                      textTransform: "uppercase",
                      letterSpacing: 1,
                    }}
                  >
                    Recovered Payload
                  </div>
                  <div
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: 16,
                      fontWeight: 600,
                      color: "var(--accent-cyan)",
                      wordBreak: "break-all",
                    }}
                  >
                    0x{result.payload_hex}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--text-muted)",
                      marginTop: 6,
                    }}
                  >
                    Decoded: &quot;
                    {(() => {
                      try {
                        return (
                          result.payload_hex
                            .match(/.{2}/g)
                            ?.map((h) => String.fromCharCode(parseInt(h, 16)))
                            .join("") || ""
                        );
                      } catch {
                        return "?";
                      }
                    })()}
                    &quot;
                  </div>
                </div>

                {/* Integrity Checks */}
                <div style={{ marginBottom: 24 }}>
                  <div
                    style={{
                      fontSize: 13,
                      color: "var(--text-muted)",
                      marginBottom: 12,
                      textTransform: "uppercase",
                      letterSpacing: 1,
                    }}
                  >
                    Integrity Layers
                  </div>
                  <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                    {[
                      { label: "Latent (WIND)", ok: result.integrity.latent },
                      { label: "Pixel (iIWN)", ok: result.integrity.pixel },
                      { label: "MetaSeal", ok: result.integrity.metaseal },
                    ].map((layer, i) => (
                      <div
                        key={i}
                        style={{
                          flex: 1,
                          minWidth: 140,
                          padding: "14px 16px",
                          borderRadius: "var(--radius-sm)",
                          background: layer.ok
                            ? "rgba(16,185,129,0.08)"
                            : "rgba(244,63,94,0.08)",
                          border: `1px solid ${layer.ok ? "rgba(16,185,129,0.25)" : "rgba(244,63,94,0.25)"}`,
                          textAlign: "center",
                        }}
                      >
                        <div style={{ fontSize: 20, marginBottom: 6 }}>
                          {layer.ok ? "✓" : "✗"}
                        </div>
                        <div
                          style={{
                            fontSize: 13,
                            fontWeight: 600,
                            color: layer.ok
                              ? "var(--accent-emerald)"
                              : "var(--accent-rose)",
                          }}
                        >
                          {layer.label}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Tamper Visualization */}
                {preview && (
                  <TamperOverlay
                    imageUrl={preview}
                    tamperMask={result.tamper_mask}
                  />
                )}

                {/* Reset */}
                <button
                  onClick={() => {
                    sfxClick();
                    handleReset();
                  }}
                  style={{
                    ...submitBtn,
                    background: "transparent",
                    border: "1px solid var(--border-glow)",
                    boxShadow: "none",
                    color: "var(--text-primary)",
                    marginTop: 24,
                  }}
                >
                  Analyze Another Image
                </button>
              </div>
            </SpotlightCard>
          </div>
        </ScrollReveal>
      )}
    </div>
  );
}
