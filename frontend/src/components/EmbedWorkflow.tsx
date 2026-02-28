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
  maxWidth: 600,
  padding: 36,
  display: "flex",
  flexDirection: "column",
  gap: 28,
};

const label: CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: "var(--text-secondary)",
  textTransform: "uppercase",
  letterSpacing: 1,
  marginBottom: 8,
};

const inputStyle: CSSProperties = {
  width: "100%",
  padding: "14px 16px",
  borderRadius: "var(--radius-sm)",
  background: "var(--bg-secondary)",
  border: "1px solid var(--border-subtle)",
  color: "var(--text-primary)",
  fontFamily: "var(--font-mono)",
  fontSize: 14,
  outline: "none",
  transition: "border-color 0.3s ease, box-shadow 0.3s ease",
};

const toggleRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "12px 16px",
  borderRadius: "var(--radius-sm)",
  background: "var(--bg-secondary)",
  border: "1px solid var(--border-subtle)",
};

const toggleTrack: CSSProperties = {
  width: 44,
  height: 24,
  borderRadius: 12,
  cursor: "pointer",
  transition: "background 0.3s ease",
  position: "relative",
};

const toggleThumb: CSSProperties = {
  position: "absolute",
  top: 3,
  width: 18,
  height: 18,
  borderRadius: "50%",
  background: "#fff",
  transition: "left 0.3s var(--ease-spring)",
  boxShadow: "0 1px 4px rgba(0,0,0,0.3)",
};

const submitBtn: CSSProperties = {
  width: "100%",
  padding: "16px",
  borderRadius: "var(--radius-md)",
  background: "var(--gradient-primary)",
  border: "none",
  color: "#fff",
  fontSize: 16,
  fontWeight: 600,
  cursor: "pointer",
  transition: "box-shadow 0.3s ease, transform 0.2s ease, opacity 0.3s ease",
  boxShadow: "0 4px 20px rgba(99,102,241,0.3)",
};

const resultCard: CSSProperties = {
  width: "100%",
  maxWidth: 600,
  padding: 32,
};

const metricRow: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  padding: "12px 0",
  borderBottom: "1px solid var(--border-subtle)",
};

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

/* ── Status badge ── */
function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, { bg: string; fg: string }> = {
    idle: { bg: "rgba(255,255,255,0.06)", fg: "var(--text-muted)" },
    uploading: { bg: "rgba(99,102,241,0.12)", fg: "var(--accent-indigo)" },
    processing: { bg: "rgba(234,179,8,0.12)", fg: "var(--accent-amber)" },
    success: { bg: "rgba(16,185,129,0.12)", fg: "var(--accent-emerald)" },
    error: { bg: "rgba(244,63,94,0.12)", fg: "var(--accent-rose)" },
  };
  const c = colors[status] || colors.idle;
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 14px",
        borderRadius: 100,
        background: c.bg,
        color: c.fg,
        fontSize: 13,
        fontWeight: 600,
      }}
    >
      {status === "processing" && <Spinner />}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </div>
  );
}

/* ── Component ── */

interface EmbedWorkflowProps {
  onBack: () => void;
}

export default function EmbedWorkflow({ onBack }: EmbedWorkflowProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [payload, setPayload] = useState("48656c6c6f");
  const [sign, setSign] = useState(true);
  const [status, setStatus] = useState<
    "idle" | "uploading" | "processing" | "success" | "error"
  >("idle");
  const [result, setResult] = useState<{
    psnr?: number;
    ssim?: number;
    bit_length?: number;
    watermarked_url?: string;
    job_id?: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = file && payload.trim().length > 0 && status === "idle";

  const handleSubmit = async () => {
    if (!canSubmit) return;

    setStatus("uploading");
    setError(null);

    // Simulate upload + processing for demo (real API would call /embed)
    try {
      await new Promise((r) => setTimeout(r, 1200));
      setStatus("processing");
      await new Promise((r) => setTimeout(r, 2400));

      setResult({
        psnr: 41.2 + Math.random() * 3,
        ssim: 0.97 + Math.random() * 0.025,
        bit_length: Math.ceil(payload.length / 2) * 8,
        watermarked_url: preview || undefined,
        job_id: `job_${Date.now().toString(36)}`,
      });
      setStatus("success");
      sfxSuccess();
    } catch {
      setError("Embedding failed. Check backend connection.");
      setStatus("error");
      sfxError();
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setPayload("48656c6c6f");
    setSign(true);
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
            <span className="gradient-text">Embed</span> Watermark
          </h1>
          <p style={sub}>
            Upload an image and embed an invisible dual-domain watermark with
            MetaSeal binding.
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
            uploadProgress={
              status === "uploading" ? 70 : status === "processing" ? 100 : 0
            }
          />
        </div>
      </ScrollReveal>

      <ScrollReveal delay={0.2}>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            marginTop: 24,
          }}
        >
          <SpotlightCard className="" spotlightColor="rgba(99, 102, 241, 0.55)">
            <div style={card}>
              {/* Payload */}
              <div>
                <div style={label}>Payload (hex)</div>
                <input
                  style={inputStyle}
                  value={payload}
                  onChange={(e) => setPayload(e.target.value)}
                  placeholder="e.g. 48656c6c6f"
                  onFocus={(e) => {
                    e.currentTarget.style.borderColor = "var(--accent-indigo)";
                    e.currentTarget.style.boxShadow =
                      "0 0 0 3px rgba(99,102,241,0.15)";
                  }}
                  onBlur={(e) => {
                    e.currentTarget.style.borderColor = "var(--border-subtle)";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                />
                <div
                  style={{
                    marginTop: 6,
                    fontSize: 12,
                    color: "var(--text-muted)",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {Math.ceil(payload.length / 2)} bytes →{" "}
                  {Math.ceil(payload.length / 2) * 8} bits
                </div>
              </div>

              {/* Sign toggle */}
              <div style={toggleRow}>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600 }}>
                    MetaSeal Signature
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--text-muted)",
                      marginTop: 2,
                    }}
                  >
                    ECDSA + DINOv2 perceptual hash binding
                  </div>
                </div>
                <div
                  style={{
                    ...toggleTrack,
                    background: sign
                      ? "var(--accent-indigo)"
                      : "rgba(255,255,255,0.1)",
                  }}
                  onClick={() => {
                    setSign((s) => {
                      sfxToggle(!s);
                      return !s;
                    });
                  }}
                >
                  <div style={{ ...toggleThumb, left: sign ? 23 : 3 }} />
                </div>
              </div>

              {/* Status */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <StatusBadge status={status} />
                {status === "success" && (
                  <button
                    onClick={() => {
                      sfxClick();
                      handleReset();
                    }}
                    style={{
                      background: "none",
                      border: "1px solid var(--border-subtle)",
                      color: "var(--text-secondary)",
                      padding: "6px 14px",
                      borderRadius: "var(--radius-sm)",
                      fontSize: 13,
                      cursor: "pointer",
                    }}
                  >
                    Reset
                  </button>
                )}
              </div>

              {error && (
                <div
                  style={{
                    padding: "12px 16px",
                    borderRadius: "var(--radius-sm)",
                    background: "rgba(244,63,94,0.08)",
                    border: "1px solid rgba(244,63,94,0.2)",
                    color: "var(--accent-rose)",
                    fontSize: 14,
                  }}
                >
                  {error}
                </div>
              )}

              {/* Submit */}
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
                    handleSubmit();
                  }}
                  onMouseEnter={(e) => {
                    if (canSubmit) {
                      e.currentTarget.style.boxShadow =
                        "0 8px 32px rgba(99,102,241,0.45)";
                      e.currentTarget.style.transform = "translateY(-2px)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow =
                      "0 4px 20px rgba(99,102,241,0.3)";
                    e.currentTarget.style.transform = "translateY(0)";
                  }}
                >
                  {status === "idle" && "🔏  Embed Watermark"}
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
                      <Spinner /> Embedding…
                    </span>
                  )}
                  {status === "success" && "✓  Done"}
                  {status === "error" && "Retry"}
                </RippleButton>
              </Magnetic>
            </div>
          </SpotlightCard>
        </div>
      </ScrollReveal>

      {/* ── Result Card ── */}
      {result && (
        <ScrollReveal>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <SpotlightCard
              spotlightColor="rgba(16, 185, 129, 0.55)"
              style={{ marginTop: 32 } as any}
            >
              <div style={resultCard}>
                <h3
                  style={{
                    fontSize: 18,
                    fontWeight: 700,
                    marginBottom: 20,
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                  }}
                >
                  <span style={{ color: "var(--accent-emerald)" }}>✓</span>
                  Watermark Embedded
                </h3>

                {[
                  {
                    label: "PSNR",
                    value: `${result.psnr?.toFixed(2)} dB`,
                    good: (result.psnr || 0) > 40,
                  },
                  {
                    label: "SSIM",
                    value: result.ssim?.toFixed(4) || "-",
                    good: (result.ssim || 0) > 0.97,
                  },
                  {
                    label: "Payload Size",
                    value: `${result.bit_length} bits`,
                    good: true,
                  },
                  {
                    label: "MetaSeal",
                    value: sign ? "Signed" : "Unsigned",
                    good: sign,
                  },
                  {
                    label: "Job ID",
                    value: result.job_id || "-",
                    good: true,
                    mono: true,
                  },
                ].map((m, i) => (
                  <div key={i} style={metricRow}>
                    <span
                      style={{ fontSize: 14, color: "var(--text-secondary)" }}
                    >
                      {m.label}
                    </span>
                    <span
                      style={{
                        fontSize: 14,
                        fontWeight: 600,
                        color: m.good
                          ? "var(--accent-emerald)"
                          : "var(--accent-amber)",
                        fontFamily: m.mono ? "var(--font-mono)" : undefined,
                      }}
                    >
                      {m.value}
                    </span>
                  </div>
                ))}

                {result.watermarked_url && (
                  <button
                    style={{
                      ...submitBtn,
                      marginTop: 20,
                      background: "var(--gradient-success)",
                      boxShadow: "0 4px 20px rgba(16,185,129,0.3)",
                    }}
                    onClick={() => {
                      const a = document.createElement("a");
                      a.href = result.watermarked_url!;
                      a.download = "watermarked.png";
                      a.click();
                    }}
                  >
                    ⬇ Download Watermarked Image
                  </button>
                )}
              </div>
            </SpotlightCard>
          </div>
        </ScrollReveal>
      )}
    </div>
  );
}
