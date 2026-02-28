import { useCallback, useState, type CSSProperties } from "react";
import { useDropzone } from "react-dropzone";
import { sfxClick, sfxDrop } from "../utils/sfx";

/* ── Styles ── */

const wrapper: CSSProperties = {
  width: "100%",
  maxWidth: 540,
};

const dropzone: CSSProperties = {
  position: "relative",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  padding: "56px 32px",
  borderRadius: "var(--radius-lg)",
  border: "2px dashed var(--border-subtle)",
  background: "var(--bg-card)",
  cursor: "pointer",
  transition:
    "border-color 0.3s ease, background 0.3s ease, box-shadow 0.3s ease",
  overflow: "hidden",
};

const dropzoneActive: CSSProperties = {
  ...dropzone,
  borderColor: "var(--accent-indigo)",
  background: "rgba(99,102,241,0.06)",
  boxShadow: "var(--shadow-glow)",
};

const iconCircle: CSSProperties = {
  width: 72,
  height: 72,
  borderRadius: "50%",
  background: "rgba(99,102,241,0.1)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 32,
  marginBottom: 20,
  transition: "transform 0.4s var(--ease-spring)",
};

const previewContainer: CSSProperties = {
  position: "relative",
  width: "100%",
  maxWidth: 540,
  borderRadius: "var(--radius-lg)",
  overflow: "hidden",
  border: "1px solid var(--border-subtle)",
  background: "var(--bg-card)",
};

const previewImg: CSSProperties = {
  width: "100%",
  display: "block",
  objectFit: "cover",
  maxHeight: 360,
};

const previewOverlay: CSSProperties = {
  position: "absolute",
  bottom: 0,
  left: 0,
  right: 0,
  padding: "16px 20px",
  background: "linear-gradient(transparent, rgba(0,0,0,0.85))",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
};

const removeBtn: CSSProperties = {
  padding: "6px 14px",
  borderRadius: "var(--radius-sm)",
  fontSize: 13,
  fontWeight: 500,
  color: "var(--accent-rose)",
  background: "rgba(244,63,94,0.1)",
  border: "1px solid rgba(244,63,94,0.25)",
  cursor: "pointer",
  transition: "background 0.25s ease",
};

const progressBar: CSSProperties = {
  position: "absolute",
  bottom: 0,
  left: 0,
  height: 3,
  background: "var(--gradient-primary)",
  borderRadius: "0 2px 2px 0",
  transition: "width 0.3s ease",
};

/* ── Component ── */

interface ImageUploaderProps {
  onFileSelected: (file: File, preview: string) => void;
  onClear: () => void;
  preview: string | null;
  uploading?: boolean;
  uploadProgress?: number;
}

export default function ImageUploader({
  onFileSelected,
  onClear,
  preview,
  uploading = false,
  uploadProgress = 0,
}: ImageUploaderProps) {
  const [dragHover, setDragHover] = useState(false);

  const onDrop = useCallback(
    (accepted: File[]) => {
      setDragHover(false);
      if (accepted.length === 0) return;
      const file = accepted[0];
      const url = URL.createObjectURL(file);
      sfxDrop();
      onFileSelected(file, url);
    },
    [onFileSelected],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp", ".bmp"] },
    maxFiles: 1,
    multiple: false,
    onDragEnter: () => setDragHover(true),
    onDragLeave: () => setDragHover(false),
  });

  const active = isDragActive || dragHover;

  /* ── Already has image ── */
  if (preview) {
    return (
      <div style={previewContainer}>
        <img src={preview} alt="Selected" style={previewImg} />
        <div style={previewOverlay}>
          <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
            {uploading ? "Uploading…" : "Ready"}
          </span>
          {!uploading && (
            <button
              style={removeBtn}
              onClick={() => {
                sfxClick();
                onClear();
              }}
            >
              Remove
            </button>
          )}
        </div>
        {uploading && (
          <div style={{ ...progressBar, width: `${uploadProgress}%` }} />
        )}
      </div>
    );
  }

  /* ── Dropzone ── */
  return (
    <div style={wrapper}>
      <div {...getRootProps()} style={active ? dropzoneActive : dropzone}>
        <input {...getInputProps()} />

        <div
          style={{
            ...iconCircle,
            transform: active
              ? "scale(1.15) rotate(8deg)"
              : "scale(1) rotate(0)",
          }}
        >
          📷
        </div>

        <p
          style={{
            fontSize: 16,
            fontWeight: 600,
            color: "var(--text-primary)",
            marginBottom: 8,
          }}
        >
          {active ? "Drop it!" : "Drop an image here"}
        </p>
        <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
          or{" "}
          <span style={{ color: "var(--accent-indigo)", fontWeight: 500 }}>
            browse files
          </span>
          &nbsp; · PNG, JPEG, WebP
        </p>

        {/* Animated border glow on drag */}
        {active && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              borderRadius: "var(--radius-lg)",
              boxShadow: "inset 0 0 40px rgba(99,102,241,0.1)",
              pointerEvents: "none",
            }}
          />
        )}
      </div>
    </div>
  );
}
