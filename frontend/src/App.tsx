import { useState } from "react";
import ClickSpark from "./components/ClickSpark";
import DetectWorkflow from "./components/DetectWorkflow";
import { AnimatedBackground, GrainOverlay } from "./components/Effects";
import EmbedWorkflow from "./components/EmbedWorkflow";
import Hero from "./components/Hero";
import {
  CursorTrail,
  FloatingParticles,
  PageTransition,
} from "./components/Interactive";
import Navbar from "./components/Navbar";
import { sfxNavigate } from "./utils/sfx";

type Page = "home" | "embed" | "detect";

export default function App() {
  const [page, setPage] = useState<Page>("home");

  const navigate = (p: Page | "embed" | "detect") => {
    sfxNavigate();
    setPage(p as Page);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <ClickSpark>
      {/* Global FX layers */}
      <CursorTrail />
      <FloatingParticles count={35} />
      <AnimatedBackground />
      <GrainOverlay />

      {/* Nav */}
      <Navbar currentPage={page} onNavigate={navigate} />

      {/* Pages with transition */}
      <PageTransition pageKey={page}>
        {page === "home" && <Hero onNavigate={navigate} />}
        {page === "embed" && <EmbedWorkflow onBack={() => navigate("home")} />}
        {page === "detect" && (
          <DetectWorkflow onBack={() => navigate("home")} />
        )}
      </PageTransition>

      {/* Footer */}
      <footer
        style={{
          position: "relative",
          zIndex: 1,
          textAlign: "center",
          padding: "40px 24px 32px",
          borderTop: "1px solid var(--border-subtle)",
          fontSize: 13,
          color: "var(--text-muted)",
        }}
      >
        <span className="gradient-text" style={{ fontWeight: 700 }}>
          WatermarkAI
        </span>{" "}
        — AI-Powered Watermark Recovery System · AMD Hackathon 2025
      </footer>
    </ClickSpark>
  );
}
