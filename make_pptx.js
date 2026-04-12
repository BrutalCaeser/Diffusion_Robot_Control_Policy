/**
 * make_pptx.js — Diffusion Policy for Robot Manipulation  (v2 — visual redesign)
 *
 * Design principles:
 *  • ONE message per slide, stated as a headline
 *  • Lead with visuals (embedded plots, drawn diagrams) not bullet walls
 *  • Generous whitespace — breathe between elements
 *  • Text is SHORT and impactful; diagrams explain the complex parts
 *  • Story arc: problem → insight → solution → results → why it works
 *
 * Run:  node make_pptx.js
 */

"use strict";

const pptxgen = require("pptxgenjs");
const path = require("path");

const PLOTS = path.resolve(__dirname, "plots");

// ─── Palette ──────────────────────────────────────────────────────────────────
const C = {
  NAVY:       "1E2761",  // dominant dark
  NAVY2:      "2D3580",  // slightly lighter navy
  ICE:        "CADCFC",  // ice blue
  WHITE:      "FFFFFF",
  GOLD:       "F9C74F",  // highlight / success
  TEAL:       "06D6A0",  // FM / best result
  CORAL:      "EF476F",  // BC failure / red
  LIGHT:      "F0F4FF",  // content slide bg
  MEDIUM:     "E4ECF9",  // slightly deeper blue-white
  DARK_TXT:   "1A1A2E",
  MUTED:      "6B7CAB",
  CARD:       "FFFFFF",
  BORDER:     "BDD0F0",
};

const FONT  = "Calibri";
const MONO  = "Consolas";

const pres = new pptxgen();
pres.layout  = "LAYOUT_16x9";   // 10 × 5.625 inches
pres.author  = "Yashvardhan Gupta, Vineeth Sakhamuru, Sai Krishna Reddy Maligireddy";
pres.title   = "Diffusion Policy for Robot Manipulation";

// ─── Slide factories ──────────────────────────────────────────────────────────
function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: C.NAVY };
  return s;
}
function lightSlide() {
  const s = pres.addSlide();
  s.background = { color: C.LIGHT };
  return s;
}

// Slim navy header bar + slide title
function header(s, title, sub) {
  const h = sub ? 0.82 : 0.68;
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h, fill: { color: C.NAVY }, line: { color: C.NAVY } });
  s.addText(title, { x: 0.4, y: 0, w: 9.2, h, fontSize: 21, bold: true, color: C.WHITE, fontFace: FONT, valign: "middle", margin: 0 });
  if (sub) {
    // sub lives inside the bar — reduce font
    s.addText(title, { x: 0.4, y: 0, w: 9.2, h: 0.5, fontSize: 21, bold: true, color: C.WHITE, fontFace: FONT, valign: "middle", margin: 0 });
  }
}

const mkShadow = () => ({ type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.1 });

// ─── Helper: draw a white card ────────────────────────────────────────────────
function wCard(s, { x, y, w, h, accent = C.NAVY2 }) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.CARD }, line: { color: C.BORDER, width: 0.5 }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h, fill: { color: accent }, line: { color: accent } });
}

// ─── Helper: draw a solid-color label pill ────────────────────────────────────
function pill(s, text, { x, y, w = 1.4, h = 0.34, bg = C.NAVY, fg = C.WHITE, fontSize = 10 } = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: bg }, line: { color: bg }, rectRadius: 0.05 });
  s.addText(text, { x, y, w, h, fontSize, bold: true, color: fg, fontFace: FONT, align: "center", valign: "middle", margin: 0 });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = darkSlide();

  // Subtle geometric accent strip (right side)
  for (let i = 0; i < 8; i++) {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 8.5 + i * 0.18, y: 0, w: 0.12, h: 5.625,
      fill: { color: "192260", transparency: 40 + i * 7 }, line: { color: "192260" },
    });
  }

  // Gold top accent
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.055, fill: { color: C.GOLD }, line: { color: C.GOLD } });

  // Main title
  s.addText("Diffusion Policy", { x: 0.55, y: 0.5, w: 7.5, h: 1.15, fontSize: 46, bold: true, color: C.WHITE, fontFace: FONT });
  s.addText("for Robot Manipulation", { x: 0.55, y: 1.55, w: 7.5, h: 0.85, fontSize: 28, color: C.ICE, fontFace: FONT });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 2.55, w: 2.2, h: 0.055, fill: { color: C.GOLD }, line: { color: C.GOLD } });

  s.addText("Teaching a Robot to Push using Generative Models", {
    x: 0.55, y: 2.72, w: 7.8, h: 0.42,
    fontSize: 15, color: "8FA8D8", fontFace: FONT, italic: true,
  });
  s.addText("ML 6140 — Machine Learning  |  Northeastern University  |  April 2026", {
    x: 0.55, y: 3.22, w: 7.8, h: 0.38,
    fontSize: 12, color: "6680B0", fontFace: FONT,
  });
  s.addText("Yashvardhan Gupta  ·  Vineeth Sakhamuru  ·  Sai Krishna Reddy Maligireddy", {
    x: 0.55, y: 3.68, w: 7.8, h: 0.38,
    fontSize: 13, bold: true, color: C.ICE, fontFace: FONT,
  });

  // Result mini-panel (bottom right)
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 4.25, w: 8.8, h: 1.05, fill: { color: "141D55" }, line: { color: "141D55" } });
  const cols = [
    { m: "BC Baseline",  v: "4%",   c: C.CORAL },
    { m: "DDPM",         v: "80%",  c: C.MUTED },
    { m: "DDIM",         v: "92%",  c: C.ICE   },
    { m: "Flow Matching",v: "98%",  c: C.GOLD  },
  ];
  cols.forEach((c, i) => {
    const x = 0.85 + i * 2.15;
    // bar
    const barH = (parseFloat(c.v) / 100) * 0.55;
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.45, y: 4.6 - barH + 0.55, w: 0.35, h: barH,
      fill: { color: c.c }, line: { color: c.c },
    });
    s.addText(c.v, { x, y: 4.62, w: 1.3, h: 0.28, fontSize: 11, bold: true, color: c.c, fontFace: FONT, align: "center" });
    s.addText(c.m, { x, y: 4.9, w: 1.3, h: 0.32, fontSize: 9, color: "6680B0", fontFace: FONT, align: "center" });
  });

  // Bottom
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.28, w: 10, h: 0.345, fill: { color: "0E1340" }, line: { color: "0E1340" } });
  s.addText("github.com/BrutalCaeser/Diffusion_Robot_Control_Policy  |  120 unit tests passing", {
    x: 0.3, y: 5.28, w: 9.4, h: 0.345, fontSize: 9, color: "4A5A90", fontFace: FONT, align: "center", valign: "middle",
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — THE TASK: PUSHT
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "The Task: Push a T-Shaped Block onto a Target");

  // ── Left: drawn PushT scene ─────────────────────────────────────────────────
  // Environment boundary
  s.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 0.85, w: 4.2, h: 4.35, fill: { color: C.MEDIUM }, line: { color: C.BORDER, width: 1 } });

  // Target zone (gray filled rectangle — where T should go)
  s.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 2.3, w: 2.0, h: 1.5, fill: { color: "B8C8D8", transparency: 30 }, line: { color: "8AAABB", width: 1.5 } });
  s.addText("TARGET", { x: 1.5, y: 2.9, w: 2.0, h: 0.4, fontSize: 10, color: "5A7A8A", fontFace: FONT, align: "center", bold: true });

  // T-block (two rectangles forming a T)
  s.addShape(pres.shapes.RECTANGLE, { x: 1.55, y: 1.4, w: 1.9, h: 0.45, fill: { color: "5B7CBA" }, line: { color: "3A5A99", width: 1 } }); // horizontal bar
  s.addShape(pres.shapes.RECTANGLE, { x: 2.2, y: 1.4, w: 0.6, h: 0.85, fill: { color: "5B7CBA" }, line: { color: "3A5A99", width: 1 } }); // vertical bar
  s.addText("T-block", { x: 1.5, y: 1.12, w: 1.9, h: 0.28, fontSize: 9, color: "3A5A99", fontFace: FONT, align: "center" });

  // Robot agent (circle)
  s.addShape(pres.shapes.OVAL, { x: 0.7, y: 2.9, w: 0.55, h: 0.55, fill: { color: C.CORAL }, line: { color: "C03050", width: 2 } });
  s.addText("🤖", { x: 0.7, y: 2.9, w: 0.55, h: 0.55, fontSize: 14, align: "center", valign: "middle" });

  // Push arrow (agent → block)
  s.addShape(pres.shapes.LINE, { x: 1.28, y: 3.18, w: 0.35, h: -0.9, line: { color: C.CORAL, width: 2.5 } });

  // Arrow tip
  s.addText("→", { x: 1.05, y: 2.15, w: 0.35, h: 0.3, fontSize: 14, color: C.CORAL, fontFace: FONT });

  // Labels
  s.addText("Agent (end-effector)", { x: 0.3, y: 3.55, w: 1.3, h: 0.55, fontSize: 9, color: C.CORAL, fontFace: FONT, align: "center" });
  s.addText("Push!", { x: 1.0, y: 2.55, w: 0.5, h: 0.28, fontSize: 9, color: C.CORAL, fontFace: FONT, bold: true });

  // Success indicator
  s.addShape(pres.shapes.OVAL, { x: 3.55, y: 0.92, w: 0.72, h: 0.72, fill: { color: "06D6A0" }, line: { color: "04A87D" } });
  s.addText("✓", { x: 3.55, y: 0.92, w: 0.72, h: 0.72, fontSize: 20, color: C.WHITE, fontFace: FONT, align: "center", valign: "middle", margin: 0 });
  s.addText("≥90% overlap", { x: 3.35, y: 1.7, w: 1.1, h: 0.36, fontSize: 8.5, color: "04A87D", fontFace: FONT, align: "center" });

  // ── Right: 5 quick facts ────────────────────────────────────────────────────
  const facts = [
    { icon: "📍", label: "Agent",    text: "Circular end-effector, moves in 2D" },
    { icon: "📦", label: "Goal",     text: "Push T-block onto gray target region" },
    { icon: "✅", label: "Success",  text: "Block covers target by ≥ 90%" },
    { icon: "📊", label: "State",    text: "5 numbers: agent (x,y) + block (x,y,θ)" },
    { icon: "🎮", label: "Actions",  text: "2 numbers: velocity (vₓ, vy)" },
    { icon: "💾", label: "Data",     text: "206 expert demos · 25,650 steps" },
  ];

  facts.forEach((f, i) => {
    const y = 0.92 + i * 0.72;
    s.addShape(pres.shapes.OVAL, { x: 4.82, y: y + 0.08, w: 0.42, h: 0.42, fill: { color: C.NAVY }, line: { color: C.NAVY } });
    s.addText(f.icon, { x: 4.82, y: y + 0.08, w: 0.42, h: 0.42, fontSize: 13, align: "center", valign: "middle" });
    s.addText(f.label + ":", { x: 5.35, y, w: 1.0, h: 0.55, fontSize: 12, bold: true, color: C.NAVY, fontFace: FONT, valign: "middle" });
    s.addText(f.text,         { x: 6.4,  y, w: 3.35, h: 0.55, fontSize: 11.5, color: C.DARK_TXT, fontFace: FONT, valign: "middle" });
  });

  // Bottom strip
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.17, w: 10, h: 0.455, fill: { color: C.NAVY }, line: { color: C.NAVY } });
  s.addText("Receding-horizon control: predict 16 future actions → execute first 8 → replan with fresh observations", {
    x: 0.3, y: 5.17, w: 9.4, h: 0.455, fontSize: 11, color: C.ICE, fontFace: FONT, align: "center", valign: "middle", italic: true,
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — WHY BC FAILS (embed multimodal plot + explanation)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "Why Standard ML Fails: Expert Behavior is Multi-Modal");

  // Embed the actual multimodal motivation plot
  s.addImage({
    path: path.join(PLOTS, "multimodal_motivation.png"),
    x: 0.25, y: 0.78, w: 6.5, h: 3.85,
    sizing: { type: "contain", w: 6.5, h: 3.85 },
  });

  // Right panel explanation
  wCard(s, { x: 6.95, y: 0.82, w: 2.85, h: 1.7, accent: C.TEAL });
  s.addText("Expert strategies", { x: 7.1, y: 0.88, w: 2.55, h: 0.38, fontSize: 13, bold: true, color: "047857", fontFace: FONT });
  s.addText("Each colored line = one expert's\napproach to the task.\nMultiple valid strategies exist!", {
    x: 7.1, y: 1.28, w: 2.55, h: 0.95, fontSize: 11, color: C.DARK_TXT, fontFace: FONT,
  });

  wCard(s, { x: 6.95, y: 2.65, w: 2.85, h: 1.7, accent: C.CORAL });
  s.addText("BC averages modes", { x: 7.1, y: 2.71, w: 2.55, h: 0.38, fontSize: 13, bold: true, color: "BE123C", fontFace: FONT });
  s.addText("The action space is bimodal.\nBC predicts the middle →\nrobot gets stuck.", {
    x: 7.1, y: 3.11, w: 2.55, h: 0.95, fontSize: 11, color: C.DARK_TXT, fontFace: FONT,
  });

  // Bottom callout
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.75, w: 10, h: 0.875, fill: { color: C.NAVY }, line: { color: C.NAVY } });
  s.addText([
    { text: "BC result: ", options: { color: C.ICE } },
    { text: "4% success", options: { bold: true, color: C.CORAL } },
    { text: "   |   Averaging valid strategies produces an ", options: { color: C.ICE } },
    { text: "INVALID strategy", options: { bold: true, color: C.GOLD } },
  ], {
    x: 0.3, y: 4.75, w: 9.4, h: 0.875, fontSize: 15, fontFace: FONT, align: "center", valign: "middle",
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — THE DIFFUSION IDEA (intuitive visual)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "The Diffusion Idea: Iteratively Remove Noise to Get an Action");

  // 3 panels: Noise → Partial → Clean
  const panels = [
    { title: "Start:\nPure Noise", sub: "Random action sequence", num: "1", numC: C.CORAL },
    { title: "Iterate:\nDenoise", sub: "U-Net removes noise\n10 times", num: "2", numC: C.GOLD },
    { title: "Result:\nClean Action", sub: "Valid robot motion\nemerges!", num: "3", numC: C.TEAL },
  ];

  panels.forEach((p, i) => {
    const x = 0.2 + i * 3.27;

    // Panel background
    const bg = i === 0 ? "F5F0FF" : i === 1 ? "FFF8E7" : "EDFDF6";
    const borderC = i === 0 ? "C4B5FD" : i === 1 ? "FDE68A" : "6EE7B7";
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.82, w: 3.0, h: 4.45, fill: { color: bg }, line: { color: borderC, width: 1 }, shadow: mkShadow() });

    // Step badge
    s.addShape(pres.shapes.OVAL, { x: x + 1.2, y: 0.72, w: 0.6, h: 0.6, fill: { color: p.numC }, line: { color: p.numC } });
    s.addText(p.num, { x: x + 1.2, y: 0.72, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: C.WHITE, fontFace: FONT, align: "center", valign: "middle", margin: 0 });

    // Visual area (simulated)
    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.15, y: 0.95, w: 2.7, h: 2.2, fill: { color: "1E1E3A" }, line: { color: "1E1E3A" } });

    // Visual content depends on panel
    if (i === 0) {
      // Random scattered dots (noise)
      const noisePositions = [
        [0.25, 1.1], [0.9, 1.3], [1.6, 1.05], [2.2, 1.25], [0.4, 1.7],
        [1.1, 1.8], [1.8, 1.6], [2.4, 1.85], [0.2, 2.3], [0.8, 2.5],
        [1.5, 2.2], [2.1, 2.4], [2.6, 2.1], [0.5, 2.9], [1.3, 3.0],
        [2.0, 2.85], [0.7, 1.45], [1.7, 2.05], [2.3, 2.65], [0.35, 2.0],
      ];
      noisePositions.forEach(([dx, dy]) => {
        const cc = ["A78BFA", "F472B6", "60A5FA", "34D399"][Math.floor(dx * dy) % 4];
        s.addShape(pres.shapes.OVAL, { x: x + 0.15 + dx * 0.9, y: 0.95 + dy * 0.75, w: 0.12, h: 0.12, fill: { color: cc }, line: { color: cc } });
      });
    } else if (i === 1) {
      // Semi-organized dots with slight directionality
      const partialPos = [
        [0.3, 1.0], [0.9, 0.9], [1.5, 1.05], [2.1, 0.95], [2.55, 1.0],
        [0.35, 1.55], [0.95, 1.5], [1.55, 1.6], [2.05, 1.5], [2.5, 1.55],
        [0.4, 2.05], [1.0, 2.0], [1.6, 2.1], [2.1, 2.0], [2.55, 2.1],
      ];
      partialPos.forEach(([dx, dy]) => {
        s.addShape(pres.shapes.OVAL, { x: x + 0.15 + dx * 0.9, y: 0.95 + dy * 0.75, w: 0.15, h: 0.1, fill: { color: "FDE68A" }, line: { color: "F59E0B" } });
      });
    } else {
      // Clean arrows pointing right (organized action sequence)
      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 6; col++) {
          s.addText("→", {
            x: x + 0.25 + col * 0.4, y: 1.05 + row * 0.5, w: 0.38, h: 0.42,
            fontSize: 15, color: C.TEAL, fontFace: FONT, align: "center",
          });
        }
      }
    }

    // Title
    s.addText(p.title, {
      x: x + 0.1, y: 3.22, w: 2.8, h: 0.75,
      fontSize: 15, bold: true, color: C.DARK_TXT, fontFace: FONT, align: "center",
    });
    s.addText(p.sub, {
      x: x + 0.1, y: 3.95, w: 2.8, h: 0.75,
      fontSize: 11, color: C.MUTED, fontFace: FONT, align: "center",
    });

    // Arrow between panels
    if (i < 2) {
      s.addText("→", { x: x + 3.07, y: 2.3, w: 0.2, h: 0.4, fontSize: 22, color: C.NAVY, fontFace: FONT, align: "center" });
    }
  });

  // Key insight
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.07, w: 10, h: 0.555, fill: { color: C.NAVY }, line: { color: C.NAVY } });
  s.addText([
    { text: "Different starting noise  →  Different valid action  |  ", options: { bold: true, color: C.GOLD } },
    { text: "This is how diffusion handles multi-modal expert behavior!", options: { color: C.ICE } },
  ], { x: 0.3, y: 5.07, w: 9.4, h: 0.555, fontSize: 12.5, fontFace: FONT, align: "center", valign: "middle" });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — DATASET
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "Training Data: 206 Expert Demonstrations");

  // 3 big stat callouts
  const stats = [
    { val: "206",    label: "Expert Episodes",    sub: "Human demos from Columbia", color: C.NAVY,  bg: "EEF2FF", border: "BDD0F0" },
    { val: "25,650", label: "Total Timesteps",     sub: "Observations + actions",   color: "7E22CE", bg: "F5F3FF", border: "C4B5FD" },
    { val: "~25K",   label: "Training Windows",    sub: "After sliding-window split", color: "047857", bg: "ECFDF5", border: "6EE7B7" },
  ];
  stats.forEach((s2, i) => {
    const x = 0.3 + i * 3.17;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.85, w: 2.9, h: 2.15, fill: { color: s2.bg }, line: { color: s2.border, width: 1 }, shadow: mkShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.85, w: 2.9, h: 0.06, fill: { color: s2.color }, line: { color: s2.color } });
    s.addText(s2.val,   { x: x + 0.1, y: 0.95, w: 2.7, h: 1.0, fontSize: 46, bold: true, color: s2.color, fontFace: FONT, align: "center", valign: "middle" });
    s.addText(s2.label, { x: x + 0.1, y: 1.85, w: 2.7, h: 0.42, fontSize: 13, bold: true, color: C.DARK_TXT, fontFace: FONT, align: "center" });
    s.addText(s2.sub,   { x: x + 0.1, y: 2.25, w: 2.7, h: 0.6, fontSize: 10, color: C.MUTED, fontFace: FONT, align: "center" });
  });

  // Embed dataset summary plot
  s.addImage({
    path: path.join(PLOTS, "dataset", "dataset_summary.png"),
    x: 0.3, y: 3.1, w: 5.0, h: 2.2,
    sizing: { type: "contain", w: 5.0, h: 2.2 },
  });

  // Sliding window explanation on the right
  wCard(s, { x: 5.6, y: 3.1, w: 4.1, h: 2.2, accent: C.NAVY });
  s.addText("How training windows work:", { x: 5.8, y: 3.18, w: 3.7, h: 0.4, fontSize: 12, bold: true, color: C.NAVY, fontFace: FONT });

  const steps2 = [
    "Observe last 2 timesteps  (T_obs = 2)",
    "Predict next 16 actions  (T_pred = 16)",
    "Execute first 8, then replan  (T_action = 8)",
    "Repeat until task done or timeout",
  ];
  steps2.forEach((line, i) => {
    pill(s, String(i + 1), { x: 5.8, y: 3.65 + i * 0.4, w: 0.28, h: 0.28, bg: C.NAVY, fontSize: 9 });
    s.addText(line, { x: 6.15, y: 3.65 + i * 0.4, w: 3.35, h: 0.35, fontSize: 10.5, color: C.DARK_TXT, fontFace: FONT, valign: "middle" });
  });
  s.addText("All values normalised to [−1, 1] before training", {
    x: 5.8, y: 5.05, w: 3.7, h: 0.28, fontSize: 10, color: C.MUTED, fontFace: FONT, italic: true,
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — ARCHITECTURE (visual U-Net hourglass)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "Model Architecture: Conditional 1D Temporal U-Net");

  // Drawn U-Net hourglass
  const cx = 3.5;  // center x of hourglass
  const levels = [
    { w: 2.8, label: "In: (16, 2)", ch: "2 ch",    y: 0.85 },
    { w: 2.0, label: "Down 1",      ch: "256 ch",  y: 1.55 },
    { w: 1.4, label: "Down 2",      ch: "512 ch",  y: 2.25 },
    { w: 0.8, label: "Bottleneck",  ch: "1024 ch", y: 2.95 },
    { w: 1.4, label: "Up 2",        ch: "512 ch",  y: 3.65 },
    { w: 2.0, label: "Up 1",        ch: "256 ch",  y: 4.35 },
    { w: 2.8, label: "Out: (16, 2)",ch: "2 ch",    y: 5.05 },
  ];

  const boxH = 0.52;
  const colors = ["EEF2FF","E0E7FF","C7D2FE","818CF8","C7D2FE","E0E7FF","D1FAE5"];
  const txColors = [C.NAVY, C.NAVY, C.NAVY2, C.WHITE, C.NAVY2, C.NAVY, "047857"];

  levels.forEach((lv, i) => {
    if (i >= levels.length - 1) return; // skip last in this loop — draw separately
    const x = cx - lv.w / 2;
    // Draw box
    s.addShape(pres.shapes.RECTANGLE, { x, y: lv.y, w: lv.w, h: boxH, fill: { color: colors[i] }, line: { color: C.BORDER, width: 0.75 } });
    s.addText(lv.label + "   " + lv.ch, { x, y: lv.y, w: lv.w, h: boxH, fontSize: 10, color: txColors[i], fontFace: FONT, align: "center", valign: "middle" });
    // Arrow down
    if (i < 5) {
      s.addText("↓", { x: cx - 0.15, y: lv.y + boxH, w: 0.3, h: 0.25, fontSize: 14, color: C.MUTED, fontFace: FONT, align: "center" });
    }
    // Skip connections (encoder→decoder)
    if (i === 1 || i === 2) {
      const mirrorY = levels[7 - i - 1]?.y ?? (lv.y + 2.8);
      s.addShape(pres.shapes.LINE, {
        x: cx + lv.w / 2, y: lv.y + boxH / 2,
        w: 0.6, h: mirrorY - lv.y,
        line: { color: C.GOLD, width: 1.5, dashType: "dash" },
      });
    }
  });

  // Last box
  const last = levels[levels.length - 1];
  const x = cx - last.w / 2;
  s.addShape(pres.shapes.RECTANGLE, { x, y: last.y, w: last.w, h: boxH, fill: { color: colors[colors.length - 1] }, line: { color: C.BORDER, width: 0.75 } });
  s.addText(last.label + "   " + last.ch, { x, y: last.y, w: last.w, h: boxH, fontSize: 10, color: txColors[txColors.length - 1], fontFace: FONT, align: "center", valign: "middle" });

  // Conditioning input (right side)
  wCard(s, { x: 6.3, y: 0.85, w: 3.45, h: 2.3, accent: C.GOLD });
  s.addText("Conditioning Inputs", { x: 6.5, y: 0.92, w: 3.0, h: 0.4, fontSize: 13, bold: true, color: "854D0E", fontFace: FONT });
  const inputs = [
    "Observation history (2 × 5 = 10 numbers)",
    "Diffusion timestep k (noise level)",
    "→ embed each → concat → 512-d vector",
    "→ FiLM: scale & shift every ResBlock",
  ];
  inputs.forEach((line, i) => {
    s.addText(line, { x: 6.5, y: 1.38 + i * 0.37, w: 3.1, h: 0.35, fontSize: 11, color: i >= 2 ? "854D0E" : C.DARK_TXT, fontFace: FONT, bold: i >= 2, italic: i >= 2 });
  });

  wCard(s, { x: 6.3, y: 3.35, w: 3.45, h: 1.95, accent: C.TEAL });
  s.addText("Output", { x: 6.5, y: 3.42, w: 3.0, h: 0.4, fontSize: 13, bold: true, color: "065F46", fontFace: FONT });
  s.addText("DDPM:  predict noise  ε  → shape (16 × 2)", { x: 6.5, y: 3.88, w: 3.1, h: 0.38, fontSize: 11, color: C.DARK_TXT, fontFace: FONT });
  s.addText("FM:    predict velocity  v = ε − a₀", { x: 6.5, y: 4.28, w: 3.1, h: 0.38, fontSize: 11, color: C.DARK_TXT, fontFace: FONT });
  s.addText("68.95 M params  |  EMA decay = 0.995", { x: 6.5, y: 4.78, w: 3.1, h: 0.38, fontSize: 10, color: C.MUTED, fontFace: FONT, italic: true });

  // Skip connection legend
  s.addShape(pres.shapes.LINE, { x: 0.3, y: 5.28, w: 0.5, h: 0, line: { color: C.GOLD, width: 1.5, dashType: "dash" } });
  s.addText("Skip connections (encoder → decoder)", { x: 0.9, y: 5.18, w: 3.0, h: 0.35, fontSize: 10, color: "854D0E", fontFace: FONT, italic: true });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — THREE INFERENCE METHODS (visual step comparison)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "Three Ways to Run the Model at Inference Time");

  // Method columns
  const methods = [
    {
      name: "DDPM",  steps: 100, time: "84 ms", success: "80%",
      color: "64748B", bg: "F8FAFC", bar: "94A3B8",
      desc: "Full stochastic reverse chain.\nSlowly denoises over 100 steps.",
      dotColor: "94A3B8", dotCount: 10,
    },
    {
      name: "DDIM",  steps: 10, time: "10 ms", success: "92%",
      color: C.NAVY2, bg: "EEF2FF", bar: C.NAVY2,
      desc: "Skips most timesteps algebraically.\n8× faster — same trained model.",
      dotColor: C.NAVY2, dotCount: 3,
    },
    {
      name: "Flow Matching ★", steps: 10, time: "8.9 ms", success: "98%",
      color: "065F46", bg: "ECFDF5", bar: C.TEAL,
      desc: "Straight-line ODE from noise to action.\nBest conditioned — best results.",
      dotColor: C.TEAL, dotCount: 3,
    },
  ];

  methods.forEach((m, i) => {
    const x = 0.2 + i * 3.25;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.82, w: 3.0, h: 4.6, fill: { color: m.bg }, line: { color: C.BORDER, width: 0.75 }, shadow: mkShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.82, w: 3.0, h: 0.06, fill: { color: m.bar }, line: { color: m.bar } });

    // Name
    s.addText(m.name, { x: x + 0.1, y: 0.95, w: 2.8, h: 0.55, fontSize: 18, bold: true, color: m.color, fontFace: FONT, align: "center" });

    // Steps visualization (dots = steps)
    s.addText("Denoising steps:", { x: x + 0.1, y: 1.58, w: 2.8, h: 0.3, fontSize: 10, color: C.MUTED, fontFace: FONT, align: "center" });
    // draw dots
    const maxDots = 10;
    for (let d = 0; d < maxDots; d++) {
      const isFilled = d < m.dotCount;
      const dotColor = isFilled ? m.dotColor : "D1D5DB";
      s.addShape(pres.shapes.OVAL, {
        x: x + 0.2 + d * 0.265, y: 1.93, w: 0.2, h: 0.2,
        fill: { color: dotColor }, line: { color: dotColor },
      });
    }
    if (m.steps > 10) {
      s.addText("×10 more", { x: x + 0.1, y: 2.15, w: 2.8, h: 0.25, fontSize: 9, color: m.color, fontFace: FONT, align: "center" });
    }

    // Stats
    const sy = 2.5;
    [["Steps", String(m.steps)], ["Time", m.time], ["Success", m.success]].forEach(([k, v], j) => {
      s.addText(k + ":", { x: x + 0.15, y: sy + j * 0.52, w: 1.1, h: 0.45, fontSize: 11, bold: true, color: m.color, fontFace: FONT, valign: "middle" });
      const isSuccess = k === "Success";
      s.addText(v, { x: x + 1.3, y: sy + j * 0.52, w: 1.5, h: 0.45, fontSize: isSuccess ? 16 : 13, bold: true, color: isSuccess ? m.bar : C.DARK_TXT, fontFace: FONT, align: "right", valign: "middle" });
    });

    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.15, y: 4.08, w: 2.7, h: 0.04, fill: { color: C.BORDER }, line: { color: C.BORDER } });
    s.addText(m.desc, { x: x + 0.1, y: 4.18, w: 2.8, h: 1.0, fontSize: 10.5, color: C.DARK_TXT, fontFace: FONT, align: "center" });
  });

  // Bottom equation
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.12, w: 10, h: 0.505, fill: { color: C.NAVY }, line: { color: C.NAVY } });
  s.addText("Flow Matching:  x_t = (1−t)·action + t·noise    →    target: v = noise − action    →    inference: Euler ODE", {
    x: 0.3, y: 5.12, w: 9.4, h: 0.505, fontSize: 12, color: C.ICE, fontFace: MONO, align: "center", valign: "middle",
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — THE NUMERICAL BUG WE HIT
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "The Bug That Took DDIM from 0%  →  92% Success");

  // Problem side
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.82, w: 4.6, h: 4.6, fill: { color: "FFF1F2" }, line: { color: "FCA5A5", width: 1 }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 0.82, w: 4.6, h: 0.06, fill: { color: C.CORAL }, line: { color: C.CORAL } });

  pill(s, "THE PROBLEM", { x: 0.35, y: 0.93, w: 1.8, h: 0.36, bg: C.CORAL });
  s.addText("Cosine noise schedule at K = 100 steps:", { x: 0.35, y: 1.38, w: 4.3, h: 0.35, fontSize: 12, bold: true, color: "BE123C", fontFace: FONT });
  s.addText("ᾱ₉₉  ≈  2 × 10⁻⁸  (essentially zero)", { x: 0.35, y: 1.75, w: 4.3, h: 0.38, fontSize: 13, color: C.DARK_TXT, fontFace: MONO });
  s.addText("DDIM recovers the clean action by dividing by √ᾱ_t", { x: 0.35, y: 2.18, w: 4.3, h: 0.38, fontSize: 11, color: C.DARK_TXT, fontFace: FONT });
  s.addText("At t=99:   √ᾱ = 0.00016", { x: 0.35, y: 2.58, w: 4.3, h: 0.35, fontSize: 12, color: "BE123C", fontFace: MONO });
  s.addText("Prediction explodes to ±millions", { x: 0.35, y: 2.95, w: 4.3, h: 0.35, fontSize: 12, bold: true, color: C.CORAL, fontFace: FONT });
  s.addText("Agent never moves  →  0% success", { x: 0.35, y: 3.35, w: 4.3, h: 0.38, fontSize: 13, bold: true, color: C.CORAL, fontFace: FONT });
  s.addText("…despite perfect training loss! 🤯", { x: 0.35, y: 3.78, w: 4.3, h: 0.35, fontSize: 12, color: "BE123C", fontFace: FONT, italic: true });

  // Fix side
  s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 0.82, w: 4.6, h: 4.6, fill: { color: "F0FDF9" }, line: { color: "6EE7B7", width: 1 }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 0.82, w: 4.6, h: 0.06, fill: { color: C.TEAL }, line: { color: C.TEAL } });

  pill(s, "THE FIX  (2 lines)", { x: 5.25, y: 0.93, w: 1.95, h: 0.36, bg: C.TEAL, fg: C.NAVY });
  s.addText("Before (broken):", { x: 5.25, y: 1.38, w: 4.3, h: 0.32, fontSize: 11, color: "BE123C", fontFace: FONT });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.25, y: 1.72, w: 4.3, h: 0.5, fill: { color: "FFE4E8" }, line: { color: "FCA5A5" } });
  s.addText("a0 = (noisy − √(1−ᾱ)·ε) / √ᾱ", { x: 5.35, y: 1.72, w: 4.1, h: 0.5, fontSize: 11, color: "BE123C", fontFace: MONO, valign: "middle" });

  s.addText("After (fixed):", { x: 5.25, y: 2.32, w: 4.3, h: 0.32, fontSize: 11, color: "065F46", fontFace: FONT });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.25, y: 2.66, w: 4.3, h: 0.9, fill: { color: "DCFCE7" }, line: { color: "6EE7B7" } });
  s.addText([
    { text: "a0 = (noisy − √(1−ᾱ)·ε) / √ᾱ", options: { color: "065F46" } },
    { text: "\n       .clamp(min=1e-3)", options: { color: "047857", bold: true } },
  ], { x: 5.35, y: 2.66, w: 4.1, h: 0.45, fontSize: 11, fontFace: MONO, valign: "middle" });
  s.addText("a0 = a0.clamp(−1.0,  1.0)", { x: 5.35, y: 3.1, w: 4.1, h: 0.44, fontSize: 11, color: "047857", fontFace: MONO, bold: true, valign: "middle" });

  s.addText("Result:", { x: 5.25, y: 3.68, w: 0.9, h: 0.42, fontSize: 13, bold: true, color: "065F46", fontFace: FONT, valign: "middle" });
  s.addText("0%  →  92% success rate", { x: 6.2, y: 3.68, w: 3.4, h: 0.42, fontSize: 14, bold: true, color: C.TEAL, fontFace: FONT, valign: "middle" });

  s.addText("Lesson: perfect loss ≠ working policy. Check edge cases in math.", {
    x: 5.25, y: 4.22, w: 4.3, h: 0.65, fontSize: 11, color: "047857", fontFace: FONT, italic: true,
  });

  // VS divider
  s.addShape(pres.shapes.OVAL, { x: 4.6, y: 2.9, w: 0.8, h: 0.8, fill: { color: C.NAVY }, line: { color: C.NAVY } });
  s.addText("VS", { x: 4.6, y: 2.9, w: 0.8, h: 0.8, fontSize: 13, bold: true, color: C.WHITE, fontFace: FONT, align: "center", valign: "middle", margin: 0 });

  // Lesson footer
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.52, w: 10, h: 0.105, fill: { color: C.GOLD }, line: { color: C.GOLD } });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — RESULTS (embed actual bar-chart plot)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "Results: Diffusion Crushes the BC Baseline");

  // Embed the actual comparison plot (left 2/3)
  s.addImage({
    path: path.join(PLOTS, "final_comparison.png"),
    x: 0.2, y: 0.78, w: 6.6, h: 4.0,
    sizing: { type: "contain", w: 6.6, h: 4.0 },
  });

  // Right panel: key numbers
  const kpis = [
    { val: "98%",  lbl: "Flow Matching",  sub: "8.9ms · 10 steps", color: "065F46", bg: "ECFDF5" },
    { val: "92%",  lbl: "DDIM",           sub: "10ms · 10 steps",  color: C.NAVY2,  bg: "EEF2FF" },
    { val: "80%",  lbl: "DDPM",           sub: "84ms · 100 steps", color: "64748B", bg: "F8FAFC" },
    { val: "4%",   lbl: "BC Baseline",    sub: "0.6ms · 1 step",   color: C.CORAL,  bg: "FFF1F2" },
  ];

  kpis.forEach((k, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 7.05, y: 0.85 + i * 1.1, w: 2.75, h: 0.98, fill: { color: k.bg }, line: { color: C.BORDER, width: 0.5 }, shadow: mkShadow() });
    s.addText(k.val, { x: 7.05, y: 0.85 + i * 1.1, w: 1.0, h: 0.98, fontSize: 26, bold: true, color: k.color, fontFace: FONT, align: "center", valign: "middle" });
    s.addText(k.lbl, { x: 8.1, y: 0.88 + i * 1.1, w: 1.6, h: 0.4, fontSize: 12, bold: true, color: k.color, fontFace: FONT });
    s.addText(k.sub, { x: 8.1, y: 1.27 + i * 1.1, w: 1.6, h: 0.3, fontSize: 9.5, color: C.MUTED, fontFace: FONT });
  });

  // Footer callout
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.95, w: 10, h: 0.675, fill: { color: C.NAVY }, line: { color: C.NAVY } });
  s.addText([
    { text: "24.5× gap between BC (4%) and Flow Matching (98%)  |  ", options: { color: C.ICE } },
    { text: "FM matches & exceeds Chi et al. 2023 (paper: ~90%)", options: { bold: true, color: C.GOLD } },
  ], { x: 0.3, y: 4.95, w: 9.4, h: 0.675, fontSize: 13, fontFace: FONT, align: "center", valign: "middle" });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — STEPS ABLATION (embed plot)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "Ablation: How Many Denoising Steps Does DDIM Really Need?");

  // Embed the ablation plot
  s.addImage({
    path: path.join(PLOTS, "steps_ablation.png"),
    x: 0.2, y: 0.78, w: 6.3, h: 3.8,
    sizing: { type: "contain", w: 6.3, h: 3.8 },
  });

  // Key insight cards on right
  const insights = [
    { step: "1 step", emoji: "❌", insight: "Complete failure. Proves iterative denoising is genuinely needed.", color: C.CORAL, bg: "FFF1F2" },
    { step: "5 steps", emoji: "⭐", insight: "100% success at 2× the speed of default. Sweet spot for deployment!", color: "065F46", bg: "ECFDF5" },
    { step: "100 steps", emoji: "⚡", insight: "No better than 5 steps. 20× more compute, zero gain.", color: "64748B", bg: "F8FAFC" },
  ];

  insights.forEach((ins, i) => {
    wCard(s, { x: 6.7, y: 0.85 + i * 1.27, w: 3.05, h: 1.17, accent: ins.color });
    s.addText(ins.emoji + " " + ins.step, { x: 6.9, y: 0.9 + i * 1.27, w: 2.7, h: 0.4, fontSize: 14, bold: true, color: ins.color, fontFace: FONT });
    s.addText(ins.insight, { x: 6.9, y: 1.3 + i * 1.27, w: 2.7, h: 0.6, fontSize: 10.5, color: C.DARK_TXT, fontFace: FONT });
  });

  // Footer
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.73, w: 10, h: 0.895, fill: { color: C.NAVY }, line: { color: C.NAVY } });
  s.addText([
    { text: "Same trained model — no retraining needed. ", options: { color: C.ICE } },
    { text: "Just change the number of inference steps.", options: { color: C.ICE } },
    { text: "\nThe denoising landscape is so well-conditioned that 5 steps suffice for 100% success.", options: { color: C.GOLD, bold: true } },
  ], { x: 0.3, y: 4.73, w: 9.4, h: 0.895, fontSize: 12, fontFace: FONT, align: "center", valign: "middle" });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — WHY FLOW MATCHING WINS (visual path comparison)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "Why Flow Matching Beats DDPM: Straight Paths Are Easier to Invert");

  // Draw path comparison
  // DDPM panel (left)
  s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y: 0.82, w: 4.5, h: 4.5, fill: { color: "F8FAFC" }, line: { color: C.BORDER, width: 0.75 }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y: 0.82, w: 4.5, h: 0.06, fill: { color: "94A3B8" }, line: { color: "94A3B8" } });
  s.addText("DDPM / DDIM", { x: 0.3, y: 0.93, w: 4.3, h: 0.45, fontSize: 17, bold: true, color: "475569", fontFace: FONT, align: "center" });
  s.addText("Curved, roundabout path", { x: 0.3, y: 1.35, w: 4.3, h: 0.32, fontSize: 11, color: C.MUTED, fontFace: FONT, align: "center", italic: true });

  // Draw a zigzag curved path from "Noise" to "Action" using line segments
  const zigzagPoints = [
    [2.4, 4.6], [1.6, 4.0], [2.8, 3.4], [1.4, 2.8], [2.6, 2.2], [1.8, 1.7], [2.4, 1.4],
  ];
  for (let pi = 0; pi < zigzagPoints.length - 1; pi++) {
    const [x1, y1] = zigzagPoints[pi];
    const [x2, y2] = zigzagPoints[pi + 1];
    s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: "64748B", width: 2 } });
  }

  // Noise and Action endpoints
  s.addShape(pres.shapes.OVAL, { x: 2.15, y: 4.45, w: 0.5, h: 0.5, fill: { color: "94A3B8" }, line: { color: "64748B" } });
  s.addText("ε", { x: 2.15, y: 4.45, w: 0.5, h: 0.5, fontSize: 14, bold: true, color: C.WHITE, fontFace: MONO, align: "center", valign: "middle", margin: 0 });
  s.addText("Pure Noise", { x: 1.8, y: 4.97, w: 1.2, h: 0.28, fontSize: 9, color: "64748B", fontFace: FONT, align: "center" });

  s.addShape(pres.shapes.OVAL, { x: 2.15, y: 1.25, w: 0.5, h: 0.5, fill: { color: C.NAVY2 }, line: { color: C.NAVY } });
  s.addText("a₀", { x: 2.15, y: 1.25, w: 0.5, h: 0.5, fontSize: 12, bold: true, color: C.WHITE, fontFace: MONO, align: "center", valign: "middle", margin: 0 });
  s.addText("Clean Action", { x: 1.8, y: 0.97, w: 1.2, h: 0.28, fontSize: 9, color: C.NAVY, fontFace: FONT, align: "center" });

  // Step markers along path
  s.addText("100 tiny steps\n(many model calls)", { x: 0.3, y: 3.0, w: 1.3, h: 0.65, fontSize: 9, color: "94A3B8", fontFace: FONT, align: "center", italic: true });
  s.addText("↗", { x: 1.5, y: 3.1, w: 0.3, h: 0.3, fontSize: 12, color: "94A3B8", fontFace: FONT });

  // FM panel (right)
  s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 0.82, w: 4.5, h: 4.5, fill: { color: "ECFDF5" }, line: { color: "6EE7B7", width: 1 }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 0.82, w: 4.5, h: 0.06, fill: { color: C.TEAL }, line: { color: C.TEAL } });
  s.addText("Flow Matching  ★", { x: 5.4, y: 0.93, w: 4.3, h: 0.45, fontSize: 17, bold: true, color: "065F46", fontFace: FONT, align: "center" });
  s.addText("Straight line: x_t = (1−t)·a₀ + t·ε", { x: 5.4, y: 1.35, w: 4.3, h: 0.32, fontSize: 11, color: "047857", fontFace: MONO, align: "center" });

  // Draw straight line from noise to action
  s.addShape(pres.shapes.LINE, { x: 7.55, y: 4.6, w: 0, h: -3.15, line: { color: C.TEAL, width: 3 } });

  // Noise endpoint
  s.addShape(pres.shapes.OVAL, { x: 7.3, y: 4.45, w: 0.5, h: 0.5, fill: { color: "94A3B8" }, line: { color: "64748B" } });
  s.addText("ε", { x: 7.3, y: 4.45, w: 0.5, h: 0.5, fontSize: 14, bold: true, color: C.WHITE, fontFace: MONO, align: "center", valign: "middle", margin: 0 });
  s.addText("Pure Noise", { x: 6.95, y: 4.97, w: 1.2, h: 0.28, fontSize: 9, color: "64748B", fontFace: FONT, align: "center" });

  // Action endpoint
  s.addShape(pres.shapes.OVAL, { x: 7.3, y: 1.25, w: 0.5, h: 0.5, fill: { color: "065F46" }, line: { color: "047857" } });
  s.addText("a₀", { x: 7.3, y: 1.25, w: 0.5, h: 0.5, fontSize: 12, bold: true, color: C.WHITE, fontFace: MONO, align: "center", valign: "middle", margin: 0 });
  s.addText("Clean Action", { x: 6.95, y: 0.97, w: 1.2, h: 0.28, fontSize: 9, color: "065F46", fontFace: FONT, align: "center" });

  // Small tick marks on straight line (10 steps)
  for (let step = 1; step <= 9; step++) {
    const ty = 4.6 - step * 0.35;
    s.addShape(pres.shapes.RECTANGLE, { x: 7.45, y: ty, w: 0.2, h: 0.04, fill: { color: "047857" }, line: { color: "047857" } });
  }
  s.addText("10 steps\n(each perfectly\nalong straight line)", { x: 8.1, y: 2.5, w: 1.5, h: 0.75, fontSize: 9, color: "047857", fontFace: FONT, italic: true });
  s.addText("↖", { x: 7.82, y: 2.7, w: 0.3, h: 0.3, fontSize: 12, color: "047857", fontFace: FONT });

  // Footer
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.12, w: 10, h: 0.505, fill: { color: C.NAVY }, line: { color: C.NAVY } });
  s.addText("Straight paths are easier to invert. The model learns a simpler mapping → better convergence → 98% success.", {
    x: 0.3, y: 5.12, w: 9.4, h: 0.505, fontSize: 12.5, color: C.ICE, fontFace: FONT, align: "center", valign: "middle", italic: true,
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — TRAINING CURVES (embed real plot)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "Training Convergence: 300 Epochs on NVIDIA V100 SXM2");

  s.addImage({
    path: path.join(PLOTS, "training_curves", "ddpm_vs_fm_300ep.png"),
    x: 0.2, y: 0.78, w: 6.8, h: 4.15,
    sizing: { type: "contain", w: 6.8, h: 4.15 },
  });

  // Right: key notes
  wCard(s, { x: 7.2, y: 0.82, w: 2.6, h: 4.1, accent: C.NAVY });
  s.addText("Training Details", { x: 7.4, y: 0.9, w: 2.2, h: 0.42, fontSize: 13, bold: true, color: C.NAVY, fontFace: FONT });

  const details = [
    ["Optimizer",  "AdamW (lr = 1e-4)"],
    ["Schedule",   "Cosine decay"],
    ["Warmup",     "500-step linear"],
    ["Epochs",     "300 each method"],
    ["GPU",        "V100 SXM2 (32 GB)"],
    ["Time",       "~11 hrs each"],
    ["EMA decay",  "0.995"],
  ];
  details.forEach(([k, v], i) => {
    s.addText(k + ":", { x: 7.4, y: 1.42 + i * 0.45, w: 1.0, h: 0.4, fontSize: 10.5, bold: true, color: C.NAVY, fontFace: FONT, valign: "middle" });
    s.addText(v,       { x: 8.45, y: 1.42 + i * 0.45, w: 1.2, h: 0.4, fontSize: 10.5, color: C.DARK_TXT, fontFace: FONT, valign: "middle" });
  });

  s.addText("Note: losses aren't\ndirectly comparable.\nDDPM predicts ε;\nFM predicts velocity v.", {
    x: 7.4, y: 4.6, w: 2.2, h: 0.6, fontSize: 9.5, color: C.MUTED, fontFace: FONT, italic: true,
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 13 — TESTS & CODE QUALITY
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  header(s, "120 Unit Tests — Every Module Verified End-to-End");

  // Big "120" callout
  s.addShape(pres.shapes.OVAL, { x: 0.25, y: 0.85, w: 2.4, h: 2.4, fill: { color: "EEF2FF" }, line: { color: C.BORDER, width: 1 } });
  s.addText("120", { x: 0.25, y: 0.85, w: 2.4, h: 1.7, fontSize: 62, bold: true, color: C.NAVY, fontFace: FONT, align: "center", valign: "bottom" });
  s.addText("tests", { x: 0.25, y: 2.4, w: 2.4, h: 0.5, fontSize: 16, color: C.NAVY, fontFace: FONT, align: "center" });
  s.addText("all passing ✓", { x: 0.25, y: 2.8, w: 2.4, h: 0.35, fontSize: 12, color: C.TEAL, fontFace: FONT, align: "center", bold: true });

  // Module breakdown — compact table
  const modules = [
    ["test_bc_policy",      "19",  "Shape, gradients, determinism"],
    ["test_unet1d",         "20",  "FiLM, skip connections, gradients"],
    ["test_ddpm",           "11",  "Noise schedule, forward & reverse"],
    ["test_ddim",           "8",   "Determinism, numerical fix"],
    ["test_flow_matching",  "7",   "Straight-line ODE step"],
    ["test_ema",            "10",  "Decay, apply/restore cycle"],
    ["test_normalizer",     "14",  "Fit, round-trip, checkpoint"],
    ["test_integration",    "8",   "End-to-end forward pass"],
    ["test_vision_encoder", "23",  "ResNet, frozen backbone, FiLM"],
  ];

  modules.forEach((m, i) => {
    const y = 0.9 + i * 0.48;
    const bg = i % 2 === 0 ? "F8FAFC" : C.CARD_BG ?? "EEF2FF";
    s.addShape(pres.shapes.RECTANGLE, { x: 2.9, y, w: 6.9, h: 0.45, fill: { color: bg }, line: { color: C.BORDER, width: 0.25 } });
    s.addText(m[0], { x: 3.0, y, w: 2.8, h: 0.45, fontSize: 10.5, color: C.DARK_TXT, fontFace: MONO, valign: "middle" });
    s.addShape(pres.shapes.OVAL, { x: 5.9, y: y + 0.09, w: 0.27, h: 0.27, fill: { color: C.NAVY }, line: { color: C.NAVY } });
    s.addText(m[1], { x: 5.9, y: y + 0.09, w: 0.27, h: 0.27, fontSize: 8.5, bold: true, color: C.WHITE, fontFace: FONT, align: "center", valign: "middle", margin: 0 });
    s.addText(m[2], { x: 6.25, y, w: 3.4, h: 0.45, fontSize: 10.5, color: C.MUTED, fontFace: FONT, valign: "middle" });
  });

  // Command
  s.addShape(pres.shapes.RECTANGLE, { x: 0.25, y: 5.12, w: 9.5, h: 0.38, fill: { color: "1E1E3A" }, line: { color: "1E1E3A" } });
  s.addText("$ pytest tests/ -v   →   120 passed in 43s", {
    x: 0.5, y: 5.12, w: 9.0, h: 0.38, fontSize: 12, color: C.TEAL, fontFace: MONO, valign: "middle",
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
// SLIDE 14 — KEY TAKEAWAYS (dark)
// ═══════════════════════════════════════════════════════════════════════════════
{
  const s = darkSlide();

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.055, fill: { color: C.GOLD }, line: { color: C.GOLD } });

  s.addText("What We Learned", { x: 0.4, y: 0.12, w: 9.2, h: 0.6, fontSize: 26, bold: true, color: C.WHITE, fontFace: FONT, align: "center" });

  const takeaways = [
    { icon: "🔀", color: C.CORAL, head: "The multimodal problem is real",
      body: "BC averages → 4%. Diffusion picks one valid strategy → 98%. The gap is architectural, not a tuning issue." },
    { icon: "📈", color: C.TEAL, head: "Flow Matching > DDPM",
      body: "Straight-line ODE paths give the model a simpler job. Better accuracy (98%), faster inference (8.9ms)." },
    { icon: "🔧", color: C.GOLD, head: "One missing clamp = 0% success",
      body: "DDIM divided by ᾱ ≈ 2×10⁻⁸ at the final step → explosion. Two lines of code fixed it." },
    { icon: "⚡", color: C.ICE, head: "5 denoising steps achieves 100%",
      body: "The model learns such a clean landscape that 5 inference steps are enough. Real robots need speed." },
    { icon: "🏗️", color: "A8D8EA", head: "Diffusion is learnable from scratch",
      body: "68.95M-parameter U-Net, built line by line, matches published results. No black box — every piece understood." },
  ];

  takeaways.forEach((t, i) => {
    const y = 0.88 + i * 0.88;
    // Number badge
    s.addShape(pres.shapes.OVAL, { x: 0.3, y: y + 0.2, w: 0.42, h: 0.42, fill: { color: t.color }, line: { color: t.color } });
    s.addText(t.icon, { x: 0.3, y: y + 0.2, w: 0.42, h: 0.42, fontSize: 13, align: "center", valign: "middle" });
    s.addText(t.head, { x: 0.88, y: y + 0.1, w: 9.0, h: 0.38, fontSize: 13, bold: true, color: t.color, fontFace: FONT });
    s.addText(t.body, { x: 0.88, y: y + 0.46, w: 9.0, h: 0.38, fontSize: 11, color: "8090B8", fontFace: FONT });
  });

  // Footer bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.28, w: 10, h: 0.345, fill: { color: "0E1340" }, line: { color: "0E1340" } });
  s.addText("FM: 98%  |  DDIM: 92%  |  DDPM: 80%  |  BC: 4%  |  120 tests  |  github.com/BrutalCaeser/Diffusion_Robot_Control_Policy", {
    x: 0.3, y: 5.28, w: 9.4, h: 0.345, fontSize: 9, color: "4A5A90", fontFace: FONT, align: "center", valign: "middle",
  });
}


// ─── Write file ───────────────────────────────────────────────────────────────
const outPath = path.resolve(__dirname, "Diffusion_Policy_PushT.pptx");
pres.writeFile({ fileName: outPath })
  .then(() => console.log("✅  Diffusion_Policy_PushT.pptx created:", outPath))
  .catch(err => { console.error("❌", err); process.exit(1); });
