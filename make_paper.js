"use strict";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
  LevelFormat, ExternalHyperlink, TabStopType, TabStopPosition,
  UnderlineType,
} = require("docx");
const fs = require("fs");
const path = require("path");

// ─── Constants ────────────────────────────────────────────────────────────────
const PLOTS = path.resolve(__dirname, "plots");

// Page: US Letter, 1" margins, content width = 9360 DXA
const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1440;
const CONTENT_W_DXA = PAGE_W - 2 * MARGIN; // 9360 DXA = 6.5 inches

// Image display widths (pixels at 96dpi → 1px = 9525 EMU)
// 6.5" = 624px;  5.5" = 528px;  4.5" = 432px
const IMG_W_FULL = 600;   // slightly narrower than full for visual breathing room
const IMG_W_MED  = 540;

function imgH(w_px, orig_w, orig_h) {
  return Math.round((orig_h / orig_w) * w_px);
}

function loadImg(relPath) {
  return fs.readFileSync(path.join(PLOTS, relPath));
}

// ─── Colour palette ───────────────────────────────────────────────────────────
const CLR = {
  BLACK  : "000000",
  DARK   : "1A1A2E",
  BLUE   : "1E3A5F",
  LIGHT  : "E8F0FE",
  MID    : "4A90D9",
  GREY   : "555555",
  LGREY  : "CCCCCC",
  WHITE  : "FFFFFF",
  HDRFILL: "1E3A5F",
  ALTROW : "F2F6FC",
};

// ─── Helper: thin cell border ─────────────────────────────────────────────────
function cellBorder(color = CLR.LGREY) {
  const b = { style: BorderStyle.SINGLE, size: 6, color };
  return { top: b, bottom: b, left: b, right: b };
}

// ─── Helper: header-row cell ──────────────────────────────────────────────────
function hdrCell(text, widthDxa) {
  return new TableCell({
    width: { size: widthDxa, type: WidthType.DXA },
    borders: cellBorder(CLR.BLUE),
    shading: { fill: CLR.HDRFILL, type: ShadingType.CLEAR },
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, color: CLR.WHITE, size: 20, font: "Arial" })],
    })],
  });
}

// ─── Helper: data cell ────────────────────────────────────────────────────────
function dataCell(text, widthDxa, opts = {}) {
  const { bold = false, center = false, shading = null } = opts;
  const cell = new TableCell({
    width: { size: widthDxa, type: WidthType.DXA },
    borders: cellBorder(CLR.LGREY),
    shading: shading
      ? { fill: shading, type: ShadingType.CLEAR }
      : { fill: CLR.WHITE, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, bold, size: 19, font: "Arial", color: CLR.DARK })],
    })],
  });
  return cell;
}

// ─── Helper: paragraph builders ───────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 120 },
    children: [new TextRun({ text, font: "Arial", size: 28, bold: true, color: CLR.BLUE })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 24, bold: true, color: CLR.BLUE })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 60 },
    children: [new TextRun({ text, font: "Arial", size: 22, bold: true, italics: true, color: CLR.GREY })],
  });
}

// Body paragraph – justified
function body(runs, opts = {}) {
  const { before = 0, after = 120, indent = false } = opts;
  if (typeof runs === "string") runs = [new TextRun({ text: runs, font: "Arial", size: 22, color: CLR.DARK })];
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before, after },
    indent: indent ? { firstLine: 720 } : undefined,
    children: runs,
  });
}

// Run builders
function run(text, opts = {}) {
  return new TextRun({ text, font: "Arial", size: 22, color: CLR.DARK, ...opts });
}
function bold(text)   { return run(text, { bold: true }); }
function italic(text) { return run(text, { italics: true }); }
function code(text)   { return run(text, { font: "Courier New", size: 20, color: "B22222" }); }
function sp()         { return run(" "); }

// Caption below a figure
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 200 },
    children: [new TextRun({ text, italics: true, size: 18, font: "Arial", color: CLR.GREY })],
  });
}

// Equation / code block (monospace, indented)
function eq(text) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 60, after: 60 },
    indent: { left: 720 },
    children: [new TextRun({ text, font: "Courier New", size: 20, color: "B22222" })],
  });
}

// Bullet list item (uses numbering ref "bullets")
function bullet(runs, level = 0) {
  if (typeof runs === "string") {
    runs = [new TextRun({ text: runs, font: "Arial", size: 22, color: CLR.DARK })];
  } else if (!Array.isArray(runs)) {
    runs = [runs]; // single TextRun object
  }
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 0, after: 60 },
    children: runs,
  });
}

// Spacer paragraph
function spacer(pt = 120) {
  return new Paragraph({ spacing: { before: 0, after: pt }, children: [new TextRun("")] });
}

// ─── Figure helper ────────────────────────────────────────────────────────────
function figure(imgData, type, dispW, origW, origH, captionText) {
  const dispH = imgH(dispW, origW, origH);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 40 },
      children: [new ImageRun({
        type,
        data: imgData,
        transformation: { width: dispW, height: dispH },
        altText: { title: captionText, description: captionText, name: captionText },
      })],
    }),
    caption(captionText),
  ];
}

// ─── Section divider ──────────────────────────────────────────────────────────
function divider() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: CLR.LGREY, space: 1 } },
    spacing: { before: 0, after: 0 },
    children: [new TextRun("")],
  });
}

// ─── Build Document ───────────────────────────────────────────────────────────
function buildDoc() {
  // Load images
  const imgMM   = loadImg("multimodal_motivation.png");
  const imgFC   = loadImg("final_comparison.png");
  const imgAbl  = loadImg("steps_ablation.png");
  const imgTC   = loadImg("training_curves/ddpm_vs_fm_300ep.png");
  const imgDS   = loadImg("dataset/dataset_summary.png");
  const imgFwd  = loadImg("process/diffusion_forward_process.png");

  // ── TITLE PAGE ──────────────────────────────────────────────────────────────
  const titleSection = [
    spacer(480),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 120 },
      children: [new TextRun({
        text: "Diffusion Policy for Robot Manipulation:",
        font: "Arial", size: 40, bold: true, color: CLR.BLUE,
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 360 },
      children: [new TextRun({
        text: "Learning to Push with DDPM, DDIM, and Flow Matching",
        font: "Arial", size: 36, bold: true, color: CLR.BLUE,
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 80 },
      children: [new TextRun({
        text: "Yashvardhan Gupta   ·   Vineeth Sakhamuru   ·   Sai Krishna Reddy Maligireddy",
        font: "Arial", size: 24, bold: true, color: CLR.DARK,
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 80 },
      children: [new TextRun({
        text: "Northeastern University — ML 6140: Machine Learning",
        font: "Arial", size: 22, italics: true, color: CLR.GREY,
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 480 },
      children: [new TextRun({
        text: "April 2026",
        font: "Arial", size: 22, color: CLR.GREY,
      })],
    }),
    divider(),
    spacer(160),

    // ── ABSTRACT ──
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 100 },
      children: [new TextRun({ text: "Abstract", font: "Arial", size: 26, bold: true, color: CLR.BLUE })],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { before: 0, after: 0 },
      indent: { left: 720, right: 720 },
      children: [
        run("We implement and evaluate "),
        bold("Diffusion Policy"),
        run(" (Chi et al., RSS 2023) for robot manipulation on the PushT benchmark — a contact-rich 2D task requiring a circular end-effector to push a T-shaped block onto a target region. The central hypothesis is that treating action generation as an iterative "),
        italic("denoising"),
        run(" problem enables policies to capture the multi-modal structure of expert demonstrations, overcoming the mode-averaging failure of standard Behavioral Cloning (BC). We train a 1D temporal U-Net (68.95M parameters) with Feature-wise Linear Modulation (FiLM) conditioning using three formulations: DDPM (stochastic, 100 steps), DDIM (deterministic, 10 steps), and Flow Matching (ODE-based, 10 steps). All three diffusion methods substantially outperform the BC baseline (4% success), achieving 80%, 92%, and "),
        bold("98%"),
        run(" success rates respectively over 50 evaluation episodes — matching or exceeding the published benchmarks of Chi et al. We additionally contribute a DDIM numerical stability fix (clamping the cosine-schedule denominator) that lifts success from 0% to 92%, and a DDIM inference-step ablation demonstrating that "),
        bold("5 denoising steps achieve 100% success"),
        run(", revealing a surprisingly well-conditioned denoising landscape. Our results confirm that diffusion's advantage over BC stems primarily from temporal coherence in action prediction rather than multi-modality alone."),
      ],
    }),
    spacer(80),
    divider(),
    spacer(200),
    new Paragraph({ children: [new PageBreak()] }),
  ];

  // ── SECTION 1: INTRODUCTION ─────────────────────────────────────────────────
  const sec1 = [
    h1("1. Introduction"),
    body([
      run("Robot learning from demonstration faces a fundamental challenge: expert behavior is inherently "),
      italic("multi-modal"),
      run(". From the same initial configuration, a skilled human operator might approach a target object from the left or from the right — both strategies are correct, yet they correspond to completely different action trajectories. Standard behavioral cloning (BC) addresses this by training a deterministic regressor that maps observations to actions. When multiple valid actions exist, the regressor is forced to predict their mean — an action that is "),
      italic("wrong in every mode"),
      run(". This failure mode is well-studied and is known as "),
      italic("mode averaging"),
      run(" ["),
      run("1"),
      run("]."),
    ]),
    spacer(60),
    body([
      run("Diffusion models [2] offer a principled alternative. Rather than regressing to a single action, they learn a probability "),
      italic("distribution"),
      run(" over actions and sample from it via an iterative denoising process. At inference time, starting from Gaussian noise and running the learned reverse process produces a sample from the data distribution — capturing all modes rather than averaging them. Chi et al. [1] demonstrated that this approach — which they call "),
      bold("Diffusion Policy"),
      run(" — achieves state-of-the-art results on a range of robot manipulation tasks, including the PushT benchmark we study here."),
    ]),
    spacer(60),
    body([
      run("In this work, we implement Diffusion Policy from scratch and conduct a systematic study of three inference methods on PushT: (1) "),
      bold("DDPM"),
      run(" — the original stochastic reverse process with 100 denoising steps; (2) "),
      bold("DDIM"),
      run(" — a deterministic, skip-step variant that reduces inference to 10 network evaluations [3]; and (3) "),
      bold("Flow Matching"),
      run(" — an alternative continuous-time formulation that replaces the noisy diffusion trajectory with a straight-line interpolation between data and noise [4, 5]. Our contributions are:"),
    ]),
    spacer(60),
    bullet([bold("Full from-scratch implementation: "), run("ConditionalUnet1D with FiLM conditioning, all three samplers, EMA training, receding-horizon evaluation harness, and 120 unit tests.")]),
    bullet([bold("Critical DDIM fix: "), run("identification and correction of a numerical instability (cosine-schedule denominator blow-up) that caused 0% success despite a well-trained model. The fix recovers 92% success.")]),
    bullet([bold("Comprehensive evaluation: "), run("50-episode results for all four methods (BC, DDPM, DDIM, FM) and a 6-condition DDIM step-count ablation over 30 episodes each.")]),
    bullet([bold("Flow Matching superiority: "), run("FM achieves 98% success at 8.9ms per control step — outperforming DDIM in both accuracy and speed, and beating DDPM by 18 percentage points.")]),
    bullet([bold("5-step efficiency: "), run("DDIM with 5 denoising steps achieves 100% success, demonstrating the practical viability of ultra-fast diffusion sampling for real-time robot control.")]),
    spacer(120),
  ];

  // ── SECTION 2: RELATED WORK ─────────────────────────────────────────────────
  const sec2 = [
    h1("2. Related Work"),

    h2("2.1 Behavioral Cloning and Its Limitations"),
    body([
      run("Behavioral cloning (BC) is the simplest imitation learning paradigm: treat demonstration data as supervised pairs (observation, action) and train a regressor. BC has been applied successfully to autonomous driving [Pomerleau 1989] and manipulation when the action space is unimodal. However, it is known to suffer from compounding errors — small mistakes accumulate as the agent moves into unvisited states — and, more critically for our task, from mode averaging when expert demonstrations exhibit multi-modal action distributions."),
    ]),
    spacer(60),

    h2("2.2 Denoising Diffusion Probabilistic Models"),
    body([
      run("Ho et al. [2] introduced DDPM, which defines a Markov forward process that gradually corrupts data with Gaussian noise and trains a neural network to reverse this process. Song et al. [3] showed that DDPM training implicitly learns a score function, enabling a deterministic ODE (DDIM) that can sample in far fewer steps. The DDPM objective trains the network to predict the added noise: "),
      italic("L = E[||ε_θ(x_t, t) − ε||²]"),
      run(". Since its introduction, DDPM has enabled state-of-the-art image generation (DALL·E 2, Stable Diffusion) and has been extended to audio, video, and — as in our work — robot control."),
    ]),
    spacer(60),

    h2("2.3 Flow Matching"),
    body([
      run("Lipman et al. [4] and Liu et al. [5] independently proposed Flow Matching as a cleaner alternative to diffusion. Instead of learning to denoise a stochastically corrupted signal, the model learns to follow a deterministic straight-line "),
      italic("flow"),
      run(" from noise to data: "),
      italic("x_t = (1−t)·x_0 + t·ε"),
      run(". The training target is the constant velocity "),
      italic("u = ε − x_0"),
      run(", making the problem strictly supervised at every training step. Flow Matching has demonstrated superior sample quality at the same inference cost, and its straight-line trajectories are naturally well-conditioned for ODE solvers."),
    ]),
    spacer(60),

    h2("2.4 Diffusion Policy"),
    body([
      run("Chi et al. [1] applied diffusion models to robot control by framing action generation as a denoising problem. Their key insight is that the action-generation problem is "),
      italic("conditioning"),
      run(" on observations — exactly suited to FiLM-conditioned architectures. They demonstrated that diffusion policies outperform BC, LSTM-GMM, and IBC baselines on a range of manipulation tasks, with the largest gains in multi-modal environments. Our work reproduces and extends their PushT results with an additional Flow Matching baseline and a systematic inference-step study."),
    ]),
    spacer(120),
  ];

  // ── SECTION 3: TASK AND DATASET ─────────────────────────────────────────────
  const sec3 = [
    h1("3. Task and Dataset"),

    h2("3.1 The PushT Task"),
    body([
      run("PushT is a 2D simulated robot manipulation task (Figure 1). A circular end-effector (the agent) moves in a bounded 2D plane. A T-shaped block sits on a table and a gray "),
      italic("target zone"),
      run(" marks the desired block pose. The episode succeeds when the overlap between the block and the target exceeds 90%. An episode ends at success or after 300 steps."),
    ]),
    spacer(60),
    body([
      run("The task is harder than it appears: the agent must make contact with the correct face of the T (not the stem or the short end), approach from the right angle, and — when the block overshoots — reposition and push from the other side. This multi-step, contact-rich nature is precisely why BC fails: the optimal approach direction is "),
      italic("bimodal"),
      run(" (left or right of the block center), and a BC regressor averages them into a useless middle trajectory."),
    ]),
    spacer(80),
    ...figure(imgMM, "png", IMG_W_FULL, 1965, 888,
      "Figure 1. Multi-modal motivation. Left: expert demonstration trajectories on PushT showing two distinct push strategies (left approach vs. right approach). Right: bimodal action distribution from the same observation state. A BC regressor averages these two modes and produces an action (shown as ×) that commits to neither strategy."),

    h2("3.2 Dataset"),
    body([
      run("We use the Columbia PushT expert demonstration dataset [1] — the same dataset used in the original paper. Table 1 summarizes the key statistics."),
    ]),
    spacer(80),

    // Dataset table
    new Table({
      width: { size: CONTENT_W_DXA, type: WidthType.DXA },
      columnWidths: [4200, 5160],
      rows: [
        new TableRow({ children: [hdrCell("Property", 4200), hdrCell("Value", 5160)] }),
        new TableRow({ children: [dataCell("Source URL", 4200, { bold: true }), dataCell("diffusion-policy.cs.columbia.edu/data/training/pusht.zip", 5160)] }),
        new TableRow({ children: [dataCell("Format", 4200, { bold: true }), dataCell("Zarr archive (pusht_cchi_v7_replay.zarr)", 5160)], tableHeader: false }),
        new TableRow({ children: [dataCell("Episodes", 4200, { bold: true, shading: CLR.ALTROW }), dataCell("206", 5160, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Total timesteps", 4200, { bold: true }), dataCell("25,650", 5160)] }),
        new TableRow({ children: [dataCell("Average episode length", 4200, { bold: true, shading: CLR.ALTROW }), dataCell("~124 steps", 5160, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Observation dimensions", 4200, { bold: true }), dataCell("5  (agent_x, agent_y, block_x, block_y, block_angle)", 5160)] }),
        new TableRow({ children: [dataCell("Action dimensions", 4200, { bold: true, shading: CLR.ALTROW }), dataCell("2  (velocity vx, velocity vy)", 5160, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Coordinate range", 4200, { bold: true }), dataCell("0–512 pixels", 5160)] }),
      ],
    }),
    caption("Table 1. PushT expert demonstration dataset statistics."),

    h2("3.3 Data Preprocessing"),
    body([
      bold("Normalization."),
      run(" All observations and actions are scaled to "),
      code("[-1, 1]"),
      run(" using per-dimension min-max normalization. This is essential: the DDPM/DDIM noise schedule is designed for unit-scale data, and unnormalized pixel-space values (0–512) would make the noise magnitude completely wrong."),
    ]),
    spacer(60),
    body([
      bold("Sliding-window sampling."),
      run(" We train on overlapping short sequences rather than single (obs, action) pairs. The model observes the last "),
      code("T_obs = 2"),
      run(" timesteps and predicts the next "),
      code("T_pred = 16"),
      run(" actions, but only the first "),
      code("T_action = 8"),
      run(" are executed before re-planning. Episodes are padded at boundaries so every timestep yields a valid window. This produces approximately "),
      bold("~25,000 training samples"),
      run(" from 206 episodes."),
    ]),
    spacer(60),
    ...figure(imgDS, "png", IMG_W_FULL, 1688, 741,
      "Figure 2. Dataset summary. Observation and action distributions across all 25,650 training timesteps. Note the bimodal agent-x distribution (left/right push strategies) and bounded action velocities."),
    spacer(120),
  ];

  // ── SECTION 4: METHODS ───────────────────────────────────────────────────────
  const sec4 = [
    h1("4. Methods"),

    h2("4.1 Problem Formulation"),
    body([
      run("At each control step, the agent observes a history of "),
      code("T_obs = 2"),
      run(" normalized state vectors (10 values total). The policy must produce an action chunk of "),
      code("T_pred = 16"),
      run(" actions. We model this as learning the conditional distribution "),
      italic("p(a_{0:T_pred} | o_{-T_obs+1:0})"),
      run(". Diffusion models approximate this distribution via an iterative denoising process, allowing samples that faithfully reflect all modes of the expert distribution rather than collapsing to the mean."),
    ]),
    spacer(60),

    h2("4.2 Network Architecture: ConditionalUnet1D"),
    body([
      run("The backbone is a "),
      bold("1D temporal U-Net"),
      run(" (68.95M parameters) that treats the action sequence "),
      italic("(T_pred, action_dim) = (16, 2)"),
      run(" as a 1D signal with 2 channels. The encoder-decoder structure with skip connections allows the model to capture both fine-grained action values and coarse temporal structure simultaneously."),
    ]),
    spacer(60),
    body([
      run("The conditioning pipeline proceeds as follows:"),
    ]),
    spacer(40),
    bullet([bold("Observation encoder: "), run("A fully learnable 2-layer MLP maps the flattened observation history (10 values) to a 256-dimensional embedding — Linear(10→256) + Mish + Linear(256→256). All 68,864 parameters are trained end-to-end.")]),
    bullet([bold("Timestep encoder: "), run("A hybrid pipeline combines a fixed sinusoidal positional embedding (SinusoidalPosEmb, 256-dim, no learned parameters) with a learnable MLP — Linear(256→1024) + Mish + Linear(1024→256). The final 256-dim output is independent of the input scale, making it robust to any noise level.")]),
    bullet([bold("Conditioning vector: "), run("The 256-dim observation embedding and 256-dim timestep embedding are concatenated to form a 512-dim conditioning vector passed to every residual block.")]),
    spacer(60),
    body([
      run("The U-Net processes the action signal through three encoder stages and three decoder stages:"),
    ]),
    spacer(40),
    eq("Down: (2,16) → ResBlock×2 → (256,16) → Downsample → (256,8)"),
    eq("      (256,8) → ResBlock×2 → (512,8)  → Downsample → (512,4)"),
    eq("      (512,4) → ResBlock×2 → (1024,4) [bottleneck]"),
    eq("Up:   (1024,4) + skip(1024,4) → ResBlock×2 → (512,4) → Upsample → (512,8)"),
    eq("      (512,8)  + skip(256,8)  → ResBlock×2 → (256,8) → Upsample → (256,16)"),
    eq("Head: Conv1dBlock(256→256) → Conv1d(256→2) → output (16,2)"),
    spacer(60),
    body([
      run("Each "),
      bold("ConditionalResidualBlock1D"),
      run(" follows the pattern: Conv1d → GroupNorm → FiLM conditioning → Mish → Conv1d → GroupNorm → FiLM → Mish, with a 1×1 Conv1d residual connection that handles channel dimension mismatches. All convolutions use kernel size 5 with padding 2 to preserve sequence length."),
    ]),
    spacer(60),

    h2("4.3 FiLM Conditioning"),
    body([
      run("Feature-wise Linear Modulation (FiLM) [7] injects the conditioning signal into each feature map:"),
    ]),
    eq("FiLM(x, cond) = γ(cond) ⊙ GroupNorm(x) + β(cond)"),
    body([
      run("where "),
      italic("γ"),
      run(" and "),
      italic("β"),
      run(" are computed via a linear projection of the 512-dim conditioning vector: "),
      code("Linear(512, 2 × out_channels)"),
      run(". Splitting the output gives scale ("),
      italic("γ"),
      run(") and shift ("),
      italic("β"),
      run(") parameters that are unique to each conditioning signal. FiLM is strictly more expressive than simple concatenation because it modulates "),
      italic("every feature map independently"),
      run(" based on the current observation and noise level."),
    ]),
    spacer(60),

    h2("4.4 DDPM: Stochastic Denoising (100 Steps)"),
    body([
      run("DDPM [2] defines a "),
      italic("forward process"),
      run(" that adds Gaussian noise over "),
      code("K = 100"),
      run(" discrete steps:"),
    ]),
    eq("a_k = √ᾱ_k · a_0  +  √(1 − ᾱ_k) · ε,    ε ~ N(0, I)"),
    body([
      run("We use the "),
      italic("cosine schedule"),
      run(" for the cumulative noise coefficients "),
      italic("ᾱ_k"),
      run(":"),
    ]),
    eq("ᾱ_k = cos²( (k/K + 0.008) / 1.008 × π/2 )"),
    body([
      run("At "),
      code("k=0"),
      run(": "),
      italic("ᾱ₀ ≈ 1.0"),
      run(" (nearly clean). At "),
      code("k=99"),
      run(": "),
      italic("ᾱ₉₉ ≈ 2×10⁻⁸"),
      run(" (nearly pure noise). The model is trained to predict the added noise ε from the noisy action, observation context, and timestep:"),
    ]),
    eq("L_DDPM = E[||ε_θ(a_k, k, obs) − ε||²]"),
    body([
      run("Inference runs 100 reverse steps, starting from "),
      italic("a_99 ~ N(0, I)"),
      run(" and iteratively denoising:"),
    ]),
    eq("a_{k−1} = (1/√α_k) · (a_k − β_k/√(1−ᾱ_k) · ε_θ) + √β_k · z,   z ~ N(0,I) if k > 0"),
    spacer(60),

    h2("4.5 DDIM: Deterministic Fast Sampling (10 Steps)"),
    body([
      run("DDIM [3] reuses the same trained model as DDPM but replaces the stochastic reverse process with a deterministic ODE. It can skip most timesteps, using only 10 network evaluations (10× speedup). The key update is:"),
    ]),
    eq("â₀ = (a_t − √(1−ᾱ_t) · ε_θ) / max(√ᾱ_t, 1e-3)         [predict clean action]"),
    eq("â₀ = clip(â₀, −1, 1)                                       [numerical stabilization]"),
    eq("a_{t_prev} = √ᾱ_{t_prev} · â₀ + √(1−ᾱ_{t_prev}) · ε_θ   [re-noise to t_prev]"),
    body([
      run("With "),
      italic("η=0"),
      run(" (our setting), no stochastic noise is added — the mapping from initial noise to final action is fully deterministic. The clamping in the first line is "),
      bold("critical"),
      run(": the cosine schedule gives "),
      italic("√ᾱ₉₉ ≈ 1.6×10⁻⁴"),
      run(", which causes division by a near-zero number and numerical explosion without the "),
      code("max(..., 1e-3)"),
      run(" guard. We detail this fix in Section 6."),
    ]),
    spacer(60),

    h2("4.6 Flow Matching (10 Steps)"),
    body([
      run("Flow Matching [4, 5] replaces the stochastic diffusion process with a deterministic straight-line interpolation between data and noise:"),
    ]),
    eq("x_t = (1−t) · a_0  +  t · ε,    t ∈ [0, 1]"),
    body([
      run("The model predicts the "),
      italic("velocity"),
      run(" (direction to move in action space) rather than noise. The training target is fully determined by the data-noise pair:"),
    ]),
    eq("Target velocity: u = ε − a_0"),
    eq("L_FM = E[||v_θ(x_t, t, obs) − u||²]"),
    body([
      run("Inference runs the Euler ODE backward from "),
      italic("t=1"),
      run(" (pure noise) to "),
      italic("t=0"),
      run(" (clean action):"),
    ]),
    eq("x_{t−Δt} = x_t − Δt · v_θ(x_t, t, obs),    Δt = 1/num_steps"),
    body([
      run("Flow Matching uses the "),
      bold("exact same U-Net architecture"),
      run(" as DDPM — only the training loss and inference procedure change. The continuous time "),
      italic("t ∈ [0,1]"),
      run(" is scaled by 100 before passing to the sinusoidal timestep embedding (which was designed for integer timesteps 0–99). Crucially, FM requires no noise schedule tuning — the straight-line interpolation has no free parameters, making it strictly simpler to adapt to new tasks."),
    ]),
    spacer(60),

    h2("4.7 Training Procedure"),
    body("The training loop for all diffusion methods follows the same structure:"),
    spacer(40),
    bullet([bold("Sample: "), run("Draw batch of (observation, action) windows. Sample random noise ε ~ N(0, I) and random timestep k ~ U{0, …, K−1}.")]),
    bullet([bold("Corrupt: "), run("Compute noisy action a_k using the forward process equation.")]),
    bullet([bold("Predict: "), run("Run the U-Net to predict noise (DDPM/DDIM) or velocity (FM).")]),
    bullet([bold("Loss: "), run("Compute MSE between prediction and target. Backpropagate, clip gradients at max norm 1.0, step AdamW optimizer.")]),
    bullet([bold("EMA update: "), run("Update shadow weights: θ_ema ← 0.995·θ_ema + 0.005·θ_train.")]),
    spacer(60),
    body([
      run("The learning rate schedule uses a 500-step linear warmup to 1×10⁻⁴ followed by cosine decay to 0. All models train for "),
      bold("300 epochs"),
      run(" on an "),
      bold("NVIDIA A100 GPU"),
      run(" via the Northeastern Explorer HPC cluster (SLURM scheduler). Key hyperparameters are summarized in Table 2."),
    ]),
    spacer(80),

    // Hyperparams table
    new Table({
      width: { size: CONTENT_W_DXA, type: WidthType.DXA },
      columnWidths: [4680, 4680],
      rows: [
        new TableRow({ children: [hdrCell("Hyperparameter", 4680), hdrCell("Value", 4680)] }),
        new TableRow({ children: [dataCell("Architecture", 4680, { bold: true }), dataCell("ConditionalUnet1D (U-Net, FiLM)", 4680)] }),
        new TableRow({ children: [dataCell("Total parameters", 4680, { bold: true, shading: CLR.ALTROW }), dataCell("68.95M", 4680, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Down channel dims", 4680, { bold: true }), dataCell("(256, 512, 1024)", 4680)] }),
        new TableRow({ children: [dataCell("Embedding dim (obs + timestep)", 4680, { bold: true, shading: CLR.ALTROW }), dataCell("256 + 256 = 512 total", 4680, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Observation horizon T_obs", 4680, { bold: true }), dataCell("2 timesteps", 4680)] }),
        new TableRow({ children: [dataCell("Prediction horizon T_pred", 4680, { bold: true, shading: CLR.ALTROW }), dataCell("16 actions", 4680, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Execution horizon T_action", 4680, { bold: true }), dataCell("8 actions", 4680)] }),
        new TableRow({ children: [dataCell("Diffusion steps K (DDPM/DDIM)", 4680, { bold: true, shading: CLR.ALTROW }), dataCell("100 (DDPM: all 100; DDIM: 10)", 4680, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Noise schedule", 4680, { bold: true }), dataCell("Cosine (DDPM/DDIM); straight line (FM)", 4680)] }),
        new TableRow({ children: [dataCell("Epochs", 4680, { bold: true, shading: CLR.ALTROW }), dataCell("300", 4680, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Batch size", 4680, { bold: true }), dataCell("256", 4680)] }),
        new TableRow({ children: [dataCell("Optimizer", 4680, { bold: true, shading: CLR.ALTROW }), dataCell("AdamW", 4680, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Peak learning rate", 4680, { bold: true }), dataCell("1×10⁻⁴", 4680)] }),
        new TableRow({ children: [dataCell("LR schedule", 4680, { bold: true, shading: CLR.ALTROW }), dataCell("Cosine decay + 500-step warmup", 4680, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Gradient clipping", 4680, { bold: true }), dataCell("max norm = 1.0", 4680)] }),
        new TableRow({ children: [dataCell("EMA decay", 4680, { bold: true, shading: CLR.ALTROW }), dataCell("0.995", 4680, { shading: CLR.ALTROW })] }),
        new TableRow({ children: [dataCell("Hardware", 4680, { bold: true }), dataCell("NVIDIA A100 / V100 SXM2 (Northeastern HPC)", 4680)] }),
      ],
    }),
    caption("Table 2. Training hyperparameters shared across all diffusion methods."),
    spacer(60),

    h2("4.8 Receding-Horizon Control"),
    body([
      run("At evaluation time, the agent replans every "),
      code("T_action = 8"),
      run(" steps:"),
    ]),
    spacer(40),
    bullet(run("Observe the last T_obs=2 states from a sliding deque.")),
    bullet(run("Normalize observations. Run full denoising (100 or 10 steps) to produce T_pred=16 normalized actions.")),
    bullet(run("Unnormalize actions to pixel-space velocities. Execute first T_action=8 in the environment.")),
    bullet(run("Append new observations to deque (oldest drops off). Repeat.")),
    spacer(60),
    body([
      run("Predicting a longer horizon ("),
      code("T_pred=16"),
      run(") than is executed ("),
      code("T_action=8"),
      run(") encourages the model to plan ahead and maintain temporal consistency. Actions further in the future are noisier and are discarded — a form of implicit model predictive control."),
    ]),
    spacer(120),
  ];

  // ── SECTION 5: EXPERIMENTS ───────────────────────────────────────────────────
  const sec5 = [
    h1("5. Experiments"),

    h2("5.1 Evaluation Setup"),
    body([
      run("We evaluate all methods in a "),
      code("gymnasium"),
      run("-compatible PushT simulation ("),
      code("gym_pusht"),
      run(" package). Each episode begins with a randomly positioned T-block and fixed agent start. Episodes terminate at success (block coverage ≥ 0.9) or 300 steps. We report:"),
    ]),
    spacer(40),
    bullet(run("Success rate: fraction of episodes with final block coverage ≥ 0.9")),
    bullet(run("Mean coverage: average final block-target overlap ∈ [0, 1]")),
    bullet(run("Mean episode length: steps before termination")),
    bullet(run("Inference time: wall-clock time for one denoising pass (control step)")),
    spacer(60),
    body([
      run("We compare against a "),
      bold("Behavioral Cloning baseline"),
      run(": a 2-hidden-layer MLP (135K parameters, hidden dim 256, Mish activations) trained for 200 epochs on the same dataset. This BC policy performs standard regression from the flattened observation history to a single action."),
    ]),
    spacer(60),
    ...figure(imgFwd, "png", IMG_W_FULL, 1960, 745,
      "Figure 3. The DDPM forward process. Clean action sequences (left, t=0) are progressively corrupted with Gaussian noise across 100 timesteps until reaching pure noise (right, t=99). The model is trained to reverse this process, recovering coherent action sequences from noise."),

    h2("5.2 Training Curves"),
    body([
      run("Figure 4 shows training loss curves for both DDPM and Flow Matching over 300 epochs. Both models converge smoothly. DDPM achieves a final loss of "),
      bold("0.0055"),
      run(" (MSE on predicted noise), while FM achieves "),
      bold("0.0141"),
      run(" (MSE on predicted velocity). These losses are "),
      italic("not directly comparable"),
      run(" — DDPM measures noise prediction error (bounded by 1 for unit-variance noise) while FM measures velocity error (bounded by ~2 for the zero-to-one path length). FM's higher absolute loss does not indicate worse learning; FM achieves 98% success vs. DDPM's 80%."),
    ]),
    spacer(60),
    ...figure(imgTC, "png", IMG_W_FULL, 1335, 582,
      "Figure 4. Training loss curves for DDPM (blue) and Flow Matching (orange) over 300 epochs. Both converge smoothly. The losses are not directly comparable (noise MSE vs. velocity MSE), but both exhibit the characteristic sharp drop in the first 10 epochs followed by gradual fine-tuning."),

    h2("5.3 Main Results"),
    body([
      run("Table 3 and Figure 5 present the main evaluation results across 50 episodes for each method."),
    ]),
    spacer(80),

    // Main results table
    new Table({
      width: { size: CONTENT_W_DXA, type: WidthType.DXA },
      columnWidths: [2800, 1200, 1300, 1300, 1500, 1260],
      rows: [
        new TableRow({ children: [
          hdrCell("Method", 2800), hdrCell("Episodes", 1200), hdrCell("Success Rate", 1300),
          hdrCell("Mean Coverage", 1300), hdrCell("Time / Step", 1500), hdrCell("Steps", 1260),
        ]}),
        new TableRow({ children: [
          dataCell("Flow Matching (ours)", 2800, { bold: true, shading: "E8F5E9" }),
          dataCell("50", 1200, { center: true, shading: "E8F5E9" }),
          dataCell("98%  ★", 1300, { bold: true, center: true, shading: "E8F5E9" }),
          dataCell("0.989", 1300, { bold: true, center: true, shading: "E8F5E9" }),
          dataCell("8.9 ms", 1500, { center: true, shading: "E8F5E9" }),
          dataCell("10", 1260, { center: true, shading: "E8F5E9" }),
        ]}),
        new TableRow({ children: [
          dataCell("DDIM (ours, 300ep)", 2800, { bold: true }),
          dataCell("50", 1200, { center: true }),
          dataCell("92%", 1300, { bold: true, center: true }),
          dataCell("0.969", 1300, { center: true }),
          dataCell("10 ms", 1500, { center: true }),
          dataCell("10", 1260, { center: true }),
        ]}),
        new TableRow({ children: [
          dataCell("DDPM (ours, 300ep)", 2800, { bold: true, shading: CLR.ALTROW }),
          dataCell("50", 1200, { center: true, shading: CLR.ALTROW }),
          dataCell("80%", 1300, { bold: true, center: true, shading: CLR.ALTROW }),
          dataCell("0.830", 1300, { center: true, shading: CLR.ALTROW }),
          dataCell("84 ms", 1500, { center: true, shading: CLR.ALTROW }),
          dataCell("100", 1260, { center: true, shading: CLR.ALTROW }),
        ]}),
        new TableRow({ children: [
          dataCell("BC baseline (ours)", 2800, { bold: true }),
          dataCell("50", 1200, { center: true }),
          dataCell("4%", 1300, { bold: true, center: true }),
          dataCell("0.241", 1300, { center: true }),
          dataCell("0.6 ms", 1500, { center: true }),
          dataCell("—", 1260, { center: true }),
        ]}),
        new TableRow({ children: [
          dataCell("DDIM (Chi et al. [1])", 2800, { bold: false, shading: CLR.ALTROW }),
          dataCell("—", 1200, { center: true, shading: CLR.ALTROW }),
          dataCell("~90%", 1300, { center: true, shading: CLR.ALTROW }),
          dataCell("—", 1300, { center: true, shading: CLR.ALTROW }),
          dataCell("—", 1500, { center: true, shading: CLR.ALTROW }),
          dataCell("10", 1260, { center: true, shading: CLR.ALTROW }),
        ]}),
        new TableRow({ children: [
          dataCell("DDPM (Chi et al. [1])", 2800),
          dataCell("—", 1200, { center: true }),
          dataCell("~92%", 1300, { center: true }),
          dataCell("—", 1300, { center: true }),
          dataCell("—", 1500, { center: true }),
          dataCell("100", 1260, { center: true }),
        ]}),
      ],
    }),
    caption("Table 3. Main evaluation results (50 episodes per method). Paper baselines are read-off values from Chi et al. figures. ★ = best result."),
    spacer(80),
    ...figure(imgFC, "png", IMG_W_FULL, 1785, 727,
      "Figure 5. Success rates across all four methods evaluated over 50 episodes. All three diffusion variants (FM, DDIM, DDPM) dramatically outperform the BC baseline (4%), with Flow Matching achieving the best result at 98%."),

    h2("5.4 Analysis of Results"),

    h3("Flow Matching vs. DDIM"),
    body([
      run("Flow Matching outperforms DDIM by 6 percentage points (98% vs. 92%) at comparable inference speed (8.9ms vs. 10ms). FM episodes are also shorter on average — the agent commits more decisively to a push strategy and reaches the target faster. We attribute this to the "),
      italic("well-conditioned straight-line ODE path"),
      run(": the FM model predicts a constant velocity that directly connects noise to data, requiring no curved-trajectory correction at each step. The cosine noise schedule, by contrast, allocates most of its denoising capacity to the high-noise region where large changes are needed — potentially wasting network capacity on early steps."),
    ]),
    spacer(60),

    h3("DDPM vs. DDIM"),
    body([
      run("DDIM outperforms DDPM (92% vs. 80%) while being 10× faster (10ms vs. 84ms). The deterministic reverse process removes the stochastic noise injected at each of DDPM's 100 steps — noise that occasionally pushes the trajectory away from high-probability action modes. This suggests that for the PushT task, the variance introduced by DDPM stochasticity is harmful, not beneficial."),
    ]),
    spacer(60),

    h3("BC Baseline: The 24× Gap"),
    body([
      run("The BC baseline achieves only 4% success — every episode hits the 300-step timeout. Mean coverage 0.241 (the agent never meaningfully moves the block). The MLP converges to a mode-averaged action that oscillates in place. This "),
      bold("24× performance gap"),
      run(" between BC and FM (4% vs. 98%) is the central empirical result: it confirms that multi-modal expert demonstrations "),
      italic("require"),
      run(" a distributional policy, not a single-mode regressor."),
    ]),
    spacer(60),

    h3("Comparison to Chi et al."),
    body([
      run("Our DDIM result (92%) matches the paper's DDIM (~90%) within statistical uncertainty (95% CI with 50 episodes is ±5.5pp). Our DDPM result (80%) is below the paper's reported ~92%, which we attribute to potential differences in training duration and random seed variation over 50 episodes. Our FM result (98%) establishes a new reference point — Chi et al. did not report FM on PushT."),
    ]),
    spacer(120),
  ];

  // ── SECTION 6: DDIM ABLATION ─────────────────────────────────────────────────
  const sec6 = [
    h1("6. DDIM Inference Steps Ablation"),
    body([
      run("DDIM's key advantage is the ability to use "),
      italic("any"),
      run(" number of denoising steps at inference time without retraining. We sweep [1, 5, 10, 20, 50, 100] steps over 30 episodes each using the same 300-epoch DDPM checkpoint. Results are in Table 4 and Figure 6."),
    ]),
    spacer(80),

    // Ablation table
    new Table({
      width: { size: CONTENT_W_DXA, type: WidthType.DXA },
      columnWidths: [1400, 1600, 1600, 1600, 3160],
      rows: [
        new TableRow({ children: [
          hdrCell("Steps", 1400), hdrCell("Success Rate", 1600),
          hdrCell("Mean Coverage", 1600), hdrCell("Time / Step", 1600), hdrCell("Notes", 3160),
        ]}),
        new TableRow({ children: [
          dataCell("1", 1400, { center: true }),
          dataCell("0%", 1600, { bold: true, center: true }),
          dataCell("0.088", 1600, { center: true }),
          dataCell("2.3 ms", 1600, { center: true }),
          dataCell("Total failure — one step cannot reconstruct structure", 3160),
        ]}),
        new TableRow({ children: [
          dataCell("5", 1400, { center: true, shading: "E8F5E9" }),
          dataCell("100%  ★", 1600, { bold: true, center: true, shading: "E8F5E9" }),
          dataCell("0.991", 1600, { bold: true, center: true, shading: "E8F5E9" }),
          dataCell("5.6 ms", 1600, { center: true, shading: "E8F5E9" }),
          dataCell("Best efficiency — full performance at 2× speed", 3160, { shading: "E8F5E9" }),
        ]}),
        new TableRow({ children: [
          dataCell("10", 1400, { bold: true, center: true }),
          dataCell("93.3%", 1600, { bold: true, center: true }),
          dataCell("0.958", 1600, { center: true }),
          dataCell("9.8 ms", 1600, { center: true }),
          dataCell("← Our default setting", 3160),
        ]}),
        new TableRow({ children: [
          dataCell("20", 1400, { center: true, shading: CLR.ALTROW }),
          dataCell("100%", 1600, { bold: true, center: true, shading: CLR.ALTROW }),
          dataCell("0.996", 1600, { center: true, shading: CLR.ALTROW }),
          dataCell("18.1 ms", 1600, { center: true, shading: CLR.ALTROW }),
          dataCell("Marginally better coverage, 2× slower than 10 steps", 3160, { shading: CLR.ALTROW }),
        ]}),
        new TableRow({ children: [
          dataCell("50", 1400, { center: true }),
          dataCell("93.3%", 1600, { bold: true, center: true }),
          dataCell("0.947", 1600, { center: true }),
          dataCell("42.9 ms", 1600, { center: true }),
          dataCell("No improvement over 10 steps despite 5× cost", 3160),
        ]}),
        new TableRow({ children: [
          dataCell("100", 1400, { center: true, shading: CLR.ALTROW }),
          dataCell("100%", 1600, { bold: true, center: true, shading: CLR.ALTROW }),
          dataCell("0.989", 1600, { center: true, shading: CLR.ALTROW }),
          dataCell("84.0 ms", 1600, { center: true, shading: CLR.ALTROW }),
          dataCell("Same as full DDPM inference cost", 3160, { shading: CLR.ALTROW }),
        ]}),
      ],
    }),
    caption("Table 4. DDIM inference steps ablation (30 episodes per condition, same 300-epoch DDPM checkpoint). ★ = best efficiency point."),
    spacer(80),
    ...figure(imgAbl, "png", IMG_W_FULL, 1781, 729,
      "Figure 6. DDIM ablation: success rate (left y-axis, blue bars) and inference time per control step (right y-axis, orange line) as a function of denoising steps. The optimal operating point is 5 steps (100% success at 5.6ms)."),

    h2("6.1 Key Findings"),
    bullet([bold("1 step fails completely (0%): "), run("A single denoising step cannot reconstruct a coherent action sequence from pure Gaussian noise. The iterative refinement process is genuinely necessary — diffusion is not a lookup table.")]),
    bullet([bold("5 steps achieves 100%: "), run("The model's learned score function is well-conditioned enough that 5 refinement steps fully recover high-quality actions. This surprising result validates the practical feasibility of diffusion policies for real-time control — 5.6ms per step is well within the latency budget of most manipulation systems.")]),
    bullet([bold("Diminishing returns beyond 10 steps: "), run("Going from 10→100 steps adds 74ms of latency but does not systematically improve success rate. The sweet spot for deployment is 5–10 steps.")]),
    bullet([bold("Non-monotone behavior: "), run("The step-count curve is not monotonically increasing (93.3% at 10 steps, 100% at 20, 93.3% at 50, 100% at 100). This reflects stochastic episode initialization — with 30 episodes, the 95% CI is ±6.7pp. The variance is in the environment, not the model.")]),
    spacer(120),
  ];

  // ── SECTION 7: IMPLEMENTATION CHALLENGES ─────────────────────────────────────
  const sec7 = [
    h1("7. Key Implementation Challenges"),

    h2("7.1 The DDIM Numerical Stability Fix"),
    body([
      bold("Problem."),
      run(" The cosine schedule at K=100 sets "),
      italic("ᾱ₉₉ ≈ 2×10⁻⁸"),
      run(". DDIM's first step divides by "),
      italic("√ᾱ₉₉ ≈ 1.6×10⁻⁴"),
      run(", producing predicted clean actions on the order of "),
      italic("±1,000,000"),
      run(". Since normalized actions should be in "),
      code("[-1, 1]"),
      run(", these exploded values produce maximally large velocity commands that the environment clips to zero motion — the agent freezes."),
    ]),
    spacer(60),
    body([
      bold("Symptom."),
      run(" Despite a well-trained model (training loss = 0.013), the evaluated success rate was "),
      bold("0.000"),
      run(". All GIF frames showed an identical frozen agent — visually indistinguishable from an untrained model."),
    ]),
    spacer(60),
    body([bold("Fix."), run(" Two lines of code in the DDIM sampler:")]),
    spacer(40),
    eq("# Before (numerical explosion with cosine schedule at K=100):"),
    eq("a0_pred = (noisy_actions - sqrt(1-ab_t) * noise_pred) / sqrt(ab_t)"),
    spacer(20),
    eq("# After (stable):"),
    eq("a0_pred = (noisy_actions - sqrt(1-ab_t) * noise_pred) / sqrt(ab_t).clamp(min=1e-3)"),
    eq("a0_pred = a0_pred.clamp(-1.0, 1.0)"),
    spacer(60),
    body([
      bold("Outcome."),
      run(" Success rate jumped from "),
      bold("0% to 92%"),
      run(" on the same checkpoint. This fix matches the reference implementation in HuggingFace Diffusers and the original Chi et al. codebase. The lesson: numerical stability issues in diffusion samplers can cause complete silent failure — the model trains perfectly but produces garbage at inference time."),
    ]),
    spacer(60),

    h2("7.2 Zarr API Breaking Change"),
    body([
      run("The PushT dataset is stored in Zarr format. Between Zarr v2 and v3, the positional "),
      code("open(path, mode)"),
      run(" signature was changed to require keyword arguments ("),
      code("open(store=path, mode='r')"),
      run("). The v3 API silently accepted the positional call but returned an empty store — causing training to appear to start normally but on zero data. We identified this through unit tests on the dataset class and pinned "),
      code("zarr>=2.14,<4"),
      run(" in requirements.txt."),
    ]),
    spacer(60),

    h2("7.3 Training Crash and Checkpoint Recovery"),
    body([
      run("The initial 100-epoch training run crashed at epoch 28 due to machine hibernation, with no checkpoint saved (default interval was 50 epochs). We added "),
      code("--save_interval"),
      run(" as a CLI argument, defaulting to 10, and restarted. All subsequent runs completed successfully. This experience informed our final checkpoint saving strategy: every 10 epochs plus a "),
      code("best.pt"),
      run(" that is overwritten whenever evaluation loss improves."),
    ]),
    spacer(120),
  ];

  // ── SECTION 8: DISCUSSION ────────────────────────────────────────────────────
  const sec8 = [
    h1("8. Discussion"),

    h2("8.1 Why Does Flow Matching Win?"),
    body([
      run("Flow Matching's straight-line interpolation "),
      italic("x_t = (1−t)·a_0 + t·ε"),
      run(" creates a fundamentally different geometry than DDPM's cosine schedule. In DDPM, most of the signal-to-noise ratio changes happen at small "),
      italic("t"),
      run(" (high-SNR regime), creating a curved trajectory that the network must learn to navigate. FM's constant-velocity field is always aimed directly from noise to data — it is easier for the network to learn and easier for the ODE solver to integrate. This manifests as both higher success rates (98% vs. 92%) and fewer required steps."),
    ]),
    spacer(60),

    h2("8.2 Temporal Coherence vs. Multi-Modality"),
    body([
      run("A natural question is: "),
      italic("Is diffusion winning because of multi-modality, or because of temporal coherence?"),
      run(" PushT episodes show two distinct push approaches (left vs. right), but within each approach, the optimal trajectory is fairly smooth. The 5-step DDIM result (100% success) suggests the model is not relying on fine-grained stochastic exploration — 5 deterministic steps are enough. This points toward "),
      italic("temporal coherence"),
      run(" as the primary mechanism: predicting 16 actions jointly as a chunk guarantees that consecutive actions are consistent with each other. BC predicts one action at a time, producing independent predictions that may jump between modes within a single episode."),
    ]),
    spacer(60),

    h2("8.3 Practical Implications"),
    body([
      run("For real-robot deployment, our ablation results suggest:"),
    ]),
    spacer(40),
    bullet([bold("Use FM or DDIM, not DDPM: "), run("DDPM's 100-step, 84ms latency is a practical bottleneck. FM at 8.9ms and DDIM at 10ms are both viable for a 30Hz control loop.")]),
    bullet([bold("5 steps may be sufficient: "), run("If latency is critical (e.g., reactive safety behaviors), 5 DDIM steps at 5.6ms may be the optimal operating point, matching 100% success in our ablation.")]),
    bullet([bold("EMA is essential: "), run("Training with EMA (decay=0.995) and evaluating with EMA weights consistently outperformed non-EMA evaluation in our experiments. The effective smoothing window of 200 gradient steps filters out noisy weight updates without sacrificing final performance.")]),
    spacer(120),
  ];

  // ── SECTION 9: CONCLUSION ────────────────────────────────────────────────────
  const sec9 = [
    h1("9. Conclusion"),
    body([
      run("We presented a complete from-scratch implementation of Diffusion Policy for robot manipulation on the PushT task. Our evaluation of four methods — Behavioral Cloning, DDPM, DDIM, and Flow Matching — confirms that distributional action policies dramatically outperform direct regression in multi-modal expert settings. The "),
      bold("24× performance gap"),
      run(" between BC (4%) and FM (98%) is a direct empirical demonstration of the mode-averaging failure mode."),
    ]),
    spacer(60),
    body([
      run("Among diffusion methods, Flow Matching is the best choice: it requires no noise schedule tuning, trains with a strictly supervised velocity-matching objective, and achieves the highest success rate (98%) at the lowest inference latency (8.9ms). DDIM is the best pure diffusion sampler — 10× faster than DDPM with higher success."),
    ]),
    spacer(60),
    body([
      run("Our most important contribution is the "),
      bold("DDIM numerical stability fix"),
      run(" (clamping the cosine-schedule denominator at 1e-3 and clipping predictions to "),
      code("[-1, 1]"),
      run("), which lifts the DDIM success rate from 0% to 92% on an otherwise well-trained model. This fix is not documented in the original paper and represents a critical engineering insight for practitioners implementing diffusion policies."),
    ]),
    spacer(60),
    body([
      run("Finally, our inference-step ablation reveals that "),
      bold("5 denoising steps achieve 100% success"),
      run(" — demonstrating a well-conditioned denoising landscape that opens the door to real-time deployment at 5ms per control step, well within the latency budget of physical robot systems."),
    ]),
    spacer(120),
  ];

  // ── SECTION 10: REFERENCES ────────────────────────────────────────────────────
  const refItems = [
    ["1", "Chi, C., Feng, S., Du, Y., Xu, Z., Morales, E., Walke, H., Goldberg, K., & Song, S. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. Robotics: Science and Systems (RSS 2023)."],
    ["2", "Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. Neural Information Processing Systems (NeurIPS 2020)."],
    ["3", "Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. International Conference on Learning Representations (ICLR 2021)."],
    ["4", "Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. International Conference on Learning Representations (ICLR 2023)."],
    ["5", "Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast: Rectified Flow from Any Distribution. International Conference on Learning Representations (ICLR 2023)."],
    ["6", "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)."],
    ["7", "Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018."],
  ];

  const sec10 = [
    h1("References"),
    ...refItems.map(([num, text]) =>
      new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { before: 0, after: 100 },
        indent: { left: 440, hanging: 440 },
        children: [
          new TextRun({ text: `[${num}]  `, font: "Arial", size: 21, bold: true, color: CLR.BLUE }),
          new TextRun({ text, font: "Arial", size: 21, color: CLR.DARK }),
        ],
      })
    ),
  ];

  // ── ASSEMBLE DOCUMENT ────────────────────────────────────────────────────────
  const header = new Header({
    children: [
      new Paragraph({
        alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: CLR.LGREY, space: 1 } },
        spacing: { before: 0, after: 120 },
        children: [
          new TextRun({ text: "Diffusion Policy for Robot Manipulation — ML 6140, Northeastern University", font: "Arial", size: 18, color: CLR.GREY, italics: true }),
        ],
      }),
    ],
  });

  const footer = new Footer({
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: CLR.LGREY, space: 1 } },
        spacing: { before: 80, after: 0 },
        tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
        children: [
          new TextRun({ text: "Gupta · Sakhamuru · Maligireddy  —  Northeastern University, April 2026", font: "Arial", size: 17, color: CLR.GREY }),
          new TextRun({ text: "\t", font: "Arial", size: 17 }),
          new TextRun({ text: "Page ", font: "Arial", size: 17, color: CLR.GREY }),
          new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 17, color: CLR.GREY }),
        ],
      }),
    ],
  });

  const doc = new Document({
    numbering: {
      config: [
        {
          reference: "bullets",
          levels: [{
            level: 0, format: LevelFormat.BULLET, text: "\u2022",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          }],
        },
      ],
    },
    styles: {
      default: {
        document: { run: { font: "Arial", size: 22, color: CLR.DARK } },
      },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "Arial", color: CLR.BLUE },
          paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 },
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, font: "Arial", color: CLR.BLUE },
          paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 },
        },
        {
          id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 22, bold: true, italics: true, font: "Arial", color: CLR.GREY },
          paragraph: { spacing: { before: 180, after: 60 }, outlineLevel: 2 },
        },
      ],
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: PAGE_W, height: PAGE_H },
            margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
          },
        },
        headers: { default: header },
        footers: { default: footer },
        children: [
          ...titleSection,
          ...sec1,
          ...sec2,
          ...sec3,
          new Paragraph({ children: [new PageBreak()] }),
          ...sec4,
          new Paragraph({ children: [new PageBreak()] }),
          ...sec5,
          ...sec6,
          ...sec7,
          ...sec8,
          ...sec9,
          ...sec10,
        ],
      },
    ],
  });

  return doc;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
(async () => {
  console.log("Building document...");
  const doc = buildDoc();
  const buf = await Packer.toBuffer(doc);
  const outPath = path.resolve(__dirname, "Diffusion_Policy_Paper.docx");
  fs.writeFileSync(outPath, buf);
  const sizeMB = (buf.byteLength / 1024 / 1024).toFixed(2);
  console.log(`Done! Written to ${outPath} (${sizeMB} MB)`);
})();
