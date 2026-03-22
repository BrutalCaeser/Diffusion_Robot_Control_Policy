# Diffusion Policy for Simulated Robot Control

## ML 6140 — Machine Learning Term Project Report

---

### Team Members

| Name | Northeastern University Email |
|------|-------------------------------|
| Yashvardhan Gupta | gupta.yashv@northeastern.edu |
| Vineeth Sakhamuru | sakhamuru.v@northeastern.edu |
| Sai Krishna Reddy Maligireddy | maligireddy.s@northeastern.edu |

---

## Objectives and Significance

### Goal

The goal of this project is to implement and evaluate **Diffusion Policy** — a generative-model-based approach to robot control — for the PushT simulated manipulation task, as proposed by Chi et al. (RSS 2023). Rather than training a standard behavioral cloning policy that maps observations to a single deterministic action, we train a **Denoising Diffusion Probabilistic Model (DDPM)** that learns to iteratively denoise a *sequence* of future actions conditioned on the robot's current observation. At inference time, the model starts from pure Gaussian noise and refines it over multiple denoising steps into a coherent action trajectory that the robot executes in a receding-horizon fashion. In addition to the core DDPM pipeline, we implement **DDIM** (Denoising Diffusion Implicit Models) as an accelerated inference method that uses the same trained model but achieves roughly 10× faster action generation with minimal quality loss. As a stretch goal, we also explore **Flow Matching**, a continuous-time generative framework that replaces the discrete noise schedule with straight-line interpolation and velocity prediction, offering a simpler training objective and potentially faster convergence.

### Why It Matters

Traditional behavioral cloning — the supervised learning approach of mapping observations directly to actions — suffers from a fundamental limitation when dealing with **multimodal action distributions**. In many real-world manipulation scenarios, there exist multiple equally valid strategies for the same observation. For example, in the PushT task, an agent can approach the T-shaped block from the left or the right to push it into the target pose; both trajectories are represented in the expert demonstration dataset. A standard regression-based policy (e.g., an MLP trained with mean-squared-error loss) averages over these modes, producing an action that goes *between* the two valid strategies — often an invalid or suboptimal action that goes nowhere. Diffusion Policy elegantly solves this problem because the diffusion process can naturally represent complex, multimodal distributions over action sequences without mode collapse. Each sample from the reverse diffusion process can land in a different mode, enabling the policy to commit to one coherent strategy per rollout while still capturing the full diversity of expert behavior. This makes diffusion-based policies particularly well-suited for contact-rich manipulation tasks where the inherent multimodality of feasible solutions is a core challenge.

### Motivation and Related Work

Our motivation for choosing this project is threefold. First, diffusion models have recently emerged as one of the most impactful generative modeling paradigms, achieving state-of-the-art results in image generation (Ho et al., 2020; Dhariwal & Nichol, 2021), video synthesis, and audio generation. The application of diffusion models to robot policy learning — as demonstrated by Chi et al. (2023) — represents a compelling intersection of generative modeling and sequential decision-making, and provides an opportunity to deeply understand the diffusion framework through a hands-on implementation rather than just applying it to static data. Second, the PushT environment is an ideal testbed: it is lightweight enough to train on a single GPU in a few hours, visually intuitive for debugging, and exhibits the multimodal action distribution that motivates the use of diffusion-based policies. Third, this project lays the groundwork for a longer-term research direction connecting diffusion models to **Schrödinger Bridges** and **optimal transport** — Flow Matching is the deterministic (zero-diffusion-coefficient) limit of the Schrödinger Bridge problem, and building a clean, modular codebase that supports DDPM, DDIM, and Flow Matching with a shared U-Net backbone creates a direct pathway to exploring these more advanced generative frameworks in future research. Key related works that inform our approach include: the original **DDPM** paper (Ho et al., NeurIPS 2020) for the noise schedule and training objective; **DDIM** (Song et al., ICLR 2021) for accelerated deterministic sampling; the **Diffusion Policy** paper (Chi et al., RSS 2023) for the conditional 1D temporal U-Net architecture with FiLM conditioning and the receding-horizon action execution scheme; and the **Flow Matching** literature (Lipman et al., ICLR 2023; Liu et al., ICLR 2023) for the continuous-time alternative. Our intuition is that a well-implemented diffusion policy should significantly outperform a standard MLP behavioral cloning baseline on PushT, particularly in success rate, because the diffusion model can faithfully represent the multimodal expert action distribution rather than collapsing it to a unimodal mean.
