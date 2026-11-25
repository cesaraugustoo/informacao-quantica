(function (global) {
  "use strict";

  const SEMINAR_TOPICS = [
    {
      topicId: 1,
      firstSlide: 2,
      en: "1. Classical ECC vs. QECC (No-Cloning & SU(2))",
      "pt-br": "1. ECC Clássico vs. QECC (Não-Clonagem e SU(2))"
    },
    {
      topicId: 2,
      firstSlide: 4,
      en: "2. Pauli Discretization & Knill–Laflamme",
      "pt-br": "2. Discretização de Pauli e Knill–Laflamme"
    },
    {
      topicId: 3,
      firstSlide: 6,
      en: "3. Stabilizer Codes & [[5,1,3]] Perfect Code",
      "pt-br": "3. Códigos Estabilizadores e Código Perfeito [[5,1,3]]"
    },
    {
      topicId: 4,
      firstSlide: 10,
      en: "4. Coherence |ρ₀₁| & Passive DFS (dfs_2)",
      "pt-br": "4. Coerência |ρ₀₁| e DFS Passivo (dfs_2)"
    },
    {
      topicId: 5,
      firstSlide: 9,
      en: "5. Non-Destructive Syndrome & Feed-Forward",
      "pt-br": "5. Síndrome Não Destrutiva e Feed-Forward"
    },
    {
      topicId: 6,
      firstSlide: 12,
      en: "6. Recent Hardware Milestones (Willow, qLDPC, Cat)",
      "pt-br": "6. Marcos Experimentais Recentes (Willow, qLDPC, Gato)"
    },
    {
      topicId: 7,
      firstSlide: 14,
      en: "7. Gain Thresholds (p*), Metrics & CUDA Benchmarks",
      "pt-br": "7. Limiares de Ganho (p*), Métricas e Benchmarks CUDA"
    }
  ];

  const SEMINAR_SLIDES = [
    {
      index: 0,
      topicId: 1,
      en: {
        title: "Quantum Error Correction Codes (QECC): Theory, Stabilizer Circuits, Decoherence-Free Subspaces & GPU Simulation",
        subtitle: "Graduate Seminar Presentation — Python 3.12 + PyTorch CUDA 13.0 (RTX 5080 sm_120) + eigenkit v1.3.0",
        bullets: [
          "Unifies analytical QECC theory (No-Cloning, Pauli Discretization, Knill–Laflamme, Stabilizers, DFS) with live big-endian simulation.",
          "Compares active stabilizer recovery (bit_flip_3, phase_flip_3, [[5,1,3]]) against passive symmetry protection (dfs_2).",
          "Separates Entanglement Fidelity (1 - p) from Haar State Fidelity (1 - 2p/3) for exact brentq gain thresholds p*."
        ]
      },
      "pt-br": {
        title: "Códigos de Correção de Erros Quânticos (QECC): Teoria, Circuitos Estabilizadores, Subespaços Livres de Decoerência e Simulação em GPU",
        subtitle: "Apresentação de Seminário de Pós-Graduação — Python 3.12 + PyTorch CUDA 13.0 (RTX 5080 sm_120) + eigenkit v1.3.0",
        bullets: [
          "Unifica a teoria analítica de QECC (Não-Clonagem, Discretização de Pauli, Knill–Laflamme, Estabilizadores, DFS) com simulação big-endian ao vivo.",
          "Contrasta recuperação ativa por estabilizadores (bit_flip_3, phase_flip_3, [[5,1,3]]) com proteção passiva por simetria (dfs_2).",
          "Separa estritamente Fidelidade de Emaranhamento (1 - p) de Fidelidade Média de Estado de Haar (1 - 2p/3) nos limiares exatos p*."
        ]
      },
      equationsHtml: "α|0⟩ + β|1⟩ ↦ α|0<sub>L</sub>⟩ + β|1<sub>L</sub>⟩ &nbsp;|&nbsp; ⟨i<sub>L</sub>| E<sub>a</sub><sup>†</sup> E<sub>b</sub> |j<sub>L</sub>⟩ = C<sub>ab</sub> δ<sub>ij</sub>",
      demoPreset: { target_tab: "simulator", code: "five_qubit", channel: "depolarizing", state_preset: "+i", error_mode: "pauli_string", pauli_string: "IXIII", p: 0.05, stage_key: "decoded" }
    },
    {
      index: 1,
      topicId: 1,
      en: {
        title: "Slide 1 — Seminar Roadmap (Topics 1–7)",
        subtitle: "Seven pillars from foundational obstructions to GPU-verified thresholds",
        bullets: [
          "1. Why QECC differs from Classical ECC (no-cloning theorem, SU(2) continuum, projective collapse).",
          "2. Error Discretization via Pauli basis expansion & Knill–Laflamme conditions (C_ab δ_ij).",
          "3. Stabilizer Codes & Quantum Hamming Bound (bit_flip_3, phase_flip_3, [[5,1,3]] perfect code).",
          "4. Off-Diagonal Coherence (|ρ₀₁|) & Passive Decoherence-Free Subspaces (dfs_2).",
          "5. Non-Destructive Ancilla Syndrome Extraction (phase kickback) & classical feed-forward.",
          "6. Recent Experimental Milestones (Google Willow, Harvard 48 logical qubits, IBM 288-qubit qLDPC, Cat Qubits).",
          "7. Analytical Gain Thresholds (p*), Metric Consistency (F_e vs. F_state) & RTX 5080 CUDA Benchmarks."
        ]
      },
      "pt-br": {
        title: "Slide 1 — Roteiro do Seminário (Tópicos 1–7)",
        subtitle: "Sete pilares dos obstáculos fundamentais aos limiares verificados em GPU",
        bullets: [
          "1. Por que QECC difere de ECC Clássico (Teorema da Não-Clonagem, contínuo SU(2), colapso projetivo).",
          "2. Discretização de Erros via expansão na base de Pauli e Condições de Knill–Laflamme (C_ab δ_ij).",
          "3. Códigos Estabilizadores e Limite Quântico de Hamming (bit_flip_3, phase_flip_3, código perfeito [[5,1,3]]).",
          "4. Coerência Fora da Diagonal (|ρ₀₁|) e Subespaços Livres de Decoerência Passivos (dfs_2).",
          "5. Extração Não Destrutiva de Síndrome por Ancilla (retrocesso de fase) e realimentação clássica.",
          "6. Marcos Experimentais Recentes (Google Willow, 48 qubits lógicos de Harvard, qLDPC de 288 qubits da IBM, Qubits Gato).",
          "7. Limiares Analíticos de Ganho (p*), Consistência de Métricas (F_e vs. F_state) e Benchmarks CUDA na RTX 5080."
        ]
      },
      equationsHtml: "F<sub>e,raw</sub>(p) = 1 − p &nbsp;&nbsp;vs.&nbsp;&nbsp; F<sub>state,raw</sub>(p) = 1 − (2/3)p",
      demoPreset: { target_tab: "curves", code: "five_qubit", channel: "depolarizing", state_preset: "+i", p: 0.05, metric: "entanglement" }
    },
    {
      index: 2,
      topicId: 1,
      en: {
        title: "Slide 2 — Challenge 1: The No-Cloning Theorem",
        subtitle: "Linearity and unitarity forbid copying unknown quantum states",
        bullets: [
          "Classical repetition copies b ↦ (b, b, b) via fan-out; quantum mechanics strictly forbids arbitrary-state cloning (Wootters–Zurek–Dieks).",
          "Assuming unitary U(|ψ⟩|0⟩) = |ψ⟩|ψ⟩ and U(|φ⟩|0⟩) = |φ⟩|φ⟩, inner-product preservation (U†U = I) forces ⟨ψ|φ⟩ = ⟨ψ|φ⟩², so ⟨ψ|φ⟩ ∈ {0, 1}.",
          "Instead of cloning |ψ⟩^⊗n, QECCs entangle logical information across a multi-qubit code subspace span{|0_L⟩, |1_L⟩}."
        ]
      },
      "pt-br": {
        title: "Slide 2 — Desafio 1: O Teorema da Não-Clonagem",
        subtitle: "Linearidade e unitariedade proíbem copiar estados quânticos desconhecidos",
        bullets: [
          "A repetição clássica copia b ↦ (b, b, b) por fan-out; a mecânica quântica proíbe clonar estados arbitrários (Wootters–Zurek–Dieks).",
          "Supondo unitário U(|ψ⟩|0⟩) = |ψ⟩|ψ⟩ e U(|φ⟩|0⟩) = |φ⟩|φ⟩, a preservação do produto interno (U†U = I) impõe ⟨ψ|φ⟩ = ⟨ψ|φ⟩², logo ⟨ψ|φ⟩ ∈ {0, 1}.",
          "Em vez de clonar |ψ⟩^⊗n, os QECCs emaranham a informação lógica em um subespaço de código span{|0_L⟩, |1_L⟩}."
        ]
      },
      equationsHtml: "⟨ψ|φ⟩ = ⟨ψ|φ⟩<sup>2</sup> &nbsp;⟹&nbsp; ⟨ψ|φ⟩ ∈ {0, 1}<br>α|0⟩ + β|1⟩ &nbsp;↦&nbsp; α|0<sub>L</sub>⟩ + β|1<sub>L</sub>⟩ &nbsp;(e.g. α|000⟩ + β|111⟩)",
      demoPreset: { target_tab: "simulator", code: "bit_flip_3", channel: "bit_flip", state_preset: "+i", error_mode: "pauli_string", pauli_string: "III", stage_key: "encoded" }
    },
    {
      index: 3,
      topicId: 1,
      en: {
        title: "Slide 3 — Challenges 2 & 3: Continuous SU(2) Errors & Projective Collapse",
        subtitle: "Analog rotations on the Bloch sphere and destructive direct measurement",
        bullets: [
          "Continuous SU(2) Error Manifold: Physical noise acts continuously via U(θ, n̂) = exp(-i (θ/2) n̂·σ) and CPTP Kraus maps ∑ E_k ρ E_k† (bit flips, phase flips, infinitesimal over-rotations).",
          "Projective Measurement Collapse: Measuring a data qubit of α|000⟩ + β|111⟩ directly in {|0⟩, |1⟩} collapses to |000⟩ (|α|²) or |111⟩ (|β|²), erasing off-diagonal coherence ρ₀₁ = αβ*.",
          "Design Requirement: Extract only error-syndrome parity while learning zero information about α and β."
        ]
      },
      "pt-br": {
        title: "Slide 3 — Desafios 2 e 3: Erros Contínuos em SU(2) e Colapso Projetivo",
        subtitle: "Rotações analógicas na esfera de Bloch e medição direta destrutiva",
        bullets: [
          "Variedade Contínua de Erros em SU(2): O ruído físico atua continuamente via U(θ, n̂) = exp(-i (θ/2) n̂·σ) e mapas CPTP de Kraus ∑ E_k ρ E_k† (inversões de bit, inversões de fase, sobre-rotações infinitesimais).",
          "Colapso por Medição Projetiva: Medir diretamente um qubit de dados de α|000⟩ + β|111⟩ em {|0⟩, |1⟩} colapsa para |000⟩ (|α|²) ou |111⟩ (|β|²), apagando a coerência fora da diagonal ρ₀₁ = αβ*.",
          "Requisito de Projeto: Extrair apenas a paridade da síndrome de erro sem obter nenhuma informação sobre α e β."
        ]
      },
      equationsHtml: "U(θ, n̂) = cos(θ/2) I − i sin(θ/2)(n<sub>x</sub>X + n<sub>y</sub>Y + n<sub>z</sub>Z) ∈ SU(2)<br>ρ<sub>01</sub> = αβ* → 0 (direct data readout / leitura direta)",
      demoPreset: { target_tab: "simulator", code: "bit_flip_3", channel: "bit_flip", state_preset: "+i", error_mode: "su2_rotation", su2_qubit: 0, su2_theta: 0.7853981633974483, su2_nx: 1, su2_ny: 0, su2_nz: 0, stage_key: "noisy" }
    },
    {
      index: 4,
      topicId: 2,
      en: {
        title: "Slide 4 — Error Discretization via Pauli Expansion",
        subtitle: "Syndrome measurement digitizes continuous quantum errors into discrete Paulis",
        bullets: [
          "Every single-qubit Kraus operator E_k expands in the orthonormal Pauli basis {I, X, Y, Z} via c_{k,P} = (1/2) Tr(P E_k).",
          "Acting on qubit q of |ψ_L⟩ produces a coherent superposition of Pauli-errored branches: ∑_P c_{k,P} P^(q) |ψ_L⟩.",
          "Projective Digitization: Measuring stabilizer syndromes onto ancillas projects onto a single syndrome subspace Π_s, collapsing the continuous rotation into a discrete Pauli error P_s^(q) that unitary recovery R_s fixes with unit fidelity."
        ]
      },
      "pt-br": {
        title: "Slide 4 — Discretização de Erros via Expansão de Pauli",
        subtitle: "A medição de síndrome digitaliza erros quânticos contínuos em Paulis discretos",
        bullets: [
          "Todo operador de Kraus de um qubit E_k se expande na base ortonormal de Pauli {I, X, Y, Z} com c_{k,P} = (1/2) Tr(P E_k).",
          "Atuar no qubit q de |ψ_L⟩ cria uma superposição coerente de ramos de Pauli: ∑_P c_{k,P} P^(q) |ψ_L⟩.",
          "Digitalização Projetiva: Medir síndromes de estabilizadores nas ancillas projeta o estado em um único subespaço Π_s, colapsando a rotação contínua em um erro discreto P_s^(q) corrigido com fidelidade 1.0."
        ]
      },
      equationsHtml: "E<sub>k</sub> = c<sub>k,I</sub>I + c<sub>k,X</sub>X + c<sub>k,Y</sub>Y + c<sub>k,Z</sub>Z, &nbsp;&nbsp; c<sub>k,P</sub> = ½ Tr(P E<sub>k</sub>)<br>Π<sub>s</sub> E<sub>k</sub><sup>(q)</sup>|ψ<sub>L</sub>⟩ = c<sub>k,P<sub>s</sub></sub> P<sub>s</sub><sup>(q)</sup>|ψ<sub>L</sub>⟩",
      demoPreset: { target_tab: "simulator", code: "five_qubit", channel: "depolarizing", state_preset: "+i", error_mode: "su2_rotation", su2_qubit: 0, su2_theta: 1.05, su2_nx: 1, su2_ny: -1, su2_nz: 2, stage_key: "syndrome_extracted" }
    },
    {
      index: 5,
      topicId: 2,
      en: {
        title: "Slide 5 — The Knill–Laflamme Conditions",
        subtitle: "Necessary and sufficient algebraic conditions for exact quantum error correction",
        bullets: [
          "Theorem (1997): Code C = span{|i_L⟩} with projector P_C corrects error set E = {E_a} iff ⟨i_L| E_a† E_b |j_L⟩ = C_ab δ_ij (P_C E_a† E_b P_C = C_ab P_C) for a Hermitian matrix C_ab independent of i, j.",
          "Off-diagonal i ≠ j enforces orthogonality of corrupted codewords; diagonal i = j forbids learning logical amplitudes α, β.",
          "Verified via verify_knill_laflamme (tol = 1e-12): bit_flip_3 (0.000e+00), phase_flip_3 (0.000e+00), five_qubit (1.972e-31), dfs_2 (0.000e+00)."
        ]
      },
      "pt-br": {
        title: "Slide 5 — As Condições de Knill–Laflamme",
        subtitle: "Condições algébricas necessárias e suficientes para correção exata de erros quânticos",
        bullets: [
          "Teorema (1997): O código C = span{|i_L⟩} com projetor P_C corrige o conjunto de erros E = {E_a} se, e somente se, ⟨i_L| E_a† E_b |j_L⟩ = C_ab δ_ij (P_C E_a† E_b P_C = C_ab P_C) para uma matriz Hermitiana C_ab independente de i, j.",
          "Termos i ≠ j garantem ortogonalidade dos ramos corrompidos; i = j impede vazamento de informação sobre α, β.",
          "Verificado via verify_knill_laflamme (tol = 1e-12): bit_flip_3 (0.000e+00), phase_flip_3 (0.000e+00), five_qubit (1.972e-31), dfs_2 (0.000e+00)."
        ]
      },
      equationsHtml: "⟨i<sub>L</sub>| E<sub>a</sub><sup>†</sup> E<sub>b</sub> |j<sub>L</sub>⟩ = C<sub>ab</sub> δ<sub>ij</sub> &nbsp;⟺&nbsp; P<sub>C</sub> E<sub>a</sub><sup>†</sup> E<sub>b</sub> P<sub>C</sub> = C<sub>ab</sub> P<sub>C</sub> &nbsp;(C<sub>ab</sub><sup>*</sup> = C<sub>ba</sub>)",
      demoPreset: { target_tab: "theory", code: "five_qubit", channel: "depolarizing", state_preset: "+i" }
    },
    {
      index: 6,
      topicId: 3,
      en: {
        title: "Slide 6 — 3-Qubit Bit-Flip and Phase-Flip Codes ([[3,1,1]])",
        subtitle: "Repetition stabilizers, Hadamard duality H^⊗3, and weight-2 aliasing",
        bullets: [
          "Bit-Flip (bit_flip_3): |0_L⟩ = |000⟩, |1_L⟩ = |111⟩, stabilizers ZZI, IZZ, logicals X_L = XXX, Z_L = ZII. Syndromes 00/10/11/01 → III/XII/IXI/IIX.",
          "Phase-Flip (phase_flip_3): Hadamard dual |0_L⟩ = |+++⟩, |1_L⟩ = |---⟩, stabilizers XXI, IXX, logicals X_L = ZZZ, Z_L = XII. Syndromes 00/10/11/01 → III/ZII/IZI/IIZ.",
          "Limitation: Weight-2 IXX aliases onto syndrome 10 → XII, causing logical XXX = X_L failure; orthogonal errors (Z on bit_flip_3) go undetected."
        ]
      },
      "pt-br": {
        title: "Slide 6 — Códigos de 3 Qubits de Inversão de Bit e Inversão de Fase ([[3,1,1]])",
        subtitle: "Estabilizadores de repetição, dualidade de Hadamard H^⊗3 e aliasing de peso 2",
        bullets: [
          "Inversão de Bit (bit_flip_3): |0_L⟩ = |000⟩, |1_L⟩ = |111⟩, estabilizadores ZZI, IZZ, lógicos X_L = XXX, Z_L = ZII. Síndromes 00/10/11/01 → III/XII/IXI/IIX.",
          "Inversão de Fase (phase_flip_3): Dual de Hadamard |0_L⟩ = |+++⟩, |1_L⟩ = |---⟩, estabilizadores XXI, IXX, lógicos X_L = ZZZ, Z_L = XII. Síndromes 00/10/11/01 → III/ZII/IZI/IIZ.",
          "Limitação: O erro de peso 2 IXX sofre aliasing na síndrome 10 → XII, causando inversão lógica XXX = X_L; erros ortogonais (Z em bit_flip_3) não são detectados."
        ]
      },
      equationsHtml: "F<sub>e</sub>(p) = (1 − p)<sup>3</sup> + 3p(1 − p)<sup>2</sup> = 1 − 3p<sup>2</sup> + 2p<sup>3</sup> &gt; 1 − p &nbsp;&nbsp;(0 &lt; p &lt; 0.5)",
      demoPreset: { target_tab: "simulator", code: "bit_flip_3", channel: "bit_flip", state_preset: "+i", error_mode: "pauli_string", pauli_string: "IXI", stage_key: "recovered" }
    },
    {
      index: 7,
      topicId: 3,
      en: {
        title: "Slide 7 — Quantum Hamming Bound & Synthesis to [[5,1,3]]",
        subtitle: "Non-degenerate sphere-packing bound for correcting arbitrary single-qubit X, Y, Z errors",
        bullets: [
          "Each weight-w Pauli error (w ≤ t) selects C(n,w) sites and 3^w non-identity Paulis {X, Y, Z}, mapping the 2^k code space into orthogonal 2^k-dimensional subspaces inside 2^n.",
          "For k = 1, t = 1, disjointness requires 2(1 + 3n) ≤ 2^n: n = 3 → 20 > 8 (fails), n = 4 → 26 > 16 (fails), n = 5 → 32 = 2^5 (exact saturation).",
          "All 2^4 = 16 orthogonal 2D subspaces correspond bijectively to IIIII plus the 15 single-qubit {X, Y, Z} errors."
        ]
      },
      "pt-br": {
        title: "Slide 7 — Limite Quântico de Hamming e Síntese para o [[5,1,3]]",
        subtitle: "Limite não degenerado de empacotamento de esferas para erros arbitrários X, Y, Z de 1 qubit",
        bullets: [
          "Cada erro de peso w ≤ t escolhe C(n,w) sítios e 3^w Paulis não identidade {X, Y, Z}, mapeando o espaço de código 2^k em subespaços ortogonais em 2^n.",
          "Para k = 1, t = 1, a disjunção exige 2(1 + 3n) ≤ 2^n: n = 3 → 20 > 8 (falha), n = 4 → 26 > 16 (falha), n = 5 → 32 = 2^5 (saturação exata).",
          "Todos os 2^4 = 16 subespaços ortogonais bidimensionais correspondem bijetivamente a IIIII mais os 15 erros de peso 1 em {X, Y, Z}."
        ]
      },
      equationsHtml: "2<sup>k</sup> ∑<sub>w=0..t</sub> C(n, w) 3<sup>w</sup> ≤ 2<sup>n</sup> &nbsp;⟹&nbsp; n = 5: 2(1 + 3·5) = 32 = 2<sup>5</sup>",
      demoPreset: { target_tab: "theory", code: "five_qubit", channel: "depolarizing", state_preset: "+i" }
    },
    {
      index: 8,
      topicId: 3,
      en: {
        title: "Slide 8 — [[5,1,3]] Laflamme Perfect Code: Stabilizers & 1024-Term Enumerator",
        subtitle: "Cyclic generators, 16 bijective syndromes, and exact coset weight distribution",
        bullets: [
          "Cyclic Stabilizers (five_qubit): S₀ = XZZXI, S₁ = IXZZX, S₂ = XIXZZ, S₃ = ZXIXZ; logical operators X_L = XXXXX, Z_L = ZZZZZ (with Z(0) sign alignment in encode_ops()).",
          "16 Bijective 4-Bit Syndromes (0000–1111): Every weight-1 Pauli (X, Y, Z on q0..q4) maps to a unique non-zero 4-bit syndrome.",
          "1024-Term Weight Enumerator (4^5 = 1024): Logical-I coset counts [1, 15, 0, 60, 135, 45] (sum = 256); X, Y, Z cosets each [0, 0, 30, 70, 90, 66]. Notice A₂^(I) = 0 (all 90 weight-2 errors fail)."
        ]
      },
      "pt-br": {
        title: "Slide 8 — Código Perfeito [[5,1,3]]: Estabilizadores e Enumerador de 1024 Termos",
        subtitle: "Geradores cíclicos, 16 síndromes bijetivas e distribuição exata de pesos por coclasse",
        bullets: [
          "Estabilizadores Cíclicos (five_qubit): S₀ = XZZXI, S₁ = IXZZX, S₂ = XIXZZ, S₃ = ZXIXZ; operadores lógicos X_L = XXXXX, Z_L = ZZZZZ (com alinhamento Z(0) em encode_ops()).",
          "16 Síndromes Bijetivas de 4 Bits (0000–1111): Cada Pauli de peso 1 (X, Y, Z em q0..q4) possui síndrome única de 4 bits.",
          "Enumerador de Pesos de 1024 Termos (4^5 = 1024): Coclasse lógica I: [1, 15, 0, 60, 135, 45] (soma = 256); coclasses X, Y, Z: [0, 0, 30, 70, 90, 66] cada. Note A₂^(I) = 0 (todos os 90 erros de peso 2 falham)."
        ]
      },
      equationsHtml: "A<sup>(I)</sup> = [1, 15, 0, 60, 135, 45] &nbsp;|&nbsp; A<sup>(X,Y,Z)</sup> = [0, 0, 30, 70, 90, 66]",
      demoPreset: { target_tab: "simulator", code: "five_qubit", channel: "depolarizing", state_preset: "+i", error_mode: "pauli_string", pauli_string: "IYIII", stage_key: "syndrome_extracted" }
    },
    {
      index: 9,
      topicId: 5,
      en: {
        title: "Slide 9 — Non-Destructive Ancilla Syndrome Extraction & Feed-Forward",
        subtitle: "Phase kickback onto ancilla a_k and conditional classical recovery recovery_ops(syndrome)",
        bullets: [
          "Applying H(a_k) · Λ(S_k) · H(a_k) maps |ψ_err⟩|0⟩ to ((I+S_k)/2 |ψ_err⟩)|0⟩ + ((I-S_k)/2 |ψ_err⟩)|1⟩.",
          "Because S_k E = (-1)^{s_k} E S_k, the data+ancilla state factors into a product state (E|ψ_L⟩) ⊗ |s₀…s_{r-1}⟩ with zero leakage of α, β.",
          "Classical Feed-Forward: Big-endian (B, 2^{n_data}, 2^{n_ancilla}) projective collapse extracts syndrome (ancilla n_data as MSB) and dispatches recovery_ops(syndrome) before decode_ops()."
        ]
      },
      "pt-br": {
        title: "Slide 9 — Extração Não Destrutiva de Síndrome por Ancilla e Feed-Forward",
        subtitle: "Retrocesso de fase (phase kickback) na ancilla a_k e recuperação clássica condicional recovery_ops(syndrome)",
        bullets: [
          "Aplicar H(a_k) · Λ(S_k) · H(a_k) leva |ψ_err⟩|0⟩ a ((I+S_k)/2 |ψ_err⟩)|0⟩ + ((I-S_k)/2 |ψ_err⟩)|1⟩.",
          "Como S_k E = (-1)^{s_k} E S_k, o estado conjunto dados–ancilla fatora-se no estado produto (E|ψ_L⟩) ⊗ |s₀…s_{r-1}⟩ com vazamento zero de α, β.",
          "Realimentação Clássica Condicional: O colapso projetivo big-endian (B, 2^{n_data}, 2^{n_ancilla}) lê a syndrome (ancilla n_data como MSB) e executa recovery_ops(syndrome) antes de decode_ops()."
        ]
      },
      equationsHtml: "|ψ<sub>err</sub>⟩|0⟩<sub>a<sub>k</sub></sub> →<sup>H·Λ(S<sub>k</sub>)·H</sup> (½(I + S<sub>k</sub>)|ψ<sub>err</sub>⟩)|0⟩ + (½(I − S<sub>k</sub>)|ψ<sub>err</sub>⟩)|1⟩",
      demoPreset: { target_tab: "simulator", code: "phase_flip_3", channel: "phase_flip", state_preset: "+i", error_mode: "pauli_string", pauli_string: "IZI", stage_key: "syndrome_extracted" }
    },
    {
      index: 10,
      topicId: 4,
      en: {
        title: "Slide 10 — Passive Decoherence-Free Subspaces (dfs_2)",
        subtitle: "Symmetry protection with zero ancillas vs. symmetry-breaking tradeoff",
        bullets: [
          "Encoding (X(1), CNOT(0→1)): H_DFS = span{|01⟩, |10⟩} (n_data = 2, n_ancilla = 0, empty syndrome \"\").",
          "Collective Dephasing Invariance: (Z⊗Z)|01⟩ = -|01⟩ and (Z⊗Z)|10⟩ = -|10⟩ (unobservable global phase -1); (Rz(φ)⊗Rz(φ))|ψ_L⟩ = |ψ_L⟩ identically for all φ ∈ ℝ and F = 1.0 for all p ∈ [0, 1].",
          "Symmetry-Breaking Tradeoff: Under independent phase_flip, F_e(p) = (1-p)² + p² = 1 - 2p + 2p² < 1 - p on (0, 0.5), whereas active phase_flip_3 achieves 1 - 3p² + 2p³."
        ]
      },
      "pt-br": {
        title: "Slide 10 — Subespaços Livres de Decoerência Passivos (dfs_2)",
        subtitle: "Proteção por simetria com zero ancillas vs. compromisso por quebra de simetria",
        bullets: [
          "Codificação (X(1), CNOT(0→1)): H_DFS = span{|01⟩, |10⟩} (n_data = 2, n_ancilla = 0, síndrome vazia \"\").",
          "Invariância sob Defasagem Coletiva: (Z⊗Z)|01⟩ = -|01⟩ e (Z⊗Z)|10⟩ = -|10⟩ (fase global -1 inobservável); (Rz(φ)⊗Rz(φ))|ψ_L⟩ = |ψ_L⟩ identicamente para todo φ ∈ ℝ e F = 1.0 para todo p ∈ [0, 1].",
          "Compromisso por Quebra de Simetria: Sob phase_flip independente, F_e(p) = (1-p)² + p² = 1 - 2p + 2p² < 1 - p em (0, 0.5), enquanto phase_flip_3 ativo alcança 1 - 3p² + 2p³."
        ]
      },
      equationsHtml: "(R<sub>z</sub>(φ) ⊗ R<sub>z</sub>(φ))(α|01⟩ + β|10⟩) = α|01⟩ + β|10⟩ &nbsp;∀ φ ∈ ℝ &nbsp;⟹&nbsp; F = 1.0",
      demoPreset: { target_tab: "simulator", code: "dfs_2", channel: "collective_dephasing", state_preset: "+i", error_mode: "coherent_collective", collective_phi: 1.5707963267948966, stage_key: "decoded" }
    },
    {
      index: 11,
      topicId: 4,
      en: {
        title: "Slide 11 — Off-Diagonal Coherence |ρ₀₁| & DFS Tradeoffs",
        subtitle: "Preserving quantum phase coherence on |+i⟩ (p = 0.05, 18,000 shots)",
        bullets: [
          "On |+i⟩ = (|0⟩ + i|1⟩)/√2 (ideal |ρ₀₁| = 0.5), logical Pauli channels satisfy |ρ₀₁| = (1/2)|P_I - P_X + P_Y - P_Z|.",
          "Active stabilizer codes suppress first-order O(p) coherence loss to O(p²) (bit_flip_3: 0.493389 MC vs. 0.492750 analytical; phase_flip_3: 0.491333 vs. 0.492750; five_qubit: 0.485556 vs. 0.485112).",
          "Passive dfs_2 under collective_dephasing achieves exact 0.500000 vs. 0.450000 unprotected raw."
        ]
      },
      "pt-br": {
        title: "Slide 11 — Coerência Fora da Diagonal |ρ₀₁| e Compromissos de DFS",
        subtitle: "Preservação da coerência de fase quântica em |+i⟩ (p = 0.05, 18.000 shots)",
        bullets: [
          "Em |+i⟩ = (|0⟩ + i|1⟩)/√2 (|ρ₀₁| ideal = 0.5), canais de Pauli lógicos satisfazem |ρ₀₁| = (1/2)|P_I - P_X + P_Y - P_Z|.",
          "Códigos estabilizadores ativos suprimem o decaimento de O(p) para O(p²) (bit_flip_3: 0.493389 MC vs. 0.492750 analítico; phase_flip_3: 0.491333 vs. 0.492750; five_qubit: 0.485556 vs. 0.485112).",
          "O dfs_2 passivo sob collective_dephasing atinge 0.500000 exato frente a 0.450000 sem codificação."
        ]
      },
      equationsHtml: "|ρ<sub>01</sub>| = ½ |P<sub>I</sub> − P<sub>X</sub> + P<sub>Y</sub> − P<sub>Z</sub>| &nbsp;&nbsp;(on |+i⟩: ideal |ρ₀₁| = 0.5)",
      demoPreset: { target_tab: "simulator", code: "dfs_2", channel: "collective_dephasing", state_preset: "+i", error_mode: "stochastic_sample", p: 0.05, stage_key: "decoded" }
    },
    {
      index: 12,
      topicId: 6,
      en: {
        title: "Slide 12 — Recent Developments (1/2): Threshold Theorem, Willow & Neutral Atoms",
        subtitle: "Exponential logical error suppression below threshold and reconfigurable logical processors",
        bullets: [
          "Quantum Threshold Theorem: Below circuit threshold p_th, surface codes suppress logical error exponentially: ε_L(d) ∝ (p / p_th)^{(d+1)/2} = Λ^{-(d+1)/2}.",
          "Google Quantum AI Willow (Nature 2025): Rotated surface codes d = 3 → 5 → 7 on 105 qubits achieved error suppression factor Λ = 2.14 ± 0.02 per distance step and 0.143% logical error per cycle at d = 7 (2.4× beyond physical break-even).",
          "Harvard / Lukin Neutral-Atom Processor (Bluvstein et al., Nature 2024): Coherent optical-tweezer shuttling of 87Rb atoms encoded 48 logical qubits (280 physical atoms) using [[8,3,2]] codes and transversal logical gates."
        ]
      },
      "pt-br": {
        title: "Slide 12 — Desenvolvimentos Recentes (1/2): Teorema do Limiar, Willow e Átomos Neutros",
        subtitle: "Supressão exponencial do erro lógico abaixo do limiar e processadores lógicos reconfiguráveis",
        bullets: [
          "Teorema do Limiar Quântico: Abaixo do limiar de circuito p_th, códigos de superfície suprimem o erro lógico exponencialmente: ε_L(d) ∝ (p / p_th)^{(d+1)/2} = Λ^{-(d+1)/2}.",
          "Google Quantum AI Willow (Nature 2025): Códigos de superfície rotacionados d = 3 → 5 → 7 em 105 qubits atingiram fator de supressão Λ = 2.14 ± 0.02 por passo de distância e 0.143% de erro lógico por ciclo em d = 7 (2.4× além do equilíbrio físico).",
          "Processador de Átomos Neutros de Harvard / Lukin (Bluvstein et al., Nature 2024): Transporte coerente por pinças ópticas de átomos de 87Rb codificou 48 qubits lógicos (280 átomos físicos) com códigos [[8,3,2]] e portas transversais."
        ]
      },
      equationsHtml: "ε<sub>L</sub>(d) ∝ (p / p<sub>th</sub>)<sup>(d+1)/2</sup> = Λ<sup>−(d+1)/2</sup> &nbsp;&nbsp;(Willow: Λ = 2.14 ± 0.02)",
      demoPreset: { target_tab: "curves", code: "five_qubit", channel: "depolarizing", state_preset: "+i", metric: "entanglement" }
    },
    {
      index: 13,
      topicId: 6,
      en: {
        title: "Slide 13 — Recent Developments (2/2): IBM qLDPC & Bosonic Cat Qubits",
        subtitle: "Breaking 2D encoding-rate limits and exploiting hardware-biased noise",
        bullets: [
          "IBM Bivariate Bicycle qLDPC Codes (Bravyi et al., Nature 2024): Non-local weight-6 checks bypass the 2D Bravyi–Poulin–Terhal bound (k d² = O(n)). The [[144,12,12]] gross code encodes k = 12 logical qubits at d = 12 using 288 physical qubits (~0.7% threshold, ~10× qubit savings vs. surface code).",
          "Hardware-Biased Bosonic Cat Qubits (Alice & Bob / Réglade et al. 2024; AWS / Putterman et al. 2025): Two-photon dissipation stabilizes |±α⟩, yielding exponential bit-flip suppression (Γ_X ∝ e^{-2|α|²}, >10 s lifetime) so outer repetition codes only need to correct phase flips (Z)."
        ]
      },
      "pt-br": {
        title: "Slide 13 — Desenvolvimentos Recentes (2/2): qLDPC da IBM e Qubits Gato Bosônicos",
        subtitle: "Superando limites bidimensionais de taxa e explorando ruído assimétrico em hardware",
        bullets: [
          "Códigos qLDPC Bivariate Bicycle da IBM (Bravyi et al., Nature 2024): Verificações não locais de peso 6 superam o limite 2D de Bravyi–Poulin–Terhal (k d² = O(n)). O código gross [[144,12,12]] codifica k = 12 qubits lógicos à distância 12 com 288 qubits físicos (limiar ~0.7%, economia de ~10×).",
          "Qubits Gato Bosônicos com Ruído Enviesado (Alice & Bob / Réglade et al. 2024; AWS / Putterman et al. 2025): A dissipação de dois fótons estabiliza |±α⟩, produzindo supressão exponencial de bit-flip (Γ_X ∝ e^{-2|α|²}, >10 s), deixando apenas inversões de fase (Z) para o código externo."
        ]
      },
      equationsHtml: "[[144, 12, 12]] (288 qubits) &nbsp;|&nbsp; Γ<sub>X</sub> ∝ exp(−2|α|<sup>2</sup>) (biased Z noise / ruído enviesado em Z)",
      demoPreset: { target_tab: "simulator", code: "phase_flip_3", channel: "phase_flip", state_preset: "+i", error_mode: "stochastic_sample", p: 0.05, stage_key: "decoded" }
    },
    {
      index: 14,
      topicId: 7,
      en: {
        title: "Slide 14 — Analytical Gain Thresholds & Metric Consistency",
        subtitle: "Strict separation of Entanglement Fidelity (1 - p) and Haar State Fidelity (1 - 2p/3)",
        bullets: [
          "Entanglement Fidelity (F_e): Equals logical-identity survival P_I(p); raw baseline is F_{e,raw}(p) = 1 - p.",
          "Haar-Averaged State Fidelity (F_state): Averaging uniformly over SU(2) (or the 6 cardinal states) yields F_state(p) = 1 - (2/3)(1 - F_e(p)); raw baseline is F_{state,raw}(p) = 1 - (2/3)p.",
          "Metric-Separation Rule: Never compare F_state(p) against 1 - p (creates a spurious false advantage). Because F_state - F_{state,raw} = (2/3)(F_e - F_{e,raw}), the gain threshold p* is strictly invariant."
        ]
      },
      "pt-br": {
        title: "Slide 14 — Limiares Analíticos de Ganho e Consistência de Métricas",
        subtitle: "Separação estrita entre Fidelidade de Emaranhamento (1 - p) e Fidelidade Média de Estado de Haar (1 - 2p/3)",
        bullets: [
          "Fidelidade de Emaranhamento (F_e): Igual à probabilidade de sobrevivência lógica P_I(p); referência bruta F_{e,raw}(p) = 1 - p.",
          "Fidelidade Média de Estado de Haar (F_state): Média uniforme em SU(2) (ou nos 6 estados cardinais) fornece F_state(p) = 1 - (2/3)(1 - F_e(p)); referência bruta F_{state,raw}(p) = 1 - (2/3)p.",
          "Regra de Separação de Métricas: Nunca compare F_state(p) com 1 - p (vantagem espúria falsa). Como F_state - F_{state,raw} = (2/3)(F_e - F_{e,raw}), o limiar p* é estritamente invariante."
        ]
      },
      equationsHtml: "F<sub>state</sub>(p) = 1 − (2/3)(1 − F<sub>e</sub>(p)) &nbsp;|&nbsp; F<sub>state</sub>(p) &gt; 1 − (2/3)p ⟺ F<sub>e</sub>(p) &gt; 1 − p",
      demoPreset: { target_tab: "curves", code: "five_qubit", channel: "depolarizing", state_preset: "cardinal_avg", metric: "state" }
    },
    {
      index: 15,
      topicId: 7,
      en: {
        title: "Slide 15 — Exact brentq Gain Thresholds (p*) Across All Codes",
        subtitle: "Five positive-gain regimes where QECC or DFS outperforms the unprotected qubit",
        bullets: [
          "bit_flip_3 / bit_flip & phase_flip_3 / phase_flip: p* = 0.5000000000 (0 < p < 0.5).",
          "five_qubit / depolarizing: p* = 0.1376275643 (~13.76%, using the 1024-term enumerator [1, 15, 0, 60, 135, 45]).",
          "five_qubit / bit_flip & phase_flip: p* = 0.1311231479 (~13.11%, using [1, 5, 0, 0, 0, 0]).",
          "dfs_2 / collective_dephasing: p* = 1.0000000000 (0 < p < 1.0); all mismatched pairs have p* = 0.0."
        ]
      },
      "pt-br": {
        title: "Slide 15 — Limiares de Ganho Exatos (p* via brentq) em Todos os Códigos",
        subtitle: "Cinco regimes de ganho positivo nos quais o QECC ou DFS supera o qubit não protegido",
        bullets: [
          "bit_flip_3 / bit_flip e phase_flip_3 / phase_flip: p* = 0.5000000000 (0 < p < 0.5).",
          "five_qubit / depolarizing: p* = 0.1376275643 (~13.76%, pelo enumerador de 1024 termos [1, 15, 0, 60, 135, 45]).",
          "five_qubit / bit_flip e phase_flip: p* = 0.1311231479 (~13.11%, por [1, 5, 0, 0, 0, 0]).",
          "dfs_2 / collective_dephasing: p* = 1.0000000000 (0 < p < 1.0); pares descasados possuem p* = 0.0."
        ]
      },
      equationsHtml: "F<sub>e</sub><sup>(5,dep)</sup>(p) = (1−p)<sup>5</sup> + 15(1−p)<sup>4</sup>(p/3) + 60(1−p)<sup>2</sup>(p/3)<sup>3</sup> + 135(1−p)(p/3)<sup>4</sup> + 45(p/3)<sup>5</sup>",
      demoPreset: { target_tab: "curves", code: "five_qubit", channel: "depolarizing", state_preset: "+i", metric: "entanglement" }
    },
    {
      index: 16,
      topicId: 7,
      en: {
        title: "Slide 16 — Monte Carlo Sweeps (18,000 Shots on RTX 5080 torch_cuda)",
        subtitle: "231 records across 21 error rates p ∈ [0, 0.5] (results/data/fidelity_sweeps.json)",
        bullets: [
          "At p = 0.0, every QECC restores the input state within machine precision (|F_sim - 1.0| ≤ 2.776 × 10⁻¹⁵).",
          "Across 180 p > 0 Monte Carlo points (160 non-zero-variance points + 20 exact F = 1.0 dfs_2 / collective_dephasing points): max std_error = 0.003727, max |z| = 3.3430.",
          "Confirms seamless statistical agreement between torch_cuda Monte Carlo sweeps and exact analytical polynomials."
        ]
      },
      "pt-br": {
        title: "Slide 16 — Varreduras Monte Carlo (18.000 Amostras na RTX 5080 torch_cuda)",
        subtitle: "231 registros em 21 taxas de erro p ∈ [0, 0.5] (results/data/fidelity_sweeps.json)",
        bullets: [
          "Em p = 0.0, todos os QECCs recuperam o estado inicial em precisão de máquina (|F_sim - 1.0| ≤ 2.776 × 10⁻¹⁵).",
          "Nos 180 pontos Monte Carlo com p > 0 (160 pontos com variância não nula + 20 pontos exatos F = 1.0 de dfs_2 / collective_dephasing): std_error máx. = 0.003727, |z| máx. = 3.3430.",
          "Confirma concordância estatística completa entre as varreduras em torch_cuda e os polinômios analíticos exatos."
        ]
      },
      equationsHtml: "|F<sub>sim</sub>(0) − 1.0| ≤ 2.776 × 10<sup>−15</sup> &nbsp;|&nbsp; max SE = 0.003727 &nbsp;|&nbsp; max |z| = 3.3430",
      demoPreset: { target_tab: "curves", code: "bit_flip_3", channel: "bit_flip", state_preset: "0", metric: "entanglement" }
    },
    {
      index: 17,
      topicId: 7,
      en: {
        title: "Slide 17 — Hybrid Backend Architecture & eigenkit Evaluation",
        subtitle: "Three swappable QECCBackend engines sharing big-endian qubit-0 MSB semantics",
        bullets: [
          "eigenkit v1.3.0 Audit (docs/eigenkit_evaluation.md): Clean big-endian unitary statevector engine, but lacks mid-circuit measure, c_if feed-forward, stochastic Pauli/Kraus noise (sample() raises NotImplementedError), partial_trace, and sm_120 CUDA kernels.",
          "numpy (NumPyCPUBackend): Vectorized (B, 2^n) complex128 CPU reference simulator.",
          "torch_cuda (TorchCUDABackend): Native torch==2.14.1+cu130 (sm_120) GPU engine on RTX 5080.",
          "eigenkit_unitary (EigenkitUnitaryAdapter): Translates Y → Sdg-X-S, CZ → MCZ/H-CNOT-H for unitary cross-checks (p = 0 & deterministic errors; stochastic_supported = false)."
        ]
      },
      "pt-br": {
        title: "Slide 17 — Arquitetura Híbrida de Backends e Avaliação do eigenkit",
        subtitle: "Três motores QECCBackend intercambiáveis com convenção big-endian (qubit 0 MSB)",
        bullets: [
          "Auditoria do eigenkit v1.3.0 (docs/pt-br/eigenkit_evaluation.md): Simulador unitário big-endian limpo, mas carece de measure no meio do circuito, c_if, ruído estocástico (sample() lança NotImplementedError), partial_trace e kernels CUDA sm_120.",
          "numpy (NumPyCPUBackend): Simulador de referência vetorizado em CPU complex128.",
          "torch_cuda (TorchCUDABackend): Motor GPU nativo torch==2.14.1+cu130 (sm_120) na RTX 5080.",
          "eigenkit_unitary (EigenkitUnitaryAdapter): Traduz Y → Sdg-X-S, CZ → MCZ/H-CNOT-H para verificação cruzada unitária (p = 0 e erros determinísticos; stochastic_supported = false)."
        ]
      },
      equationsHtml: "Y = S X S<sup>†</sup> &nbsp;|&nbsp; CZ = H<sub>t</sub> · CNOT<sub>c→t</sub> · H<sub>t</sub>",
      demoPreset: { target_tab: "simulator", code: "five_qubit", channel: "depolarizing", state_preset: "+i", error_mode: "pauli_string", pauli_string: "XIIII", stage_key: "decoded" }
    },
    {
      index: 18,
      topicId: 7,
      en: {
        title: "Slide 18 — RTX 5080 CUDA Benchmarks & Unitary Parity",
        subtitle: "24.40× GPU speedup on five_qubit (100k shots) with ≤ 1.06e-15 unitary parity",
        bullets: [
          "Stochastic Parity (numpy vs. torch_cuda, p = 0.05, |+i⟩): 9/9 PASS.",
          "Unitary Parity (eigenkit_unitary vs. numpy/torch_cuda): 6/6 PASS, Max |Δρ| ≤ 1.06e-15 overall (≤ 1.52e-17 eigenkit_unitary vs. numpy).",
          "Timings (100,000 shots): bit_flip_3 (0.179 s CPU vs 0.022 s GPU, 7.97×), phase_flip_3 (0.335 s vs 0.026 s, 12.76×), five_qubit (10.081 s vs 0.413 s, 24.40×)."
        ]
      },
      "pt-br": {
        title: "Slide 18 — Benchmarks CUDA na RTX 5080 e Paridade Unitária",
        subtitle: "Aceleração de 24.40× em GPU no five_qubit (100 mil shots) com paridade unitária ≤ 1.06e-15",
        bullets: [
          "Paridade Estocástica (numpy vs. torch_cuda, p = 0.05, |+i⟩): 9/9 APROVADO.",
          "Paridade Unitária (eigenkit_unitary vs. numpy/torch_cuda): 6/6 APROVADO, máx. |Δρ| ≤ 1.06e-15 no geral (≤ 1.52e-17 entre eigenkit_unitary e numpy).",
          "Tempos (100.000 shots): bit_flip_3 (0.179 s CPU vs 0.022 s GPU, 7.97×), phase_flip_3 (0.335 s vs 0.026 s, 12.76×), five_qubit (10.081 s vs 0.413 s, 24.40×)."
        ]
      },
      equationsHtml: "Max |Δρ| ≤ 1.06 × 10<sup>−15</sup> &nbsp;|&nbsp; RTX 5080 Speedup: 24.40× (100k shots, [[5,1,3]])",
      demoPreset: { target_tab: "simulator", code: "five_qubit", channel: "depolarizing", state_preset: "+i", error_mode: "stochastic_sample", p: 0.05, stage_key: "decoded" }
    },
    {
      index: 19,
      topicId: 7,
      en: {
        title: "Slide 19 — Conclusions & Reproducibility Summary",
        subtitle: "Unified theoretical derivations, live simulation, and deterministic verification",
        bullets: [
          "Theory & Simulation Unified: Knill–Laflamme (≤ 1.972e-31), 1024-term enumerator, brentq thresholds (0.5, 0.137628, 0.131123, 1.0), 18k-shot sweeps, and 24.40× RTX 5080 CUDA speedups agree seamlessly.",
          "Core Takeaway: Always separate Entanglement Fidelity (1 - p) from Haar State Fidelity (1 - 2p/3) when evaluating QECC break-even thresholds.",
          "CLI Reproducibility: scripts/verify_analytical_thresholds.py, scripts/run_sweeps.py, scripts/run_benchmarks.py, pytest tests/unit tests/e2e -v, and qecc serve."
        ]
      },
      "pt-br": {
        title: "Slide 19 — Conclusões e Resumo de Reprodutibilidade",
        subtitle: "Derivações teóricas, simulação ao vivo e verificação determinística unificadas",
        bullets: [
          "Teoria e Simulação Unificadas: Knill–Laflamme (≤ 1.972e-31), enumerador de 1024 termos, limiares via brentq (0.5, 0.137628, 0.131123, 1.0), varreduras de 18k shots e aceleração de 24.40× na RTX 5080 concordam plenamente.",
          "Mensagem Principal: Sempre separe a Fidelidade de Emaranhamento (1 - p) da Fidelidade Média de Estado de Haar (1 - 2p/3) ao avaliar limiares de ganho em QECC.",
          "Reprodutibilidade via CLI: scripts/verify_analytical_thresholds.py, scripts/run_sweeps.py, scripts/run_benchmarks.py, pytest tests/unit tests/e2e -v e qecc serve."
        ]
      },
      equationsHtml: "F<sub>e</sub>(p) &gt; 1 − p &nbsp;⟺&nbsp; F<sub>state</sub>(p) &gt; 1 − (2/3)p &nbsp;&nbsp;(exact p* invariance)",
      demoPreset: { target_tab: "curves", code: "dfs_2", channel: "collective_dephasing", state_preset: "+i", metric: "entanglement" }
    }
  ];

  function SeminarController(rootEl, onApplyDemo) {
    this.rootEl = rootEl;
    this.onApplyDemo = onApplyDemo;
    this.currentIndex = this._parseHashSlideIndex();
    this._bindHashChange();
    this.render();
  }

  SeminarController.prototype._parseHashSlideIndex = function () {
    const h = typeof global.location !== "undefined" ? String(global.location.hash || "") : "";
    const m = /^#slide-(\d+)$/.exec(h);
    if (!m) return 0;
    const idx = parseInt(m[1], 10);
    if (!Number.isInteger(idx) || idx < 0 || idx >= SEMINAR_SLIDES.length) {
      return 0;
    }
    return idx;
  };

  SeminarController.prototype._bindHashChange = function () {
    const self = this;
    global.addEventListener("hashchange", function () {
      self.goToSlide(self._parseHashSlideIndex(), false);
    });
  };

  SeminarController.prototype.goToSlide = function (idx, updateHash) {
    const clamped = Math.max(0, Math.min(SEMINAR_SLIDES.length - 1, Number(idx) || 0));
    this.currentIndex = clamped;
    if (updateHash !== false && typeof global.history !== "undefined" && global.history.replaceState) {
      try {
        global.history.replaceState(null, "", "#slide-" + clamped);
      } catch (_err) {
        // Ignore restricted history API errors
      }
    }
    this.render();
  };

  SeminarController.prototype.render = function () {
    if (!this.rootEl) return;
    const self = this;
    const i18n = global.QECC_I18N;
    const lang = i18n ? i18n.getLanguage() : "en";
    const t = i18n ? i18n.t : function (k) { return k; };

    const slide = SEMINAR_SLIDES[this.currentIndex] || SEMINAR_SLIDES[0];
    const loc = slide[lang] || slide.en;

    this.rootEl.textContent = "";

    // 1. Topic Tabs (1..7)
    const topicsBar = document.createElement("div");
    topicsBar.className = "seminar-topics";
    topicsBar.setAttribute("role", "tablist");
    SEMINAR_TOPICS.forEach(function (tp) {
      const btn = document.createElement("button");
      btn.type = "button";
      const active = slide.topicId === tp.topicId;
      btn.className = "topic-tab" + (active ? " active" : "");
      btn.setAttribute("role", "tab");
      btn.setAttribute("aria-selected", active ? "true" : "false");
      btn.setAttribute("data-testid", "seminar-topic-" + tp.topicId);
      btn.setAttribute("data-topic", String(tp.topicId));
      btn.textContent = tp[lang] || tp.en;
      btn.addEventListener("click", function () {
        self.goToSlide(tp.firstSlide, true);
      });
      topicsBar.appendChild(btn);
    });
    this.rootEl.appendChild(topicsBar);

    // 2. Navigation Bar (Prev / Select / Next)
    const navBar = document.createElement("div");
    navBar.className = "seminar-nav-bar";

    const prevBtn = document.createElement("button");
    prevBtn.type = "button";
    prevBtn.className = "btn btn-secondary";
    prevBtn.setAttribute("data-testid", "seminar-prev");
    prevBtn.disabled = this.currentIndex <= 0;
    prevBtn.textContent = t("seminar.prev");
    prevBtn.addEventListener("click", function () {
      self.goToSlide(self.currentIndex - 1, true);
    });

    const selectWrap = document.createElement("div");
    selectWrap.className = "seminar-select-wrap";
    const slideSel = document.createElement("select");
    slideSel.setAttribute("data-testid", "seminar-slide-select");
    slideSel.setAttribute("aria-label", t("seminar.slide_label"));
    SEMINAR_SLIDES.forEach(function (sl) {
      const opt = document.createElement("option");
      opt.value = String(sl.index);
      const slLoc = sl[lang] || sl.en;
      opt.textContent = slLoc.title;
      slideSel.appendChild(opt);
    });
    slideSel.value = String(this.currentIndex);
    slideSel.addEventListener("change", function () {
      self.goToSlide(parseInt(slideSel.value, 10), true);
    });
    selectWrap.appendChild(slideSel);

    const nextBtn = document.createElement("button");
    nextBtn.type = "button";
    nextBtn.className = "btn btn-secondary";
    nextBtn.setAttribute("data-testid", "seminar-next");
    nextBtn.disabled = this.currentIndex >= SEMINAR_SLIDES.length - 1;
    nextBtn.textContent = t("seminar.next");
    nextBtn.addEventListener("click", function () {
      self.goToSlide(self.currentIndex + 1, true);
    });

    navBar.appendChild(prevBtn);
    navBar.appendChild(selectWrap);
    navBar.appendChild(nextBtn);
    this.rootEl.appendChild(navBar);

    // 3. Slide Card
    const card = document.createElement("div");
    card.className = "seminar-card";

    const cardHead = document.createElement("div");
    cardHead.className = "seminar-card-header";
    const slideBadge = document.createElement("span");
    slideBadge.className = "badge badge-engine";
    slideBadge.textContent = slide.index === 0 ? t("seminar.title_badge") : t("seminar.slide_counter", { n: slide.index });
    const topicObj = SEMINAR_TOPICS.find(function (tp) { return tp.topicId === slide.topicId; });
    const topicBadge = document.createElement("span");
    topicBadge.className = "badge badge-offline";
    topicBadge.textContent = topicObj ? (topicObj[lang] || topicObj.en) : (t("seminar.topic_label") + " " + slide.topicId);
    cardHead.appendChild(slideBadge);
    cardHead.appendChild(topicBadge);
    card.appendChild(cardHead);

    const h3 = document.createElement("h3");
    h3.textContent = loc.title;
    card.appendChild(h3);

    const sub = document.createElement("p");
    sub.className = "seminar-subtitle";
    sub.textContent = loc.subtitle;
    card.appendChild(sub);

    const ul = document.createElement("ul");
    ul.className = "seminar-bullets";
    loc.bullets.forEach(function (b) {
      const li = document.createElement("li");
      li.textContent = b;
      ul.appendChild(li);
    });
    card.appendChild(ul);

    if (slide.equationsHtml) {
      const eqLabel = document.createElement("div");
      eqLabel.style.fontSize = "0.78rem";
      eqLabel.style.color = "#9fb0cb";
      eqLabel.style.marginBottom = "4px";
      eqLabel.textContent = t("seminar.equations_label");
      card.appendChild(eqLabel);

      const eqBox = document.createElement("div");
      eqBox.className = "seminar-equations";
      eqBox.innerHTML = slide.equationsHtml;
      card.appendChild(eqBox);
    }

    if (slide.demoPreset) {
      const demoBar = document.createElement("div");
      demoBar.className = "seminar-demo-bar";

      const presetSummary = document.createElement("span");
      presetSummary.className = "mono";
      presetSummary.style.fontSize = "0.78rem";
      presetSummary.style.color = "#9fb0cb";
      presetSummary.textContent = t("seminar.preset_summary") + ": " + slide.demoPreset.code + " / " + slide.demoPreset.channel;

      const applyBtn = document.createElement("button");
      applyBtn.type = "button";
      applyBtn.className = "btn btn-primary";
      applyBtn.setAttribute("data-testid", "seminar-apply-demo");
      applyBtn.setAttribute("data-code", slide.demoPreset.code);
      applyBtn.textContent = t("seminar.apply_demo");
      applyBtn.addEventListener("click", function () {
        if (typeof self.onApplyDemo === "function") {
          self.onApplyDemo(slide.demoPreset);
        }
      });

      demoBar.appendChild(presetSummary);
      demoBar.appendChild(applyBtn);
      card.appendChild(demoBar);
    }

    this.rootEl.appendChild(card);
  };

  global.QECC_SEMINAR = {
    SEMINAR_TOPICS: SEMINAR_TOPICS,
    SEMINAR_SLIDES: SEMINAR_SLIDES,
    SeminarController: SeminarController
  };
})(window);
