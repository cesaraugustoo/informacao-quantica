(function (global) {
  "use strict";

  const SUPPORTED_LANGS = ["en", "pt-br"];
  const STORAGE_KEY = "qecc_lang";

  const STRINGS = {
    en: {
      "app.title": "QECC & DFS Interactive Companion",
      "app.subtitle": "Quantum Error Correction Codes, Stabilizers, Coherence & Gain-Threshold Explorer",
      "app.badge.engine": "Backed by qecc Python Engine (Big-Endian)",
      "app.badge.offline": "Offline Scientific SPA",
      "lang.label": "Language",
      "lang.en": "EN",
      "lang.pt_br": "PT-BR",
      "nav.simulator": "Step Simulator & Bloch",
      "nav.curves": "Analytical & MC Curves (p*)",
      "nav.theory": "Stabilizers & Knill–Laflamme",
      "nav.seminar": "Seminar Companion (20 Slides)",
      "nav.aria_label": "Primary sections",

      "code.label": "Quantum Code",
      "code.raw.name": "Unprotected Raw Qubit (raw, [[1,1,1]])",
      "code.raw.short": "Raw (1 qubit, no protection)",
      "code.raw.desc": "Single unprotected physical qubit baseline (F_e = 1 - p, F_state = 1 - 2p/3).",
      "code.bit_flip_3.name": "3-Qubit Bit-Flip Code (bit_flip_3, [[3,1,1]])",
      "code.bit_flip_3.short": "Bit-Flip ([[3,1,1]], 3 data + 2 ancilla)",
      "code.bit_flip_3.desc": "Repetition code encoding |0_L⟩ = |000⟩, |1_L⟩ = |111⟩. Corrects any single-qubit X error via stabilizers ZZI, IZZ; vulnerable to Z phase errors and weight-2 X aliasing.",
      "code.phase_flip_3.name": "3-Qubit Phase-Flip Code (phase_flip_3, [[3,1,1]])",
      "code.phase_flip_3.short": "Phase-Flip ([[3,1,1]], 3 data + 2 ancilla)",
      "code.phase_flip_3.desc": "Hadamard-dual (H^⊗3) code encoding |0_L⟩ = |+++⟩, |1_L⟩ = |---⟩. Corrects any single-qubit Z error via stabilizers XXI, IXX; fails against collective ZZZ = X_L dephasing.",
      "code.five_qubit.name": "[[5,1,3]] Laflamme Perfect Code (five_qubit)",
      "code.five_qubit.short": "Perfect [[5,1,3]] (5 data + 4 ancilla)",
      "code.five_qubit.desc": "Smallest code saturating the Quantum Hamming Bound (2(1+15) = 32 = 2^5). Uses 4 cyclic stabilizers (XZZXI, IXZZX, XIXZZ, ZXIXZ) to correct any single-qubit X, Y, or Z error bijectively across 16 syndromes.",
      "code.dfs_2.name": "Passive Decoherence-Free Subspace (dfs_2)",
      "code.dfs_2.short": "Passive DFS (2 data + 0 ancilla)",
      "code.dfs_2.desc": "Symmetry-protected odd-parity subspace span{|01⟩, |10⟩} with zero ancillas and zero syndrome measurements. Invariant (F = 1.0) under collective dephasing Z⊗Z and Rz(φ)⊗Rz(φ), but vulnerable to symmetry-breaking independent phase flips.",

      "channel.label": "Noise Channel",
      "channel.bit_flip": "Bit-Flip (bit_flip — independent X)",
      "channel.phase_flip": "Phase-Flip (phase_flip — independent Z)",
      "channel.depolarizing": "Depolarizing (depolarizing — X/Y/Z at p/3)",
      "channel.collective_dephasing": "Collective Dephasing (collective_dephasing — correlated Z^⊗n)",

      "error_mode.label": "Stepper Error Mode",
      "error_mode.pauli_string": "Deterministic Pauli String (I/X/Y/Z)",
      "error_mode.single_qubit_pauli": "Single-Qubit Pauli (q, I/X/Y/Z)",
      "error_mode.stochastic_sample": "Stochastic Channel Sample (channel, p)",
      "error_mode.su2_rotation": "Continuous SU(2) Rotation U(θ, n̂)",
      "error_mode.coherent_collective": "Coherent Collective Dephasing Rz(φ)^⊗n",

      "state.label": "Input Logical State |ψ⟩",
      "state.0": "|0⟩ — Computational zero (r = (0, 0, +1))",
      "state.1": "|1⟩ — Computational one (r = (0, 0, -1))",
      "state.plus": "|+⟩ — Hadamard + (r = (+1, 0, 0))",
      "state.minus": "|-⟩ — Hadamard - (r = (-1, 0, 0))",
      "state.plus_i": "|+i⟩ — Y-eigenstate +i (r = (0, +1, 0), |ρ₀₁| = 0.5)",
      "state.minus_i": "|-i⟩ — Y-eigenstate -i (r = (0, -1, 0), |ρ₀₁| = 0.5)",
      "state.cardinal_avg": "cardinal_avg — 6-Cardinal Haar 2-Design Average",
      "state.bloch": "Custom Bloch Angles (θ, φ)",
      "state.custom": "Custom Amplitudes (α, β)",
      "state.cardinal_stepper_note": "Note: cardinal_avg averages all 6 cardinal states in Monte Carlo simulation; step-by-step inspection displays |+i⟩.",

      "param.theta": "Polar angle θ (rad)",
      "param.phi_state": "Azimuthal angle φ_state (rad)",
      "param.alpha_re": "Re(α)",
      "param.alpha_im": "Im(α)",
      "param.beta_re": "Re(β)",
      "param.beta_im": "Im(β)",
      "param.normalize_hint": "Must satisfy |α|² + |β|² = 1 (tol = 1e-6).",

      "ctrl.p": "Physical error rate p ([0, 1])",
      "ctrl.phi": "Coherent collective phase φ (rad)",
      "ctrl.pauli_string": "Pauli string ({n} chars)",
      "ctrl.error_qubit": "Target data qubit q",
      "ctrl.error_op": "Pauli operator (I/X/Y/Z)",
      "ctrl.su2_qubit": "SU(2) target qubit q",
      "ctrl.su2_theta": "Rotation angle θ (rad)",
      "ctrl.su2_nx": "Axis n_x",
      "ctrl.su2_ny": "Axis n_y",
      "ctrl.su2_nz": "Axis n_z",
      "ctrl.syndrome_branch": "Syndrome branch projection",
      "ctrl.syndrome_auto": "Highest-probability branch (auto)",
      "ctrl.num_shots": "Monte Carlo shots",
      "ctrl.seed": "PRNG seed",
      "ctrl.backend": "Simulation backend",

      "backend.auto": "auto (GPU torch_cuda if available, else numpy)",
      "backend.numpy": "numpy (NumPyCPUBackend, complex128 CPU)",
      "backend.torch_cuda": "torch_cuda (TorchCUDABackend, RTX 5080 sm_120)",
      "backend.eigenkit_unitary": "eigenkit_unitary (EigenkitUnitaryAdapter, unitary only)",
      "backend.unavailable": "Unavailable on this host",

      "btn.run_step": "Run Step Evolution",
      "btn.run_sim": "Run Monte Carlo Cycle",
      "btn.run_sweep": "Update Curves",
      "btn.resample_seed": "New Random Sample",
      "btn.normalize": "Normalize",

      "stage.section_title": "Step-by-Step State Evolution Pipeline",
      "stage.initial.title": "1. Initial State (initial)",
      "stage.initial.desc": "Logical input |ψ⟩ enters on qubit 0 (MSB in big-endian convention); remaining data qubits and ancillas start in |0⟩.",
      "stage.encoded.title": "2. Encoding (encoded)",
      "stage.encoded.desc": "Entangles logical information across the code subspace (|ψ⟩ → α|0_L⟩ + β|1_L⟩) via encode_ops(), respecting the no-cloning theorem.",
      "stage.noisy.title": "3. Noise Injection / Discretization (noisy)",
      "stage.noisy.desc": "Applies the physical error (discrete Pauli string, stochastic sample, continuous SU(2) rotation, or coherent Rz(φ)^⊗n). Continuous rotations create a superposition of Pauli error branches.",
      "stage.syndrome_extracted.title": "4. Ancilla Syndrome Extraction (syndrome_extracted)",
      "stage.syndrome_extracted.desc": "Non-destructive phase kickback (H · Λ(S_k) · H) maps stabilizer parity onto ancillas without leaking α, β. Projective ancilla measurement collapses continuous errors into a discrete Pauli coset s.",
      "stage.syndrome_extracted.desc_dfs": "dfs_2 has zero ancillas (n_ancilla = 0) and performs no syndrome extraction; protection comes passively from collective symmetry.",
      "stage.recovered.title": "5. Conditional Recovery (recovered)",
      "stage.recovered.desc": "Classical conditional feed-forward applies recovery_ops(syndrome) based on the measured ancilla bitstring, returning the state to the code subspace (unless uncorrectable aliasing occurred).",
      "stage.recovered.desc_dfs": "dfs_2 requires no recovery operations (recovery_ops(\"\") = []); collective Z⊗Z or Rz(φ)⊗Rz(φ) acts as an unobservable scalar global phase.",
      "stage.decoded.title": "6. Decoding & Partial Trace (decoded)",
      "stage.decoded.desc": "Unitary decode_ops() disentangles the logical qubit back to qubit 0, followed by partial_trace_to_qubit0 to yield the final 2×2 density matrix ρ_out, Bloch vector, |ρ₀₁|, and fidelity.",

      "stage.metric.fidelity": "Fidelity vs. Input",
      "stage.metric.subspace_prob": "Code Subspace Prob.",
      "stage.metric.gates_applied": "Gates Applied in Stage",
      "stage.metric.none": "None (identity / no-op)",
      "stage.metric.syndrome_probs": "Ancilla Syndrome Probabilities (click to select branch)",
      "stage.metric.selected_syndrome": "Collapsed Syndrome",
      "stage.metric.dfs_empty_syndrome": "\"\" (zero ancillas — passive DFS)",
      "stage.metric.pauli_coeffs": "Pauli Basis Expansion E = Σ c_P P",
      "stage.metric.nonzero_amplitudes": "Top Computational Basis Amplitudes",
      "stage.metric.logical_bloch": "Logical Bloch Vector",
      "stage.metric.abs_rho01": "Logical |ρ₀₁|",

      "circuit.title": "Quantum Circuit Diagram (Big-Endian: q₀ MSB)",
      "circuit.data_qubit": "Data q{idx}",
      "circuit.ancilla_qubit": "Ancilla a{idx} (q{q})",
      "circuit.noise_box": "Noise: {label}",
      "circuit.measure": "Measure",
      "circuit.recovery_box": "Conditional R({syndrome})",
      "circuit.stage_encode": "Encode",
      "circuit.stage_noise": "Noise",
      "circuit.stage_syndrome": "Syndrome",
      "circuit.stage_recovery": "Recovery",
      "circuit.stage_decode": "Decode",

      "bloch.title": "3D Bloch Sphere & Trajectory",
      "bloch.reset": "Reset View",
      "bloch.legend.input": "Input |ψ⟩",
      "bloch.legend.raw": "Unprotected raw",
      "bloch.legend.qecc": "Active QECC ({code})",
      "bloch.legend.dfs": "Passive dfs_2",
      "bloch.legend.stage": "Active Stage",
      "bloch.hint_drag": "Drag to rotate 3D view",

      "coherence.title": "Off-Diagonal Coherence |ρ₀₁| & 3-Way Comparison",
      "coherence.rho_matrix": "Reduced Density Matrix ρ_out (2×2)",
      "coherence.abs_rho01": "Off-Diagonal Coherence |ρ₀₁|",
      "coherence.bloch_vec": "Bloch Vector (r_x, r_y, r_z)",
      "coherence.fidelity": "Fidelity F",
      "coherence.compare.raw": "Unprotected Raw",
      "coherence.compare.qecc": "Selected Code ({code})",
      "coherence.compare.dfs": "Passive dfs_2",
      "coherence.zero_basis_note": "Note: Computational basis states |0⟩ and |1⟩ have zero initial coherence (|ρ₀₁| = 0). Select |+i⟩ (ideal |ρ₀₁| = 0.5) to inspect phase-coherence decay and protection.",

      "sim.mc_summary": "Ensemble Monte Carlo Result ({backend})",
      "sim.mean_fidelity": "Mean Fidelity",
      "sim.std_error": "Standard Error (SE)",
      "sim.ana_ent": "Analytical F_e",
      "sim.ana_state": "Analytical F_state",
      "sim.elapsed": "Elapsed Time",
      "sim.syndrome_counts": "Syndrome Counts",
      "sim.raw_baseline": "Raw Baseline Fidelity",

      "curves.title": "Analytical & Monte Carlo Fidelity vs. Error Rate (p)",
      "curves.metric.label": "Fidelity Metric",
      "curves.metric.entanglement": "Entanglement Fidelity F_e(p) (baseline 1 - p)",
      "curves.metric.state": "Haar-Averaged State Fidelity F_state(p) (baseline 1 - 2p/3)",
      "curves.metric_notice.title": "Strict Metric-Separation Rule",
      "curves.metric_notice.entanglement": "Comparing Entanglement Fidelity F_e(p) = P_I(p) strictly against raw baseline F_e,raw(p) = 1 - p. Never compare F_state against 1 - p (spurious advantage mismatch).",
      "curves.metric_notice.state": "Comparing Haar-Averaged State Fidelity F_state(p) = 1 - (2/3)(1 - F_e(p)) strictly against raw baseline F_state,raw(p) = 1 - (2/3)p. Gain threshold p* is invariant across both metrics.",
      "curves.axis.p": "Physical Error Rate p",
      "curves.axis.fidelity_ent": "Entanglement Fidelity F_e(p)",
      "curves.axis.fidelity_state": "Haar State Fidelity F_state(p)",
      "curves.axis.delta_f": "Net Fidelity Gain ΔF vs. Unprotected Raw",
      "curves.axis.abs_rho01": "Coherence |ρ₀₁| on |+i⟩",
      "curves.raw_baseline_ent": "Unprotected Raw (1 - p)",
      "curves.raw_baseline_state": "Unprotected Raw (1 - 2p/3)",
      "curves.mc_source.cached": "Cached 18k-shot torch_cuda dataset (fidelity_sweeps.json)",
      "curves.mc_source.live": "Live Monte Carlo Sweep ({shots} shots, {backend})",
      "curves.run_live_mc": "Run Live Monte Carlo Sweep",
      "curves.delta_chart.title": "Net Entanglement / State Fidelity Gain (ΔF)",
      "curves.coh_chart.title": "Analytical Coherence |ρ₀₁|(p) on |+i⟩",
      "curves.p_star.title": "Exact brentq Gain Thresholds (p*)",
      "curves.p_star.col_code": "Code",
      "curves.p_star.col_channel": "Channel",
      "curves.p_star.col_p_star": "Exact p*",
      "curves.p_star.col_region": "Advantage Region",
      "curves.p_star.none": "None (0.0000)",
      "curves.p_min": "p_min",
      "curves.p_max": "p_max",
      "curves.num_p": "Grid points",

      "theory.title": "Algebraic Structure, Syndrome Lookup & Knill–Laflamme Verification",
      "theory.stabilizers": "Stabilizer Generators (S_k)",
      "theory.stabilizers.none": "None — dfs_2 uses passive symmetry (span{|01⟩, |10⟩}) with no stabilizer ancillas.",
      "theory.logical_ops": "Logical Operators",
      "theory.syndrome_table": "Syndrome Lookup Table",
      "theory.syndrome_table.empty": "Empty — dfs_2 performs no ancilla syndrome extraction.",
      "theory.syndrome_col": "Syndrome",
      "theory.recovery_col": "Recovery Pauli",
      "theory.target_col": "Action",
      "theory.no_error": "No error (identity)",
      "theory.correct_pauli": "Correct {op}",
      "theory.aliasing.title": "Weight-2 Aliasing & Uncorrectable Error Examples",
      "theory.aliasing.error": "Injected Error",
      "theory.aliasing.syndrome": "Measured Syndrome",
      "theory.aliasing.recovery": "Recovery Applied",
      "theory.aliasing.residual": "Residual Operator",
      "theory.aliasing.effect": "Logical Effect",
      "theory.aliasing.note_col": "Explanation",
      "theory.aliasing.note_pauli": "Weight-2 error {error} aliases syndrome {syndrome}; recovery {recovery} leaves logical operator {effect} ({residual}).",
      "theory.aliasing.note_dfs_z": "Symmetry-breaking single-qubit phase error {error} induces undetected logical {effect} phase flip.",
      "theory.aliasing.note_dfs_x": "Bit-flip {error} leaks state out of decoherence-free subspace span{|01⟩, |10⟩}.",
      "theory.kl.title": "Knill–Laflamme Conditions: ⟨i_L| E_a† E_b |j_L⟩ = C_ab δ_ij",
      "theory.kl.status": "Verification Status",
      "theory.kl.passed": "Verified (tol = 1e-12)",
      "theory.kl.failed": "Violated",
      "theory.kl.max_violation": "Max Violation",
      "theory.kl.max_diag_diff": "Max Diagonal Diff",
      "theory.kl.max_off_diag": "Max Off-Diagonal",
      "theory.kl.error_basis": "Verified Error Basis E",
      "theory.kl.c_matrix": "Hermitian Matrix C_ab",
      "theory.enum.title": "[[5,1,3]] Exact 1024-Term Weight Enumerator (4^5 = 1024)",
      "theory.enum.subtitle": "Exhaustive coset counts after syndrome recovery across all 1024 5-qubit Pauli strings.",
      "theory.enum.weight": "Hamming Weight w",
      "theory.enum.coset_i": "Coset I_L S",
      "theory.enum.coset_x": "Coset X_L S",
      "theory.enum.coset_y": "Coset Y_L S",
      "theory.enum.coset_z": "Coset Z_L S",
      "theory.enum.total": "Total C(5,w)·3^w",
      "theory.enum.pure_x": "Pure-X (I coset)",
      "theory.enum.pure_z": "Pure-Z (I coset)",
      "theory.enum.sum": "Sum",

      "seminar.title": "Graduate Seminar & Live Presentation Companion",
      "seminar.topic_label": "Topic",
      "seminar.slide_label": "Slide",
      "seminar.title_badge": "Title Slide",
      "seminar.slide_counter": "Slide {n} / 19",
      "seminar.prev": "Previous",
      "seminar.next": "Next",
      "seminar.apply_demo": "Load Live Demo in Simulator",
      "seminar.demo_loaded": "Loaded slide preset into simulator: {code}",
      "seminar.equations_label": "Key Equations",
      "seminar.preset_summary": "Linked Simulator Preset",

      "status.loading": "Computing via qecc engine...",
      "status.ready": "Ready",
      "err.invalid_p": "Error rate p must be a finite number in [0, 1].",
      "err.invalid_phi": "Collective phase φ must be a finite real number.",
      "err.invalid_bloch_angles": "Bloch angles (θ, φ_state) must be finite real numbers.",
      "err.invalid_pauli": "Pauli string must have length {n} using only I, X, Y, Z.",
      "err.invalid_custom_norm": "Custom amplitudes must be unit-normalized (|α|² + |β|² = 1).",
      "err.invalid_su2_axis": "SU(2) rotation axis (n_x, n_y, n_z) must be non-zero.",
      "err.invalid_shots": "Shot count must be an integer in [1, 200000].",
      "err.invalid_sweep_shots": "Live Monte Carlo sweep shot count must be an integer in [1, 100000].",
      "err.invalid_sweep_range": "Require 0 ≤ p_min < p_max ≤ 1 and grid points in [2, 501].",
      "err.eigenkit_stochastic": "eigenkit_unitary does not support stochastic channels (p > 0); switch to numpy or torch_cuda (or set p = 0).",
      "err.invalid_qubit": "Target qubit index must be an integer in [0, {max}].",
      "err.invalid_seed": "PRNG seed must be a non-negative integer.",
      "err.api_request": "Engine request failed: {detail}"
    },

    "pt-br": {
      "app.title": "Companheiro Interativo de QECC e DFS",
      "app.subtitle": "Códigos de Correção de Erros Quânticos, Estabilizadores, Coerência e Limiares de Ganho",
      "app.badge.engine": "Motor Python qecc (Big-Endian, qubit 0 MSB)",
      "app.badge.offline": "SPA Científico Offline",
      "lang.label": "Idioma",
      "lang.en": "EN",
      "lang.pt_br": "PT-BR",
      "nav.simulator": "Simulador Passo a Passo e Bloch",
      "nav.curves": "Curvas Analíticas e MC (p*)",
      "nav.theory": "Estabilizadores e Knill–Laflamme",
      "nav.seminar": "Guia do Seminário (20 Slides)",
      "nav.aria_label": "Seções principais",

      "code.label": "Código Quântico",
      "code.raw.name": "Qubit Físico Bruto Não Protegido (raw, [[1,1,1]])",
      "code.raw.short": "Bruto (1 qubit, sem proteção)",
      "code.raw.desc": "Referência de um único qubit físico sem codificação (F_e = 1 - p, F_state = 1 - 2p/3).",
      "code.bit_flip_3.name": "Código de Inversão de Bit de 3 Qubits (bit_flip_3, [[3,1,1]])",
      "code.bit_flip_3.short": "Inversão de Bit ([[3,1,1]], 3 dados + 2 ancillas)",
      "code.bit_flip_3.desc": "Código de repetição que codifica |0_L⟩ = |000⟩, |1_L⟩ = |111⟩. Corrige qualquer erro X de 1 qubit via estabilizadores ZZI, IZZ; vulnerável a erros de fase Z e aliasing de peso 2.",
      "code.phase_flip_3.name": "Código de Inversão de Fase de 3 Qubits (phase_flip_3, [[3,1,1]])",
      "code.phase_flip_3.short": "Inversão de Fase ([[3,1,1]], 3 dados + 2 ancillas)",
      "code.phase_flip_3.desc": "Código dual de Hadamard (H^⊗3) que codifica |0_L⟩ = |+++⟩, |1_L⟩ = |---⟩. Corrige qualquer erro Z de 1 qubit via estabilizadores XXI, IXX; falha sob defasagem coletiva ZZZ = X_L.",
      "code.five_qubit.name": "Código Perfeito [[5,1,3]] de Laflamme (five_qubit)",
      "code.five_qubit.short": "Perfeito [[5,1,3]] (5 dados + 4 ancillas)",
      "code.five_qubit.desc": "Menor código que satura o Limite Quântico de Hamming (2(1+15) = 32 = 2^5). Usa 4 estabilizadores cíclicos (XZZXI, IXZZX, XIXZZ, ZXIXZ) para corrigir qualquer erro X, Y ou Z de 1 qubit bijetivamente em 16 síndromes.",
      "code.dfs_2.name": "Subespaço Livre de Decoerência Passivo (dfs_2)",
      "code.dfs_2.short": "DFS Passivo (2 dados + 0 ancillas)",
      "code.dfs_2.desc": "Subespaço de paridade ímpar protegido por simetria span{|01⟩, |10⟩} com zero ancillas e zero medições de síndrome. Invariante (F = 1.0) sob defasagem coletiva Z⊗Z e Rz(φ)⊗Rz(φ), mas vulnerável a inversões de fase independentes por quebra de simetria.",

      "channel.label": "Canal de Ruído",
      "channel.bit_flip": "Inversão de Bit (bit_flip — X independente)",
      "channel.phase_flip": "Inversão de Fase (phase_flip — Z independente)",
      "channel.depolarizing": "Canal Despolarizante (depolarizing — X/Y/Z com p/3)",
      "channel.collective_dephasing": "Defasagem Coletiva (collective_dephasing — Z^⊗n correlacionado)",

      "error_mode.label": "Modo de Erro do Passo a Passo",
      "error_mode.pauli_string": "Cadeia de Pauli Determinística (I/X/Y/Z)",
      "error_mode.single_qubit_pauli": "Pauli de 1 Qubit (q, I/X/Y/Z)",
      "error_mode.stochastic_sample": "Amostra Estocástica de Canal (canal, p)",
      "error_mode.su2_rotation": "Rotação Contínua em SU(2) U(θ, n̂)",
      "error_mode.coherent_collective": "Defasagem Coletiva Coerente Rz(φ)^⊗n",

      "state.label": "Estado Lógico de Entrada |ψ⟩",
      "state.0": "|0⟩ — Zero computacional (r = (0, 0, +1))",
      "state.1": "|1⟩ — Um computacional (r = (0, 0, -1))",
      "state.plus": "|+⟩ — Autoestado + de X (r = (+1, 0, 0))",
      "state.minus": "|-⟩ — Autoestado - de X (r = (-1, 0, 0))",
      "state.plus_i": "|+i⟩ — Autoestado +i de Y (r = (0, +1, 0), |ρ₀₁| = 0.5)",
      "state.minus_i": "|-i⟩ — Autoestado -i de Y (r = (0, -1, 0), |ρ₀₁| = 0.5)",
      "state.cardinal_avg": "cardinal_avg — Média dos 6 Estados Cardinais (2-design de Haar)",
      "state.bloch": "Ângulos de Bloch Personalizados (θ, φ)",
      "state.custom": "Amplitudes Personalizadas (α, β)",
      "state.cardinal_stepper_note": "Nota: cardinal_avg calcula a média dos 6 estados cardinais na simulação Monte Carlo; a inspeção passo a passo exibe |+i⟩.",

      "param.theta": "Ângulo polar θ (rad)",
      "param.phi_state": "Ângulo azimutal φ_state (rad)",
      "param.alpha_re": "Re(α)",
      "param.alpha_im": "Im(α)",
      "param.beta_re": "Re(β)",
      "param.beta_im": "Im(β)",
      "param.normalize_hint": "Deve satisfazer |α|² + |β|² = 1 (tol = 1e-6).",

      "ctrl.p": "Taxa de erro físico p ([0, 1])",
      "ctrl.phi": "Fase coletiva coerente φ (rad)",
      "ctrl.pauli_string": "Cadeia de Pauli ({n} caracteres)",
      "ctrl.error_qubit": "Qubit de dados alvo q",
      "ctrl.error_op": "Operador de Pauli (I/X/Y/Z)",
      "ctrl.su2_qubit": "Qubit alvo q de SU(2)",
      "ctrl.su2_theta": "Ângulo de rotação θ (rad)",
      "ctrl.su2_nx": "Eixo n_x",
      "ctrl.su2_ny": "Eixo n_y",
      "ctrl.su2_nz": "Eixo n_z",
      "ctrl.syndrome_branch": "Projeção no ramo de síndrome",
      "ctrl.syndrome_auto": "Ramo de maior probabilidade (automático)",
      "ctrl.num_shots": "Amostras (shots) de Monte Carlo",
      "ctrl.seed": "Semente (seed) do PRNG",
      "ctrl.backend": "Backend de simulação",

      "backend.auto": "auto (GPU torch_cuda se disponível, senão numpy)",
      "backend.numpy": "numpy (NumPyCPUBackend, CPU complex128)",
      "backend.torch_cuda": "torch_cuda (TorchCUDABackend, RTX 5080 sm_120)",
      "backend.eigenkit_unitary": "eigenkit_unitary (EigenkitUnitaryAdapter, apenas unitário)",
      "backend.unavailable": "Indisponível neste host",

      "btn.run_step": "Executar Evolução Passo a Passo",
      "btn.run_sim": "Executar Ciclo Monte Carlo",
      "btn.run_sweep": "Atualizar Curvas",
      "btn.resample_seed": "Nova Amostra Aleatória",
      "btn.normalize": "Normalizar",

      "stage.section_title": "Pipeline de Evolução de Estado Passo a Passo",
      "stage.initial.title": "1. Estado Inicial (initial)",
      "stage.initial.desc": "A entrada lógica |ψ⟩ entra no qubit 0 (MSB na convenção big-endian); os demais qubits de dados e ancillas iniciam em |0⟩.",
      "stage.encoded.title": "2. Codificação (encoded)",
      "stage.encoded.desc": "Emaranha a informação lógica no subespaço do código (|ψ⟩ → α|0_L⟩ + β|1_L⟩) via encode_ops(), respeitando o Teorema da Não-Clonagem.",
      "stage.noisy.title": "3. Injeção de Ruído / Discretização (noisy)",
      "stage.noisy.desc": "Aplica o erro físico (cadeia de Pauli discreta, amostra estocástica, rotação contínua em SU(2) ou Rz(φ)^⊗n coerente). Rotações contínuas criam uma superposição de ramos de Pauli.",
      "stage.syndrome_extracted.title": "4. Extração de Síndrome por Ancilla (syndrome_extracted)",
      "stage.syndrome_extracted.desc": "O retrocesso de fase (phase kickback) não destrutivo (H · Λ(S_k) · H) mapeia a paridade dos estabilizadores nas ancillas sem vazar α, β. A medição projetiva colapsa erros contínuos em uma coclasse discreta s.",
      "stage.syndrome_extracted.desc_dfs": "O dfs_2 possui zero ancillas (n_ancilla = 0) e não realiza extração de síndrome; a proteção decorre passivamente da simetria coletiva.",
      "stage.recovered.title": "5. Recuperação Condicional (recovered)",
      "stage.recovered.desc": "A realimentação clássica condicional (feed-forward) aplica recovery_ops(syndrome) conforme a síndrome medida, retornando ao subespaço do código (salvo se houver aliasing incorrigível).",
      "stage.recovered.desc_dfs": "O dfs_2 não exige operações de recuperação (recovery_ops(\"\") = []); Z⊗Z ou Rz(φ)⊗Rz(φ) coletivo atua apenas como fase global inobservável.",
      "stage.decoded.title": "6. Decodificação e Traço Parcial (decoded)",
      "stage.decoded.desc": "O circuito unitário decode_ops() desemaranha o qubit lógico de volta ao qubit 0, seguido por partial_trace_to_qubit0 para obter ρ_out (2×2), vetor de Bloch, |ρ₀₁| e fidelidade.",

      "stage.metric.fidelity": "Fidelidade vs. Entrada",
      "stage.metric.subspace_prob": "Prob. no Subespaço de Código",
      "stage.metric.gates_applied": "Portas Aplicadas na Etapa",
      "stage.metric.none": "Nenhuma (identidade / no-op)",
      "stage.metric.syndrome_probs": "Probabilidades de Síndrome das Ancillas (clique para selecionar o ramo)",
      "stage.metric.selected_syndrome": "Síndrome Colapsada",
      "stage.metric.dfs_empty_syndrome": "\"\" (zero ancillas — DFS passivo)",
      "stage.metric.pauli_coeffs": "Expansão na Base de Pauli E = Σ c_P P",
      "stage.metric.nonzero_amplitudes": "Principais Amplitudes na Base Computacional",
      "stage.metric.logical_bloch": "Vetor de Bloch Lógico",
      "stage.metric.abs_rho01": "|ρ₀₁| Lógico",

      "circuit.title": "Diagrama de Circuito Quântico (Big-Endian: q₀ MSB)",
      "circuit.data_qubit": "Dados q{idx}",
      "circuit.ancilla_qubit": "Ancilla a{idx} (q{q})",
      "circuit.noise_box": "Ruído: {label}",
      "circuit.measure": "Medir",
      "circuit.recovery_box": "R({syndrome}) Condicional",
      "circuit.stage_encode": "Codificação",
      "circuit.stage_noise": "Ruído",
      "circuit.stage_syndrome": "Síndrome",
      "circuit.stage_recovery": "Recuperação",
      "circuit.stage_decode": "Decodificação",

      "bloch.title": "Esfera de Bloch 3D e Trajetória",
      "bloch.reset": "Redefinir Vista",
      "bloch.legend.input": "Entrada |ψ⟩",
      "bloch.legend.raw": "raw Não Protegido",
      "bloch.legend.qecc": "QECC Ativo ({code})",
      "bloch.legend.dfs": "dfs_2 Passivo",
      "bloch.legend.stage": "Etapa Selecionada",
      "bloch.hint_drag": "Arraste para girar a vista 3D",

      "coherence.title": "Coerência Fora da Diagonal |ρ₀₁| e Comparação Tripla",
      "coherence.rho_matrix": "Matriz Densidade Reduzida ρ_out (2×2)",
      "coherence.abs_rho01": "Coerência Fora da Diagonal |ρ₀₁|",
      "coherence.bloch_vec": "Vetor de Bloch (r_x, r_y, r_z)",
      "coherence.fidelity": "Fidelidade F",
      "coherence.compare.raw": "Bruto Não Protegido (raw)",
      "coherence.compare.qecc": "Código Selecionado ({code})",
      "coherence.compare.dfs": "dfs_2 Passivo",
      "coherence.zero_basis_note": "Nota: Os estados da base computacional |0⟩ e |1⟩ possuem coerência inicial zero (|ρ₀₁| = 0). Selecione |+i⟩ (|ρ₀₁| ideal = 0.5) para observar o decaimento e a proteção da coerência de fase.",

      "sim.mc_summary": "Resultado Monte Carlo do Conjunto ({backend})",
      "sim.mean_fidelity": "Fidelidade Média",
      "sim.std_error": "Erro Padrão (SE)",
      "sim.ana_ent": "F_e Analítica",
      "sim.ana_state": "F_state Analítica",
      "sim.elapsed": "Tempo Decorrido",
      "sim.syndrome_counts": "Contagens de Síndrome",
      "sim.raw_baseline": "Fidelidade de Referência Bruta",

      "curves.title": "Fidelidade Analítica e Monte Carlo vs. Taxa de Erro (p)",
      "curves.metric.label": "Métrica de Fidelidade",
      "curves.metric.entanglement": "Fidelidade de Emaranhamento F_e(p) (referência 1 - p)",
      "curves.metric.state": "Fidelidade Média de Estado de Haar F_state(p) (referência 1 - 2p/3)",
      "curves.metric_notice.title": "Regra Estrita de Separação de Métricas",
      "curves.metric_notice.entanglement": "Comparando a Fidelidade de Emaranhamento F_e(p) = P_I(p) estritamente com a referência bruta F_e,raw(p) = 1 - p. Nunca compare F_state com 1 - p (incompatibilidade de vantagem espúria).",
      "curves.metric_notice.state": "Comparando a Fidelidade Média de Estado de Haar F_state(p) = 1 - (2/3)(1 - F_e(p)) estritamente com a referência bruta F_state,raw(p) = 1 - (2/3)p. O limiar p* é invariante sob ambas as métricas.",
      "curves.axis.p": "Taxa de Erro Físico p",
      "curves.axis.fidelity_ent": "Fidelidade de Emaranhamento F_e(p)",
      "curves.axis.fidelity_state": "Fidelidade Média de Estado F_state(p)",
      "curves.axis.delta_f": "Ganho Líquido de Fidelidade ΔF vs. Bruto",
      "curves.axis.abs_rho01": "Coerência |ρ₀₁| em |+i⟩",
      "curves.raw_baseline_ent": "Bruto Não Protegido (1 - p)",
      "curves.raw_baseline_state": "Bruto Não Protegido (1 - 2p/3)",
      "curves.mc_source.cached": "Conjunto em cache de 18 mil shots em torch_cuda (fidelity_sweeps.json)",
      "curves.mc_source.live": "Varredura Monte Carlo ao Vivo ({shots} shots, {backend})",
      "curves.run_live_mc": "Executar Varredura Monte Carlo ao Vivo",
      "curves.delta_chart.title": "Ganho Líquido de Fidelidade (ΔF)",
      "curves.coh_chart.title": "Coerência Analítica |ρ₀₁|(p) em |+i⟩",
      "curves.p_star.title": "Limiares Analíticos de Ganho Exatos (p* via brentq)",
      "curves.p_star.col_code": "Código",
      "curves.p_star.col_channel": "Canal",
      "curves.p_star.col_p_star": "p* Exato",
      "curves.p_star.col_region": "Região de Vantagem",
      "curves.p_star.none": "Nenhuma (0.0000)",
      "curves.p_min": "p_min",
      "curves.p_max": "p_max",
      "curves.num_p": "Pontos da grade",

      "theory.title": "Estrutura Algébrica, Tabela de Síndromes e Verificação de Knill–Laflamme",
      "theory.stabilizers": "Geradores Estabilizadores (S_k)",
      "theory.stabilizers.none": "Nenhum — o dfs_2 usa simetria passiva (span{|01⟩, |10⟩}) sem ancillas estabilizadoras.",
      "theory.logical_ops": "Operadores Lógicos",
      "theory.syndrome_table": "Tabela de Consulta de Síndromes",
      "theory.syndrome_table.empty": "Vazia — o dfs_2 não realiza extração de síndrome por ancillas.",
      "theory.syndrome_col": "Síndrome",
      "theory.recovery_col": "Pauli de Recuperação",
      "theory.target_col": "Ação",
      "theory.no_error": "Sem erro (identidade)",
      "theory.correct_pauli": "Corrigir {op}",
      "theory.aliasing.title": "Exemplos de Falha por Aliasing de Peso 2 e Erros Incorrigíveis",
      "theory.aliasing.error": "Erro Injetado",
      "theory.aliasing.syndrome": "Síndrome Medida",
      "theory.aliasing.recovery": "Recuperação Aplicada",
      "theory.aliasing.residual": "Operador Residual",
      "theory.aliasing.effect": "Efeito Lógico",
      "theory.aliasing.note_col": "Explicação",
      "theory.aliasing.note_pauli": "Erro de peso 2 {error} sofre aliasing na síndrome {syndrome}; a recuperação {recovery} deixa o operador lógico {effect} ({residual}).",
      "theory.aliasing.note_dfs_z": "Erro de fase de 1 qubit por quebra de simetria {error} induz inversão de fase lógica {effect} não detectada.",
      "theory.aliasing.note_dfs_x": "Inversão de bit {error} causa vazamento do estado para fora do subespaço livre de decoerência span{|01⟩, |10⟩}.",
      "theory.kl.title": "Condições de Knill–Laflamme: ⟨i_L| E_a† E_b |j_L⟩ = C_ab δ_ij",
      "theory.kl.status": "Status da Verificação",
      "theory.kl.passed": "Verificado (tol = 1e-12)",
      "theory.kl.failed": "Violado",
      "theory.kl.max_violation": "Violação Máxima",
      "theory.kl.max_diag_diff": "Diferença Diagonal Máxima",
      "theory.kl.max_off_diag": "Máximo Fora da Diagonal",
      "theory.kl.error_basis": "Base de Erros Verificada E",
      "theory.kl.c_matrix": "Matriz Hermitiana C_ab",
      "theory.enum.title": "Enumerador de Pesos Exato de 1024 Termos do [[5,1,3]] (4^5 = 1024)",
      "theory.enum.subtitle": "Contagem exata por coclasse após recuperação por síndrome em todas as 1024 cadeias de Pauli de 5 qubits.",
      "theory.enum.weight": "Peso de Hamming w",
      "theory.enum.coset_i": "Coclasse I_L S",
      "theory.enum.coset_x": "Coclasse X_L S",
      "theory.enum.coset_y": "Coclasse Y_L S",
      "theory.enum.coset_z": "Coclasse Z_L S",
      "theory.enum.total": "Total C(5,w)·3^w",
      "theory.enum.pure_x": "X Puro (coclasse I)",
      "theory.enum.pure_z": "Z Puro (coclasse I)",
      "theory.enum.sum": "Soma",

      "seminar.title": "Guia do Seminário de Pós-Graduação e Apresentação ao Vivo",
      "seminar.topic_label": "Tópico",
      "seminar.slide_label": "Slide",
      "seminar.title_badge": "Slide de Título",
      "seminar.slide_counter": "Slide {n} / 19",
      "seminar.prev": "Anterior",
      "seminar.next": "Próximo",
      "seminar.apply_demo": "Carregar Demonstração no Simulador",
      "seminar.demo_loaded": "Preset do slide carregado no simulador: {code}",
      "seminar.equations_label": "Equações Principais",
      "seminar.preset_summary": "Preset Vinculado ao Simulador",

      "status.loading": "Calculando via motor qecc...",
      "status.ready": "Pronto",
      "err.invalid_p": "A taxa de erro p deve ser um número finito em [0, 1].",
      "err.invalid_phi": "A fase coletiva φ deve ser um número real finito.",
      "err.invalid_bloch_angles": "Os ângulos de Bloch (θ, φ_state) devem ser números reais finitos.",
      "err.invalid_pauli": "A cadeia de Pauli deve ter comprimento {n} usando apenas I, X, Y, Z.",
      "err.invalid_custom_norm": "As amplitudes personalizadas devem ser normalizadas (|α|² + |β|² = 1).",
      "err.invalid_su2_axis": "O eixo de rotação SU(2) (n_x, n_y, n_z) deve ser não nulo.",
      "err.invalid_shots": "O número de shots deve ser um inteiro em [1, 200000].",
      "err.invalid_sweep_shots": "O número de shots da varredura Monte Carlo ao vivo deve ser um inteiro em [1, 100000].",
      "err.invalid_sweep_range": "Requer 0 ≤ p_min < p_max ≤ 1 e pontos da grade em [2, 501].",
      "err.eigenkit_stochastic": "eigenkit_unitary não suporta canais estocásticos (p > 0); selecione numpy ou torch_cuda (ou defina p = 0).",
      "err.invalid_qubit": "O índice do qubit alvo deve ser um inteiro em [0, {max}].",
      "err.invalid_seed": "A semente (seed) do PRNG deve ser um inteiro não negativo.",
      "err.api_request": "Falha na requisição ao motor: {detail}"
    }
  };

  let currentLang = "en";
  try {
    const saved = global.localStorage && global.localStorage.getItem(STORAGE_KEY);
    if (saved === "pt-br" || saved === "en") {
      currentLang = saved;
    }
  } catch (_err) {
    currentLang = "en";
  }

  const listeners = new Set();

  function normalizeLang(lang) {
    if (typeof lang !== "string") return "en";
    const low = lang.toLowerCase();
    return low === "pt-br" || low === "pt_br" || low === "pt" ? "pt-br" : "en";
  }

  function getLanguage() {
    return currentLang;
  }

  function t(key, params) {
    const dict = STRINGS[currentLang] || STRINGS.en;
    let template = dict[key];
    if (template === undefined) {
      template = STRINGS.en[key];
    }
    if (template === undefined) {
      return "";
    }
    if (!params || typeof params !== "object") {
      return template;
    }
    return template.replace(/\{([a-zA-Z0-9_]+)\}/g, function (_m, name) {
      const v = params[name];
      return v === undefined || v === null ? "" : String(v);
    });
  }

  function applyStaticDomTranslations() {
    if (typeof document === "undefined") return;
    document.documentElement.lang = currentLang === "pt-br" ? "pt-BR" : "en";
    document.title = t("app.title");

    const toggleWrap = document.querySelector('[data-testid="lang-toggle"]');
    if (toggleWrap) {
      toggleWrap.setAttribute("data-current-lang", currentLang);
    }
    const btnEn = document.querySelector('[data-testid="lang-en"]');
    const btnPt = document.querySelector('[data-testid="lang-pt-br"]');
    if (btnEn) {
      const isEn = currentLang === "en";
      btnEn.setAttribute("aria-pressed", isEn ? "true" : "false");
      btnEn.classList.toggle("active", isEn);
    }
    if (btnPt) {
      const isPt = currentLang === "pt-br";
      btnPt.setAttribute("aria-pressed", isPt ? "true" : "false");
      btnPt.classList.toggle("active", isPt);
    }

    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      const key = el.getAttribute("data-i18n");
      const val = t(key);
      if (val !== "") {
        el.textContent = val;
      }
    });

    document.querySelectorAll("[data-i18n-aria]").forEach(function (el) {
      const key = el.getAttribute("data-i18n-aria");
      const val = t(key);
      if (val !== "") {
        el.setAttribute("aria-label", val);
      }
    });
  }

  function setLanguage(lang, opts) {
    const next = normalizeLang(lang);
    const rerender = !opts || opts.rerender !== false;
    currentLang = next;
    try {
      if (global.localStorage) {
        global.localStorage.setItem(STORAGE_KEY, currentLang);
      }
    } catch (_err) {
      // Ignore storage failures in restricted contexts
    }
    applyStaticDomTranslations();
    if (rerender) {
      listeners.forEach(function (cb) {
        try {
          cb(currentLang);
        } catch (_err) {
          // Keep listener dispatch resilient
        }
      });
    }
    return currentLang;
  }

  function onLanguageChange(cb) {
    if (typeof cb === "function") {
      listeners.add(cb);
    }
    return function unsubscribe() {
      listeners.delete(cb);
    };
  }

  global.QECC_I18N = {
    SUPPORTED_LANGS: SUPPORTED_LANGS,
    STRINGS: STRINGS,
    getLanguage: getLanguage,
    setLanguage: setLanguage,
    t: t,
    applyStaticDomTranslations: applyStaticDomTranslations,
    onLanguageChange: onLanguageChange
  };
})(window);
