(function (global) {
  "use strict";

  const STAGE_KEYS = [
    "initial",
    "encoded",
    "noisy",
    "syndrome_extracted",
    "recovered",
    "decoded"
  ];

  const DEFAULT_PAULI = {
    bit_flip_3: "IXI",
    phase_flip_3: "IZI",
    five_qubit: "IXIII",
    dfs_2: "ZI"
  };

  const VALIDATION_SOURCES = ["init", "step", "sim", "sweeps", "theory"];

  const state = {
    metadata: null,
    activeStageKey: "decoded",
    selectedSyndrome: null,
    stepData: null,
    simData: null,
    sweepsData: null,
    theoryData: null,
    toastSpec: null,
    liveMcActive: false,
    inFlightCount: 0,
    validationErrors: {
      init: null,
      step: null,
      sim: null,
      sweeps: null,
      theory: null
    },
    seq: { step: 0, sim: 0, sweeps: 0, theory: 0 }
  };

  let blochRenderer = null;
  let seminarCtrl = null;

  function $(id) {
    return document.getElementById(id);
  }

  function t(key, params) {
    return global.QECC_I18N ? global.QECC_I18N.t(key, params) : key;
  }

  function renderStatusBadge() {
    const badge = $("engine-status-badge");
    if (!badge) return;
    badge.textContent = state.inFlightCount > 0 ? t("status.loading") : t("status.ready");
  }

  function beginLoading() {
    state.inFlightCount += 1;
    renderStatusBadge();
  }

  function endLoading() {
    if (state.inFlightCount > 0) {
      state.inFlightCount -= 1;
    }
    renderStatusBadge();
  }

  function formatValidationError(errObj) {
    if (!errObj) return "";
    if (typeof errObj === "string") return errObj;
    if (errObj.key) {
      const translated = t(errObj.key, errObj.params);
      if (translated) return translated;
    }
    return errObj.fallback || "";
  }

  function renderValidationErrors() {
    const box = $("validation-error");
    if (!box) return;
    const messages = [];
    VALIDATION_SOURCES.forEach(function (src) {
      const msg = formatValidationError(state.validationErrors[src]);
      if (msg && messages.indexOf(msg) === -1) {
        messages.push(msg);
      }
    });
    if (messages.length > 0) {
      box.textContent = messages.join(" | ");
      box.classList.remove("hidden");
    } else {
      box.textContent = "";
      box.classList.add("hidden");
    }
  }

  function setValidationError(source, errObj) {
    if (!Object.prototype.hasOwnProperty.call(state.validationErrors, source)) {
      state.validationErrors[source] = null;
    }
    state.validationErrors[source] = errObj || null;
    renderValidationErrors();
  }

  function clearValidationErrorIfKey(source, key) {
    const cur = state.validationErrors[source];
    if (cur && cur.key === key) {
      state.validationErrors[source] = null;
      renderValidationErrors();
    }
  }

  function renderToast() {
    const box = $("demo-toast");
    if (!box) return;
    const spec = state.toastSpec;
    if (spec && spec.key) {
      box.textContent = t(spec.key, spec.params);
      box.classList.remove("hidden");
    } else if (spec && spec.text) {
      box.textContent = spec.text;
      box.classList.remove("hidden");
    } else {
      box.textContent = "";
      box.classList.add("hidden");
    }
  }

  function showToast(keyOrMsg, params) {
    if (!keyOrMsg) {
      state.toastSpec = null;
    } else if (typeof keyOrMsg === "object" && keyOrMsg.key) {
      state.toastSpec = { key: keyOrMsg.key, params: keyOrMsg.params };
    } else if (typeof keyOrMsg === "string" && params !== undefined) {
      state.toastSpec = { key: keyOrMsg, params: params };
    } else {
      state.toastSpec = { text: String(keyOrMsg) };
    }
    renderToast();
  }

  function getNData(code) {
    if (state.metadata && state.metadata.codes && state.metadata.codes[code]) {
      return Number(state.metadata.codes[code].n_data) || 3;
    }
    const fallback = { bit_flip_3: 3, phase_flip_3: 3, five_qubit: 5, dfs_2: 2 };
    return fallback[code] || 3;
  }

  function translateSelectOptionsPreservingValues() {
    const codeSel = $("code-select");
    if (codeSel) {
      const val = codeSel.value;
      Array.from(codeSel.options).forEach(function (opt) {
        const key = "code." + opt.value + ".name";
        const translated = t(key);
        if (translated) opt.textContent = translated;
      });
      codeSel.value = val;
    }

    const chSel = $("channel-select");
    if (chSel) {
      const val = chSel.value;
      Array.from(chSel.options).forEach(function (opt) {
        const translated = t("channel." + opt.value);
        if (translated) opt.textContent = translated;
      });
      chSel.value = val;
    }

    const stSel = $("state-select");
    if (stSel) {
      const val = stSel.value;
      const map = {
        "0": "state.0",
        "1": "state.1",
        "+": "state.plus",
        "-": "state.minus",
        "+i": "state.plus_i",
        "-i": "state.minus_i",
        cardinal_avg: "state.cardinal_avg",
        bloch: "state.bloch",
        custom: "state.custom"
      };
      Array.from(stSel.options).forEach(function (opt) {
        const translated = t(map[opt.value] || "");
        if (translated) opt.textContent = translated;
      });
      stSel.value = val;
    }

    const emSel = $("error-mode-select");
    if (emSel) {
      const val = emSel.value;
      Array.from(emSel.options).forEach(function (opt) {
        const translated = t("error_mode." + opt.value);
        if (translated) opt.textContent = translated;
      });
      emSel.value = val;
    }

    const metSel = $("metric-select");
    if (metSel) {
      const val = metSel.value;
      Array.from(metSel.options).forEach(function (opt) {
        const translated = t("curves.metric." + opt.value);
        if (translated) opt.textContent = translated;
      });
      metSel.value = val;
    }

    const bkSel = $("backend-select");
    if (bkSel) {
      const val = bkSel.value;
      const metaBackends = (state.metadata && state.metadata.backends) || [];
      Array.from(bkSel.options).forEach(function (opt) {
        const info = metaBackends.find(function (b) { return b.name === opt.value; });
        let txt = t("backend." + opt.value) || opt.value;
        if (info && !info.available) {
          txt += " (" + t("backend.unavailable") + ")";
          opt.disabled = true;
        } else {
          opt.disabled = false;
        }
        opt.textContent = txt;
      });
      bkSel.value = val;
    }

    const synSel = $("syndrome-branch-select");
    if (synSel && synSel.options.length > 0) {
      const val = synSel.value;
      synSel.options[0].textContent = t("ctrl.syndrome_auto");
      synSel.value = val;
    }
  }

  function updateConditionalFieldVisibility() {
    const code = $("code-select").value;
    const nData = getNData(code);
    const stateMode = $("state-select").value;
    const errorMode = $("error-mode-select").value;

    const stateSub = $("state-subcontrols");
    const blochFields = $("bloch-state-fields");
    const customFields = $("custom-state-fields");
    const showStateSub = stateMode === "bloch" || stateMode === "custom";
    stateSub.classList.toggle("hidden", !showStateSub);
    blochFields.classList.toggle("hidden", stateMode !== "bloch");
    customFields.classList.toggle("hidden", stateMode !== "custom");

    $("mode-pauli-fields").classList.toggle("hidden", errorMode !== "pauli_string");
    $("mode-single-fields").classList.toggle("hidden", errorMode !== "single_qubit_pauli");
    $("mode-su2-fields").classList.toggle("hidden", errorMode !== "su2_rotation");

    const pauliLabel = document.querySelector('label[for="pauli-string-input"]');
    if (pauliLabel) {
      pauliLabel.textContent = t("ctrl.pauli_string", { n: nData });
    }
    const eqInput = $("error-qubit-input");
    const sqInput = $("su2-qubit-input");
    if (eqInput) {
      eqInput.max = String(nData - 1);
      if (Number(eqInput.value) >= nData) eqInput.value = "0";
    }
    if (sqInput) {
      sqInput.max = String(nData - 1);
      if (Number(sqInput.value) >= nData) sqInput.value = "0";
    }

    const summary = $("active-code-summary");
    if (summary) {
      summary.textContent = t("code." + code + ".desc") +
        (stateMode === "cardinal_avg" ? " — " + t("state.cardinal_stepper_note") : "");
    }
  }

  function parseFiniteInput(el) {
    if (!el) return NaN;
    if (el.validity && el.validity.badInput) return NaN;
    const raw = String(el.value || "").trim();
    if (raw === "") return NaN;
    const num = Number(raw);
    return Number.isFinite(num) ? num : NaN;
  }

  function clearSelectedSyndrome() {
    state.selectedSyndrome = null;
    const synSel = $("syndrome-branch-select");
    if (synSel) synSel.value = "";
  }

  function parseOptionalSeedInput() {
    const el = $("seed-input");
    if (!el) return { ok: true, value: undefined };
    if (el.validity && el.validity.badInput) {
      return { ok: false, err: { key: "err.invalid_seed" } };
    }
    const raw = String(el.value || "").trim();
    if (raw === "") {
      return { ok: true, value: undefined };
    }
    const num = Number(raw);
    if (!Number.isSafeInteger(num) || num < 0) {
      return { ok: false, err: { key: "err.invalid_seed" } };
    }
    return { ok: true, value: num === 0 ? 0 : num };
  }

  function validateStateParams(forStepper) {
    const stateVal = $("state-select").value;
    if (stateVal === "bloch") {
      const th = parseFiniteInput($("theta-input"));
      const ph = parseFiniteInput($("phi-state-input"));
      if (!Number.isFinite(th) || !Number.isFinite(ph)) {
        return { ok: false, err: { key: "err.invalid_bloch_angles" } };
      }
      return { ok: true, stateKey: "bloch", theta: th, phi_state: ph };
    }
    if (stateVal === "custom") {
      const ar = parseFiniteInput($("alpha-re-input"));
      const ai = parseFiniteInput($("alpha-im-input"));
      const br = parseFiniteInput($("beta-re-input"));
      const bi = parseFiniteInput($("beta-im-input"));
      if (![ar, ai, br, bi].every(Number.isFinite)) {
        return { ok: false, err: { key: "err.invalid_custom_norm" } };
      }
      const normSq = ar * ar + ai * ai + br * br + bi * bi;
      if (Math.abs(normSq - 1.0) > 1e-6) {
        return { ok: false, err: { key: "err.invalid_custom_norm" } };
      }
      return { ok: true, stateKey: "custom", amplitudes: [[ar, ai], [br, bi]] };
    }
    if (forStepper && stateVal === "cardinal_avg") {
      return { ok: true, stateKey: "+i" };
    }
    return { ok: true, stateKey: stateVal };
  }

  function buildValidatedStepPayload() {
    const code = $("code-select").value;
    const nData = getNData(code);
    const errorMode = $("error-mode-select").value;
    const st = validateStateParams(true);
    if (!st.ok) return { ok: false, err: st.err };

    const payload = {
      code: code,
      state_preset: st.stateKey,
      error_mode: errorMode
    };
    if (st.stateKey === "bloch") {
      payload.theta = st.theta;
      payload.phi_state = st.phi_state;
    } else if (st.stateKey === "custom") {
      payload.amplitudes = st.amplitudes;
    }

    if (errorMode === "pauli_string") {
      const ps = String($("pauli-string-input").value || "").trim().toUpperCase();
      if (ps.length !== nData || !/^[IXYZ]+$/.test(ps)) {
        return { ok: false, err: { key: "err.invalid_pauli", params: { n: nData } } };
      }
      payload.pauli_string = ps;
    } else if (errorMode === "single_qubit_pauli") {
      const q = parseFiniteInput($("error-qubit-input"));
      if (!Number.isInteger(q) || q < 0 || q >= nData) {
        return { ok: false, err: { key: "err.invalid_qubit", params: { max: nData - 1 } } };
      }
      payload.error_qubit = q;
      payload.error_op = $("error-op-select").value;
    } else if (errorMode === "stochastic_sample") {
      const p = parseFiniteInput($("p-input"));
      if (!Number.isFinite(p) || p < 0 || p > 1) {
        return { ok: false, err: { key: "err.invalid_p" } };
      }
      payload.channel = $("channel-select").value;
      payload.p = p;
      const seedParsed = parseOptionalSeedInput();
      if (!seedParsed.ok) {
        return { ok: false, err: seedParsed.err };
      }
      if (seedParsed.value !== undefined) {
        payload.seed = seedParsed.value;
      }
    } else if (errorMode === "su2_rotation") {
      const q = parseFiniteInput($("su2-qubit-input"));
      const th = parseFiniteInput($("su2-theta-input"));
      const nx = parseFiniteInput($("su2-nx-input"));
      const ny = parseFiniteInput($("su2-ny-input"));
      const nz = parseFiniteInput($("su2-nz-input"));
      if (!Number.isInteger(q) || q < 0 || q >= nData) {
        return { ok: false, err: { key: "err.invalid_qubit", params: { max: nData - 1 } } };
      }
      if (![th, nx, ny, nz].every(Number.isFinite)) {
        return { ok: false, err: { key: "err.invalid_su2_axis" } };
      }
      if (Math.hypot(nx, ny, nz) <= 1e-12) {
        return { ok: false, err: { key: "err.invalid_su2_axis" } };
      }
      payload.su2_qubit = q;
      payload.su2_theta = th;
      payload.su2_nx = nx;
      payload.su2_ny = ny;
      payload.su2_nz = nz;
    } else if (errorMode === "coherent_collective") {
      const phi = parseFiniteInput($("phi-input"));
      if (!Number.isFinite(phi)) {
        return { ok: false, err: { key: "err.invalid_phi" } };
      }
      payload.collective_phi = phi;
    }

    if (state.selectedSyndrome !== null && state.selectedSyndrome !== "") {
      payload.selected_syndrome = state.selectedSyndrome;
    }
    return { ok: true, payload: payload };
  }

  function buildValidatedSimPayload() {
    const code = $("code-select").value;
    const channel = $("channel-select").value;
    const p = parseFiniteInput($("p-input"));
    if (!Number.isFinite(p) || p < 0 || p > 1) {
      return { ok: false, err: { key: "err.invalid_p" } };
    }
    const shots = parseFiniteInput($("num-shots-input"));
    if (!Number.isInteger(shots) || shots < 1 || shots > 200000) {
      return { ok: false, err: { key: "err.invalid_shots" } };
    }
    const backend = $("backend-select").value;
    if (backend === "eigenkit_unitary" && p > 0) {
      return { ok: false, err: { key: "err.eigenkit_stochastic" } };
    }
    const st = validateStateParams(false);
    if (!st.ok) return { ok: false, err: st.err };

    const payload = {
      code: code,
      channel: channel,
      p: p,
      num_shots: shots,
      state_mode: st.stateKey,
      backend: backend
    };
    const seedParsed = parseOptionalSeedInput();
    if (!seedParsed.ok) {
      return { ok: false, err: seedParsed.err };
    }
    if (seedParsed.value !== undefined) {
      payload.seed = seedParsed.value;
    }
    if (st.stateKey === "bloch") {
      payload.theta = st.theta;
      payload.phi_state = st.phi_state;
    } else if (st.stateKey === "custom") {
      payload.amplitudes = st.amplitudes;
    }
    return { ok: true, payload: payload };
  }

  function buildValidatedSweepsPayload(runLiveMc) {
    const metric = $("metric-select").value || "entanglement";
    const pMin = parseFiniteInput($("sweep-pmin"));
    const pMax = parseFiniteInput($("sweep-pmax"));
    const numP = parseFiniteInput($("sweep-nump"));
    if (!Number.isFinite(pMin) || !Number.isFinite(pMax) || pMin < 0 || pMax > 1 || pMin >= pMax || !Number.isInteger(numP) || numP < 2 || numP > 501) {
      return { ok: false, err: { key: "err.invalid_sweep_range" } };
    }
    const shots = parseFiniteInput($("num-shots-input"));
    const clampedShots = Number.isInteger(shots) ? Math.min(100000, Math.max(1, shots)) : 2000;
    if (runLiveMc && (!Number.isInteger(shots) || shots < 1 || shots > 100000)) {
      return { ok: false, err: { key: "err.invalid_sweep_shots" } };
    }
    const backend = $("backend-select").value;
    if (runLiveMc && backend === "eigenkit_unitary") {
      return { ok: false, err: { key: "err.eigenkit_stochastic" } };
    }

    const payload = {
      codes: ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
      channels: ["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"],
      metric: metric,
      p_min: pMin,
      p_max: pMax,
      num_p: numP,
      run_live_mc: !!runLiveMc,
      num_shots: clampedShots,
      seed: 20261118,
      backend: backend
    };
    return { ok: true, payload: payload };
  }

  async function postJson(url, body) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await resp.json().catch(function () { return {}; });
    if (!resp.ok) {
      throw new Error(data.error || ("HTTP " + resp.status));
    }
    return data;
  }

  async function getJson(url) {
    const resp = await fetch(url);
    const data = await resp.json().catch(function () { return {}; });
    if (!resp.ok) {
      throw new Error(data.error || ("HTTP " + resp.status));
    }
    return data;
  }

  function updateStageButtonsUi() {
    STAGE_KEYS.forEach(function (k) {
      const btn = document.querySelector('[data-testid="stage-step-' + k + '"]');
      if (!btn) return;
      const isAct = k === state.activeStageKey;
      btn.classList.toggle("active", isAct);
      btn.setAttribute("data-active", isAct ? "true" : "false");
      if (isAct) {
        btn.setAttribute("aria-current", "step");
      } else {
        btn.removeAttribute("aria-current");
      }
    });
  }

  function syncSyndromeDropdown() {
    const synSel = $("syndrome-branch-select");
    if (!synSel) return;
    const prev = state.selectedSyndrome || "";
    synSel.textContent = "";
    const autoOpt = document.createElement("option");
    autoOpt.value = "";
    autoOpt.textContent = t("ctrl.syndrome_auto");
    synSel.appendChild(autoOpt);

    const step = state.stepData;
    if (step && Array.isArray(step.stages)) {
      const synStage = step.stages.find(function (s) { return s.key === "syndrome_extracted"; });
      const probs = synStage && synStage.syndrome_probabilities;
      if (probs && typeof probs === "object") {
        Object.keys(probs).forEach(function (synKey) {
          const pVal = Number(probs[synKey]) || 0;
          if (synKey !== "" && pVal > 1e-15) {
            const opt = document.createElement("option");
            opt.value = synKey;
            opt.textContent = synKey + " (" + (pVal * 100).toFixed(2) + "%)";
            synSel.appendChild(opt);
          }
        });
      }
    }
    const hasPrev = prev !== "" && Array.from(synSel.options).some(function (o) { return o.value === prev; });
    synSel.value = hasPrev ? prev : "";
    if (!hasPrev) {
      state.selectedSyndrome = null;
    }
  }

  function formatComplexCell(pair) {
    if (!Array.isArray(pair) || pair.length < 2) return "0";
    const re = Number(pair[0]);
    const im = Number(pair[1]);
    if (Math.abs(im) < 1e-9) return re.toFixed(4);
    const sign = im >= 0 ? "+" : "-";
    return re.toFixed(3) + sign + Math.abs(im).toFixed(3) + "i";
  }

  function renderCoherencePanel() {
    const cardsWrap = $("coherence-cards");
    const zeroHint = $("coherence-zero-hint");
    const mcBox = $("mc-result-box");
    const stateVal = $("state-select").value;

    if (zeroHint) {
      zeroHint.classList.toggle("hidden", stateVal !== "0" && stateVal !== "1");
    }

    if (cardsWrap) {
      cardsWrap.textContent = "";
      const comp = state.stepData && state.stepData.comparison;
      if (comp) {
        const entries = [
          { cls: "raw", title: t("coherence.compare.raw"), data: comp.raw },
          { cls: "qecc", title: t("coherence.compare.qecc", { code: comp.qecc.code || state.stepData.code }), data: comp.qecc },
          { cls: "dfs", title: t("coherence.compare.dfs"), data: comp.dfs_2 }
        ];
        entries.forEach(function (ent) {
          if (!ent.data) return;
          const card = document.createElement("div");
          card.className = "coh-card " + ent.cls;

          const h = document.createElement("h4");
          h.textContent = ent.title;
          card.appendChild(h);

          const fLine = document.createElement("div");
          fLine.className = "coh-stat";
          fLine.textContent = t("coherence.fidelity") + ": " + Number(ent.data.fidelity).toFixed(6);
          card.appendChild(fLine);

          const cLine = document.createElement("div");
          cLine.className = "coh-stat";
          cLine.textContent = t("coherence.abs_rho01") + ": " + Number(ent.data.abs_rho01).toFixed(6);
          card.appendChild(cLine);

          const bLine = document.createElement("div");
          bLine.className = "coh-stat";
          const bVec = Array.isArray(ent.data.bloch)
            ? "(" + ent.data.bloch.map(function (v) { return Number(v).toFixed(3); }).join(", ") + ")"
            : "—";
          bLine.textContent = t("coherence.bloch_vec") + ": " + bVec;
          card.appendChild(bLine);

          if (Array.isArray(ent.data.rho) && ent.data.rho.length === 2) {
            const tbl = document.createElement("table");
            tbl.className = "rho-matrix";
            ent.data.rho.forEach(function (row) {
              const tr = document.createElement("tr");
              row.forEach(function (cell) {
                const td = document.createElement("td");
                td.textContent = formatComplexCell(cell);
                tr.appendChild(td);
              });
              tbl.appendChild(tr);
            });
            card.appendChild(tbl);
          }

          cardsWrap.appendChild(card);
        });
      }
    }

    if (mcBox) {
      mcBox.textContent = "";
      const sim = state.simData;
      if (sim) {
        const h = document.createElement("h4");
        h.style.margin = "0 0 6px";
        h.textContent = t("sim.mc_summary", { backend: sim.backend_used });
        mcBox.appendChild(h);

        const grid = document.createElement("div");
        grid.className = "stage-metrics-grid";
        const items = [
          { k: t("sim.mean_fidelity"), v: Number(sim.mean_fidelity).toFixed(6) + " ± " + Number(sim.std_error).toFixed(6) },
          { k: t("coherence.abs_rho01"), v: Number(sim.abs_rho01).toFixed(6) },
          { k: t("sim.raw_baseline"), v: sim.raw_comparison ? Number(sim.raw_comparison.fidelity).toFixed(6) : "—" },
          { k: t("sim.elapsed"), v: Number(sim.elapsed_seconds).toFixed(4) + " s" }
        ];
        if (sim.analytical_entanglement_fidelity !== null && sim.analytical_entanglement_fidelity !== undefined) {
          items.push({ k: t("sim.ana_ent"), v: Number(sim.analytical_entanglement_fidelity).toFixed(6) });
        }
        if (sim.analytical_state_fidelity !== null && sim.analytical_state_fidelity !== undefined) {
          items.push({ k: t("sim.ana_state"), v: Number(sim.analytical_state_fidelity).toFixed(6) });
        }

        items.forEach(function (it) {
          const chip = document.createElement("div");
          chip.className = "metric-chip";
          const ks = document.createElement("span");
          ks.className = "k";
          ks.textContent = it.k;
          const vs = document.createElement("span");
          vs.className = "v";
          vs.textContent = it.v;
          chip.appendChild(ks);
          chip.appendChild(vs);
          grid.appendChild(chip);
        });
        mcBox.appendChild(grid);
      }
    }
  }

  function renderStepViews() {
    updateStageButtonsUi();
    const step = state.stepData;
    const code = step ? step.code : $("code-select").value;
    const metaCode = state.metadata && state.metadata.codes ? state.metadata.codes[code] : null;

    const errBadge = $("error-applied-badge");
    if (errBadge && step && step.error_applied) {
      errBadge.textContent = step.error_applied.label;
    }

    if (global.QECC_CIRCUIT) {
      global.QECC_CIRCUIT.renderCircuitSvg($("circuit-container"), step, metaCode, state.activeStageKey);
      global.QECC_CIRCUIT.renderStageDetails(
        $("stage-detail-card"),
        $("syndrome-dist-box"),
        $("pauli-coeffs-box"),
        step,
        state.activeStageKey,
        function (synKey) {
          state.selectedSyndrome = synKey;
          const synSel = $("syndrome-branch-select");
          if (synSel) synSel.value = synKey;
          runStep();
        }
      );
    }

    if (blochRenderer) {
      if (step && Array.isArray(step.stages)) {
        const activeStage = step.stages.find(function (s) { return s.key === state.activeStageKey; }) || step.stages[step.stages.length - 1];
        blochRenderer.setData({
          code: step.code,
          codeLabel: step.code,
          stageKey: activeStage ? activeStage.key : "decoded",
          input: step.input_state ? step.input_state.bloch : null,
          raw: step.comparison && step.comparison.raw ? step.comparison.raw.bloch : null,
          qecc: activeStage ? activeStage.logical_bloch : null,
          dfs: step.comparison && step.comparison.dfs_2 ? step.comparison.dfs_2.bloch : null
        });
      } else {
        blochRenderer.clear();
      }
    }

    renderCoherencePanel();
    syncSyndromeDropdown();
  }

  function renderMetricNoticeAndCharts() {
    const metric = $("metric-select").value || "entanglement";
    const notice = $("metric-notice-banner");
    if (notice) {
      notice.textContent = "";
      const strong = document.createElement("strong");
      strong.textContent = t("curves.metric_notice.title") + ": ";
      const span = document.createElement("span");
      span.textContent = metric === "state" ? t("curves.metric_notice.state") : t("curves.metric_notice.entanglement");
      notice.appendChild(strong);
      notice.appendChild(span);
    }

    const srcLbl = $("mc-source-label");
    if (srcLbl) {
      if (state.liveMcActive) {
        srcLbl.textContent = t("curves.mc_source.live", {
          shots: $("num-shots-input").value,
          backend: $("backend-select").value
        });
      } else {
        srcLbl.textContent = t("curves.mc_source.cached");
      }
    }

    if (global.QECC_CHARTS && state.sweepsData) {
      global.QECC_CHARTS.renderSweepsCharts(
        {
          curvesEl: $("curves-chart"),
          deltaEl: $("delta-chart"),
          cohEl: $("coherence-chart"),
          pstarTable: $("pstar-table")
        },
        state.sweepsData,
        $("code-select").value,
        $("channel-select").value
      );
    }
  }

  function renderTheoryPanel() {
    const root = $("theory-content");
    if (!root) return;
    root.textContent = "";
    const th = state.theoryData;
    if (!th) return;

    // 1. Stabilizers & Logical Ops
    const card1 = document.createElement("div");
    card1.className = "theory-card";
    const h1 = document.createElement("h3");
    h1.textContent = t("theory.stabilizers") + " — " + th.code;
    card1.appendChild(h1);

    if (Array.isArray(th.stabilizers) && th.stabilizers.length > 0) {
      const pills = document.createElement("div");
      pills.className = "stab-pills";
      th.stabilizers.forEach(function (s, idx) {
        const p = document.createElement("span");
        p.className = "stab-pill";
        p.textContent = "S" + idx + " = " + s;
        pills.appendChild(p);
      });
      card1.appendChild(pills);
    } else {
      const pNone = document.createElement("p");
      pNone.textContent = t("theory.stabilizers.none");
      card1.appendChild(pNone);
    }

    const hLog = document.createElement("h3");
    hLog.style.marginTop = "12px";
    hLog.textContent = t("theory.logical_ops");
    card1.appendChild(hLog);

    if (th.logical_ops) {
      const pills2 = document.createElement("div");
      pills2.className = "stab-pills";
      ["X_L", "Z_L"].forEach(function (k) {
        if (th.logical_ops[k]) {
          const sp = document.createElement("span");
          sp.className = "stab-pill";
          sp.textContent = k + " = " + th.logical_ops[k];
          pills2.appendChild(sp);
        }
      });
      card1.appendChild(pills2);
    }
    root.appendChild(card1);

    // 2. Syndrome Table
    const card2 = document.createElement("div");
    card2.className = "theory-card";
    const h2 = document.createElement("h3");
    h2.textContent = t("theory.syndrome_table");
    card2.appendChild(h2);

    const synEntries = th.syndrome_table ? Object.keys(th.syndrome_table) : [];
    if (synEntries.length === 0) {
      const emptyP = document.createElement("p");
      emptyP.textContent = t("theory.syndrome_table.empty");
      card2.appendChild(emptyP);
    } else {
      const tw = document.createElement("div");
      tw.className = "table-scroll";
      const tbl = document.createElement("table");
      tbl.className = "data-table";
      const thead = document.createElement("thead");
      const hr = document.createElement("tr");
      [t("theory.syndrome_col"), t("theory.recovery_col"), t("theory.target_col")].forEach(function (c) {
        const thEl = document.createElement("th");
        thEl.textContent = c;
        hr.appendChild(thEl);
      });
      thead.appendChild(hr);
      tbl.appendChild(thead);

      const tbody = document.createElement("tbody");
      synEntries.forEach(function (synKey) {
        const rec = th.syndrome_table[synKey];
        const tr = document.createElement("tr");
        const td1 = document.createElement("td");
        td1.className = "mono";
        td1.textContent = synKey;
        const td2 = document.createElement("td");
        td2.className = "mono";
        td2.textContent = rec;
        const td3 = document.createElement("td");
        const isIdent = /^I+$/.test(rec);
        td3.textContent = isIdent ? t("theory.no_error") : t("theory.correct_pauli", { op: rec });
        tr.appendChild(td1);
        tr.appendChild(td2);
        tr.appendChild(td3);
        tbody.appendChild(tr);
      });
      tbl.appendChild(tbody);
      tw.appendChild(tbl);
      card2.appendChild(tw);
    }
    root.appendChild(card2);

    // 3. Knill-Laflamme Verification & C_ab Matrix
    const card3 = document.createElement("div");
    card3.className = "theory-card wide";
    const h3 = document.createElement("h3");
    h3.textContent = t("theory.kl.title");
    card3.appendChild(h3);

    const kl = th.knill_laflamme;
    if (kl) {
      const metrics = document.createElement("div");
      metrics.className = "stage-metrics-grid";
      [
        { k: t("theory.kl.status"), v: kl.satisfies ? t("theory.kl.passed") : t("theory.kl.failed") },
        { k: t("theory.kl.max_violation"), v: Number(kl.max_violation).toExponential(3) },
        { k: t("theory.kl.max_diag_diff"), v: Number(kl.max_diagonal_diff).toExponential(3) },
        { k: t("theory.kl.max_off_diag"), v: Number(kl.max_off_diagonal).toExponential(3) }
      ].forEach(function (m) {
        const chip = document.createElement("div");
        chip.className = "metric-chip";
        const ks = document.createElement("span");
        ks.className = "k";
        ks.textContent = m.k;
        const vs = document.createElement("span");
        vs.className = "v";
        vs.textContent = m.v;
        chip.appendChild(ks);
        chip.appendChild(vs);
        metrics.appendChild(chip);
      });
      card3.appendChild(metrics);

      if (Array.isArray(kl.c_matrix) && Array.isArray(kl.error_basis)) {
        const cTitle = document.createElement("div");
        cTitle.style.fontSize = "0.82rem";
        cTitle.style.fontWeight = "600";
        cTitle.style.margin = "8px 0 6px";
        cTitle.textContent = t("theory.kl.c_matrix") + " (" + kl.error_basis.length + "×" + kl.error_basis.length + ")";
        card3.appendChild(cTitle);

        const cw = document.createElement("div");
        cw.className = "table-scroll";
        const cTbl = document.createElement("table");
        cTbl.className = "data-table mono";
        const cHead = document.createElement("thead");
        const cHr = document.createElement("tr");
        const corner = document.createElement("th");
        corner.textContent = "E_a \\ E_b";
        cHr.appendChild(corner);
        kl.error_basis.forEach(function (eb) {
          const thEl = document.createElement("th");
          thEl.textContent = eb;
          cHr.appendChild(thEl);
        });
        cHead.appendChild(cHr);
        cTbl.appendChild(cHead);

        const cBody = document.createElement("tbody");
        kl.c_matrix.forEach(function (row, rIdx) {
          const tr = document.createElement("tr");
          const rowTh = document.createElement("th");
          rowTh.textContent = kl.error_basis[rIdx] || String(rIdx);
          tr.appendChild(rowTh);
          row.forEach(function (cell) {
            const td = document.createElement("td");
            const re = Number(cell[0]);
            const im = Number(cell[1]);
            const mag = Math.hypot(re, im);
            td.style.backgroundColor = mag > 1e-6 ? "rgba(56, 189, 248, 0.22)" : "";
            td.textContent = Math.abs(im) < 1e-9 ? re.toFixed(2) : (re.toFixed(2) + (im >= 0 ? "+" : "-") + Math.abs(im).toFixed(2) + "i");
            tr.appendChild(td);
          });
          cBody.appendChild(tr);
        });
        cTbl.appendChild(cBody);
        cw.appendChild(cTbl);
        card3.appendChild(cw);
      }
    }
    root.appendChild(card3);

    // 4. Weight-2 Aliasing Examples (localized via i18n templates!)
    const card4 = document.createElement("div");
    card4.className = "theory-card wide";
    const h4 = document.createElement("h3");
    h4.textContent = t("theory.aliasing.title");
    card4.appendChild(h4);

    if (Array.isArray(th.aliasing_examples)) {
      const aw = document.createElement("div");
      aw.className = "table-scroll";
      const aTbl = document.createElement("table");
      aTbl.className = "data-table";
      const aHead = document.createElement("thead");
      const aHr = document.createElement("tr");
      [
        t("theory.aliasing.error"),
        t("theory.aliasing.syndrome"),
        t("theory.aliasing.recovery"),
        t("theory.aliasing.residual"),
        t("theory.aliasing.effect"),
        t("theory.aliasing.note_col")
      ].forEach(function (col) {
        const thEl = document.createElement("th");
        thEl.textContent = col;
        aHr.appendChild(thEl);
      });
      aHead.appendChild(aHr);
      aTbl.appendChild(aHead);

      const aBody = document.createElement("tbody");
      th.aliasing_examples.forEach(function (ex) {
        const tr = document.createElement("tr");
        let noteText = "";
        if (th.code === "dfs_2") {
          noteText = ex.logical_effect === "leakage"
            ? t("theory.aliasing.note_dfs_x", { error: ex.error, effect: ex.logical_effect })
            : t("theory.aliasing.note_dfs_z", { error: ex.error, effect: ex.logical_effect });
        } else {
          noteText = t("theory.aliasing.note_pauli", {
            error: ex.error,
            syndrome: ex.syndrome,
            recovery: ex.recovery,
            residual: ex.residual,
            effect: ex.logical_effect
          });
        }
        [
          ex.error,
          ex.syndrome === "" ? '""' : ex.syndrome,
          ex.recovery === "" ? '""' : ex.recovery,
          ex.residual,
          ex.logical_effect,
          noteText
        ].forEach(function (val, idx) {
          const td = document.createElement("td");
          if (idx < 5) td.className = "mono";
          td.textContent = val;
          tr.appendChild(td);
        });
        aBody.appendChild(tr);
      });
      aTbl.appendChild(aBody);
      aw.appendChild(aTbl);
      card4.appendChild(aw);
    }
    root.appendChild(card4);

    // 5. 1024-Term Weight Enumerator for [[5,1,3]]
    const card5 = document.createElement("div");
    card5.className = "theory-card wide";
    const h5 = document.createElement("h3");
    h5.textContent = t("theory.enum.title");
    card5.appendChild(h5);
    const sub5 = document.createElement("p");
    sub5.className = "hint-text";
    sub5.style.marginBottom = "8px";
    sub5.textContent = t("theory.enum.subtitle");
    card5.appendChild(sub5);

    const we = th.weight_enumerator;
    const sa = th.single_axis_weight_enumerator || {};
    if (we) {
      const ew = document.createElement("div");
      ew.className = "table-scroll";
      const eTbl = document.createElement("table");
      eTbl.className = "data-table mono";
      const eHead = document.createElement("thead");
      const eHr = document.createElement("tr");
      [
        t("theory.enum.weight"),
        t("theory.enum.coset_i"),
        t("theory.enum.coset_x"),
        t("theory.enum.coset_y"),
        t("theory.enum.coset_z"),
        t("theory.enum.total"),
        t("theory.enum.pure_x"),
        t("theory.enum.pure_z")
      ].forEach(function (col) {
        const thEl = document.createElement("th");
        thEl.textContent = col;
        eHr.appendChild(thEl);
      });
      eHead.appendChild(eHr);
      eTbl.appendChild(eHead);

      const eBody = document.createElement("tbody");
      const sums = { I: 0, X: 0, Y: 0, Z: 0, tot: 0, px: 0, pz: 0 };
      for (let w = 0; w <= 5; w++) {
        const wk = String(w);
        const ci = Number((we.I && we.I[wk]) || 0);
        const cx = Number((we.X && we.X[wk]) || 0);
        const cy = Number((we.Y && we.Y[wk]) || 0);
        const cz = Number((we.Z && we.Z[wk]) || 0);
        const tot = ci + cx + cy + cz;
        const px = Number((sa.X && (sa.X.I ? sa.X.I[wk] : sa.X[wk])) || 0);
        const pz = Number((sa.Z && (sa.Z.I ? sa.Z.I[wk] : sa.Z[wk])) || 0);
        sums.I += ci; sums.X += cx; sums.Y += cy; sums.Z += cz; sums.tot += tot; sums.px += px; sums.pz += pz;

        const tr = document.createElement("tr");
        [w, ci, cx, cy, cz, tot, px, pz].forEach(function (val) {
          const td = document.createElement("td");
          td.textContent = String(val);
          tr.appendChild(td);
        });
        eBody.appendChild(tr);
      }
      const sumTr = document.createElement("tr");
      sumTr.style.fontWeight = "700";
      [t("theory.enum.sum"), sums.I, sums.X, sums.Y, sums.Z, sums.tot, sums.px, sums.pz].forEach(function (val) {
        const td = document.createElement("td");
        td.textContent = String(val);
        sumTr.appendChild(td);
      });
      eBody.appendChild(sumTr);
      eTbl.appendChild(eBody);
      ew.appendChild(eTbl);
      card5.appendChild(ew);
    }
    root.appendChild(card5);
  }

  async function runStep() {
    const seqId = ++state.seq.step;
    const built = buildValidatedStepPayload();
    if (!built.ok) {
      setValidationError("step", built.err);
      return;
    }
    setValidationError("step", null);
    beginLoading();
    try {
      const data = await postJson("/api/step", built.payload);
      if (seqId !== state.seq.step) return;
      state.stepData = data;
      renderStepViews();
    } catch (err) {
      if (seqId === state.seq.step) {
        if (built.payload.selected_syndrome !== undefined) {
          clearSelectedSyndrome();
        }
        setValidationError("step", {
          key: "err.api_request",
          params: { detail: err.message || String(err) }
        });
      }
    } finally {
      endLoading();
    }
  }

  async function runSimulate() {
    const seqId = ++state.seq.sim;
    const built = buildValidatedSimPayload();
    if (!built.ok) {
      setValidationError("sim", built.err);
      return;
    }
    setValidationError("sim", null);
    beginLoading();
    try {
      const data = await postJson("/api/simulate", built.payload);
      if (seqId !== state.seq.sim) return;
      state.simData = data;
      renderCoherencePanel();
    } catch (err) {
      if (seqId === state.seq.sim) {
        setValidationError("sim", {
          key: "err.api_request",
          params: { detail: err.message || String(err) }
        });
      }
    } finally {
      endLoading();
    }
  }

  async function runSweeps(runLiveMc) {
    const seqId = ++state.seq.sweeps;
    const built = buildValidatedSweepsPayload(runLiveMc);
    if (!built.ok) {
      setValidationError("sweeps", built.err);
      return;
    }
    setValidationError("sweeps", null);
    if (global.QECC_CHARTS) {
      global.QECC_CHARTS.clearChartsForLoading($("curves-chart"), $("delta-chart"), $("coherence-chart"));
    }
    beginLoading();
    try {
      const data = await postJson("/api/sweeps", built.payload);
      if (seqId !== state.seq.sweeps) return;
      state.liveMcActive = !!runLiveMc;
      state.sweepsData = data;
      renderMetricNoticeAndCharts();
    } catch (err) {
      if (seqId === state.seq.sweeps) {
        setValidationError("sweeps", {
          key: "err.api_request",
          params: { detail: err.message || String(err) }
        });
        if (state.sweepsData) {
          renderMetricNoticeAndCharts();
        }
      }
    } finally {
      endLoading();
    }
  }

  async function loadTheory(codeName) {
    const code = codeName || $("code-select").value;
    const seqId = ++state.seq.theory;
    setValidationError("theory", null);
    beginLoading();
    try {
      const data = await getJson("/api/theory?code=" + encodeURIComponent(code));
      if (seqId !== state.seq.theory) return;
      state.theoryData = data;
      renderTheoryPanel();
    } catch (err) {
      if (seqId === state.seq.theory) {
        setValidationError("theory", {
          key: "err.api_request",
          params: { detail: err.message || String(err) }
        });
      }
    } finally {
      endLoading();
    }
  }

  function activateNavTab(targetSectionId, scroll) {
    document.querySelectorAll(".nav-tab").forEach(function (btn) {
      const match = btn.getAttribute("data-target") === targetSectionId;
      btn.classList.toggle("active", match);
      btn.setAttribute("aria-selected", match ? "true" : "false");
    });
    if (scroll) {
      const sec = $(targetSectionId);
      if (sec && sec.scrollIntoView) {
        sec.scrollIntoView({ behavior: "auto", block: "start" });
      }
    }
  }

  function applySeminarDemoPreset(preset) {
    if (!preset) return;
    const code = preset.code || "five_qubit";
    $("code-select").value = code;
    if (preset.channel) $("channel-select").value = preset.channel;
    if (preset.state_preset) $("state-select").value = preset.state_preset;
    if (preset.p !== undefined) $("p-input").value = String(preset.p);
    if (preset.collective_phi !== undefined) $("phi-input").value = String(preset.collective_phi);
    if (preset.error_mode) $("error-mode-select").value = preset.error_mode;
    if (preset.pauli_string) {
      $("pauli-string-input").value = preset.pauli_string;
    } else {
      $("pauli-string-input").value = DEFAULT_PAULI[code] || "IXIII";
    }
    if (preset.su2_qubit !== undefined) $("su2-qubit-input").value = String(preset.su2_qubit);
    if (preset.su2_theta !== undefined) $("su2-theta-input").value = String(preset.su2_theta);
    if (preset.su2_nx !== undefined) $("su2-nx-input").value = String(preset.su2_nx);
    if (preset.su2_ny !== undefined) $("su2-ny-input").value = String(preset.su2_ny);
    if (preset.su2_nz !== undefined) $("su2-nz-input").value = String(preset.su2_nz);
    if (preset.metric) $("metric-select").value = preset.metric;
    if (preset.stage_key) state.activeStageKey = preset.stage_key;
    clearSelectedSyndrome();

    updateConditionalFieldVisibility();

    const tabMap = {
      simulator: "section-simulator",
      curves: "section-curves",
      theory: "section-theory",
      seminar: "section-seminar"
    };
    const secId = tabMap[preset.target_tab] || "section-simulator";
    activateNavTab(secId, true);

    showToast("seminar.demo_loaded", { code: code });
    runStep();
    runSimulate();
    loadTheory(code);
    if (preset.metric) {
      runSweeps(false);
    } else {
      renderMetricNoticeAndCharts();
    }
  }

  function bindEventHandlers() {
    // Language switcher
    const btnEn = document.querySelector('[data-testid="lang-en"]');
    const btnPt = document.querySelector('[data-testid="lang-pt-br"]');
    if (btnEn) {
      btnEn.addEventListener("click", function () {
        global.QECC_I18N.setLanguage("en");
      });
    }
    if (btnPt) {
      btnPt.addEventListener("click", function () {
        global.QECC_I18N.setLanguage("pt-br");
      });
    }

    global.QECC_I18N.onLanguageChange(function () {
      translateSelectOptionsPreservingValues();
      updateConditionalFieldVisibility();
      renderStepViews();
      renderMetricNoticeAndCharts();
      renderTheoryPanel();
      if (seminarCtrl) seminarCtrl.render();
      renderValidationErrors();
      renderToast();
      renderStatusBadge();
    });

    // Primary Nav Tabs
    document.querySelectorAll(".nav-tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        activateNavTab(btn.getAttribute("data-target"), true);
      });
    });

    // Code select
    $("code-select").addEventListener("change", function () {
      const code = $("code-select").value;
      $("pauli-string-input").value = DEFAULT_PAULI[code] || "IXIII";
      clearSelectedSyndrome();
      updateConditionalFieldVisibility();
      runStep();
      runSimulate();
      loadTheory(code);
      renderMetricNoticeAndCharts();
    });

    // Channel select
    $("channel-select").addEventListener("change", function () {
      const em = $("error-mode-select").value;
      if (em !== "stochastic_sample") {
        $("error-mode-select").value = "stochastic_sample";
        updateConditionalFieldVisibility();
      }
      clearSelectedSyndrome();
      runStep();
      runSimulate();
      renderMetricNoticeAndCharts();
    });

    // State select
    $("state-select").addEventListener("change", function () {
      clearSelectedSyndrome();
      updateConditionalFieldVisibility();
      runStep();
      runSimulate();
    });

    // Error mode select
    $("error-mode-select").addEventListener("change", function () {
      clearSelectedSyndrome();
      updateConditionalFieldVisibility();
      runStep();
    });

    // p-input
    const onPChange = function () {
      const p = parseFiniteInput($("p-input"));
      if (!Number.isFinite(p) || p < 0 || p > 1) {
        ++state.seq.sim;
        setValidationError("sim", { key: "err.invalid_p" });
        if ($("error-mode-select").value === "stochastic_sample") {
          ++state.seq.step;
          setValidationError("step", { key: "err.invalid_p" });
        }
        return;
      }
      clearValidationErrorIfKey("sim", "err.invalid_p");
      clearValidationErrorIfKey("step", "err.invalid_p");
      if ($("error-mode-select").value !== "stochastic_sample") {
        $("error-mode-select").value = "stochastic_sample";
        updateConditionalFieldVisibility();
      }
      clearSelectedSyndrome();
      runStep();
      runSimulate();
    };
    $("p-input").addEventListener("change", onPChange);
    $("p-input").addEventListener("input", function () {
      const p = parseFiniteInput($("p-input"));
      if (!Number.isFinite(p) || p < 0 || p > 1) {
        setValidationError("sim", { key: "err.invalid_p" });
        if ($("error-mode-select").value === "stochastic_sample") {
          setValidationError("step", { key: "err.invalid_p" });
        }
      } else {
        clearValidationErrorIfKey("sim", "err.invalid_p");
        clearValidationErrorIfKey("step", "err.invalid_p");
      }
    });

    // phi-input
    const onPhiChange = function () {
      const phi = parseFiniteInput($("phi-input"));
      if (!Number.isFinite(phi)) {
        ++state.seq.step;
        setValidationError("step", { key: "err.invalid_phi" });
        return;
      }
      clearValidationErrorIfKey("step", "err.invalid_phi");
      $("error-mode-select").value = "coherent_collective";
      updateConditionalFieldVisibility();
      clearSelectedSyndrome();
      runStep();
    };
    $("phi-input").addEventListener("change", onPhiChange);

    // Custom / Bloch / Mode sub-inputs
    ["theta-input", "phi-state-input", "alpha-re-input", "alpha-im-input", "beta-re-input", "beta-im-input"].forEach(function (id) {
      const el = $(id);
      if (el) {
        el.addEventListener("change", function () {
          clearSelectedSyndrome();
          runStep();
          runSimulate();
        });
      }
    });

    $("normalize-custom-btn").addEventListener("click", function () {
      const ar = parseFiniteInput($("alpha-re-input")) || 0;
      const ai = parseFiniteInput($("alpha-im-input")) || 0;
      const br = parseFiniteInput($("beta-re-input")) || 0;
      const bi = parseFiniteInput($("beta-im-input")) || 0;
      const norm = Math.hypot(ar, ai, br, bi);
      if (norm > 1e-12) {
        $("alpha-re-input").value = (ar / norm).toFixed(8);
        $("alpha-im-input").value = (ai / norm).toFixed(8);
        $("beta-re-input").value = (br / norm).toFixed(8);
        $("beta-im-input").value = (bi / norm).toFixed(8);
      } else {
        $("alpha-re-input").value = "1";
        $("alpha-im-input").value = "0";
        $("beta-re-input").value = "0";
        $("beta-im-input").value = "0";
      }
      clearSelectedSyndrome();
      runStep();
      runSimulate();
    });

    ["pauli-string-input", "error-qubit-input", "error-op-select", "su2-qubit-input", "su2-theta-input", "su2-nx-input", "su2-ny-input", "su2-nz-input"].forEach(function (id) {
      const el = $(id);
      if (el) {
        el.addEventListener("change", function () {
          clearSelectedSyndrome();
          runStep();
        });
      }
    });

    $("syndrome-branch-select").addEventListener("change", function () {
      const val = $("syndrome-branch-select").value;
      state.selectedSyndrome = val === "" ? null : val;
      runStep();
    });

    const onSeedChange = function () {
      const seedParsed = parseOptionalSeedInput();
      if (!seedParsed.ok) {
        ++state.seq.sim;
        setValidationError("sim", seedParsed.err);
        if ($("error-mode-select").value === "stochastic_sample") {
          ++state.seq.step;
          setValidationError("step", seedParsed.err);
        }
        return;
      }
      clearValidationErrorIfKey("sim", "err.invalid_seed");
      clearValidationErrorIfKey("step", "err.invalid_seed");
      clearSelectedSyndrome();
      runStep();
      runSimulate();
    };
    $("seed-input").addEventListener("change", onSeedChange);
    $("seed-input").addEventListener("input", function () {
      const seedParsed = parseOptionalSeedInput();
      if (!seedParsed.ok) {
        setValidationError("sim", seedParsed.err);
        if ($("error-mode-select").value === "stochastic_sample") {
          setValidationError("step", seedParsed.err);
        }
      } else {
        clearValidationErrorIfKey("sim", "err.invalid_seed");
        clearValidationErrorIfKey("step", "err.invalid_seed");
      }
    });

    $("num-shots-input").addEventListener("change", function () {
      const shots = parseFiniteInput($("num-shots-input"));
      if (Number.isInteger(shots) && shots >= 1 && shots <= 100000) {
        clearValidationErrorIfKey("sweeps", "err.invalid_sweep_shots");
      }
      runSimulate();
    });
    $("backend-select").addEventListener("change", function () {
      if ($("backend-select").value !== "eigenkit_unitary") {
        clearValidationErrorIfKey("sweeps", "err.eigenkit_stochastic");
      }
      runSimulate();
    });

    ["sweep-pmin", "sweep-pmax", "sweep-nump"].forEach(function (id) {
      const el = $(id);
      if (el) {
        el.addEventListener("change", function () {
          const pMin = parseFiniteInput($("sweep-pmin"));
          const pMax = parseFiniteInput($("sweep-pmax"));
          const numP = parseFiniteInput($("sweep-nump"));
          if (Number.isFinite(pMin) && Number.isFinite(pMax) && pMin >= 0 && pMax <= 1 && pMin < pMax && Number.isInteger(numP) && numP >= 2 && numP <= 501) {
            clearValidationErrorIfKey("sweeps", "err.invalid_sweep_range");
          }
        });
      }
    });

    $("run-step-btn").addEventListener("click", runStep);
    $("run-sim-btn").addEventListener("click", runSimulate);
    $("resample-seed-btn").addEventListener("click", function () {
      const nextSeed = Math.floor(Math.random() * 1000000) + 1;
      $("seed-input").value = String(nextSeed);
      if ($("error-mode-select").value !== "stochastic_sample") {
        $("error-mode-select").value = "stochastic_sample";
        updateConditionalFieldVisibility();
      }
      clearSelectedSyndrome();
      runStep();
      runSimulate();
    });

    // 6-stage buttons
    STAGE_KEYS.forEach(function (k) {
      const btn = document.querySelector('[data-testid="stage-step-' + k + '"]');
      if (btn) {
        btn.addEventListener("click", function () {
          state.activeStageKey = k;
          renderStepViews();
        });
      }
    });

    // Bloch reset
    $("bloch-reset-btn").addEventListener("click", function () {
      if (blochRenderer) blochRenderer.resetView();
    });

    // Sweeps controls
    $("metric-select").addEventListener("change", function () {
      runSweeps(state.liveMcActive);
    });
    $("refresh-sweeps-btn").addEventListener("click", function () {
      runSweeps(false);
    });
    $("run-live-mc-btn").addEventListener("click", function () {
      runSweeps(true);
    });
  }

  async function init() {
    global.QECC_I18N.applyStaticDomTranslations();
    blochRenderer = new global.QECC_BLOCH.BlochRenderer($("bloch-canvas"), $("bloch-legend"));
    seminarCtrl = new global.QECC_SEMINAR.SeminarController($("seminar-root"), applySeminarDemoPreset);

    translateSelectOptionsPreservingValues();
    updateConditionalFieldVisibility();
    bindEventHandlers();

    beginLoading();
    try {
      const results = await Promise.all([getJson("/api/health"), getJson("/api/metadata")]);
      state.metadata = results[1];
      setValidationError("init", null);
      translateSelectOptionsPreservingValues();
      updateConditionalFieldVisibility();
    } catch (err) {
      setValidationError("init", {
        key: "err.api_request",
        params: { detail: err.message || String(err) }
      });
    } finally {
      endLoading();
    }

    await Promise.all([
      runStep(),
      runSimulate(),
      runSweeps(false),
      loadTheory($("code-select").value)
    ]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(window);
