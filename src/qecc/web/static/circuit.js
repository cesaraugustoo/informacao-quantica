(function (global) {
  "use strict";

  const SVG_NS = ["http:", "", "www.w3.org", "2000", "svg"].join("/");

  function svgEl(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        node.setAttribute(k, String(attrs[k]));
      });
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function formatGateList(ops) {
    if (!Array.isArray(ops) || ops.length === 0) return "";
    return ops.map(function (op) {
      const ctrl = op.controls && op.controls.length ? "c[" + op.controls.join(",") + "]→" : "";
      const targ = op.targets && op.targets.length ? op.targets.join(",") : "";
      const param = op.params && op.params.length ? "(" + op.params.map(function (v) { return Number(v).toFixed(3); }).join(",") + ")" : "";
      return op.name + param + "(" + ctrl + targ + ")";
    }).join(" · ");
  }

  function renderCircuitSvg(container, stepData, metaCode, activeStageKey) {
    if (!container) return;
    const t = global.QECC_I18N ? global.QECC_I18N.t : function (k) { return k; };
    container.textContent = "";

    const nData = stepData ? stepData.n_data : (metaCode ? metaCode.n_data : 3);
    const nAncilla = stepData ? stepData.n_ancilla : (metaCode ? metaCode.n_ancilla : 2);
    const nTotal = nData + nAncilla;

    const stagesByKey = {};
    if (stepData && Array.isArray(stepData.stages)) {
      stepData.stages.forEach(function (st) {
        stagesByKey[st.key] = st;
      });
    }

    const encodeOps = (stagesByKey.encoded && stagesByKey.encoded.gate_ops) || (metaCode && metaCode.encode_ops) || [];
    const noiseOps = (stepData && stepData.error_applied && stepData.error_applied.ops) || [];
    const syndromeOps = (stagesByKey.syndrome_extracted && stagesByKey.syndrome_extracted.gate_ops) || (metaCode && metaCode.syndrome_ops) || [];
    const recoveryOps = (stagesByKey.recovered && stagesByKey.recovered.recovery_ops) || [];
    const decodeOps = (stagesByKey.decoded && stagesByKey.decoded.gate_ops) || (metaCode && metaCode.decode_ops) || [];

    const columns = [
      { key: "encoded", label: t("circuit.stage_encode"), ops: encodeOps, color: "#38bdf8" },
      { key: "noisy", label: t("circuit.stage_noise"), ops: noiseOps, color: "#fb7185", isNoise: true },
      { key: "syndrome_extracted", label: t("circuit.stage_syndrome"), ops: syndromeOps, color: "#a78bfa", hasMeasure: nAncilla > 0 },
      { key: "recovered", label: t("circuit.stage_recovery"), ops: recoveryOps, color: "#34d399" },
      { key: "decoded", label: t("circuit.stage_decode"), ops: decodeOps, color: "#60a5fa" }
    ];

    // Assign sub-columns per gate op (compressed slightly if large)
    const leftPad = 108;
    const colGap = 24;
    const gateStep = 28;
    let cursorX = leftPad;
    const colLayouts = columns.map(function (col) {
      const nOps = Math.max(1, col.ops.length + (col.hasMeasure ? 1 : 0));
      const width = Math.max(76, nOps * gateStep + 14);
      const layout = {
        key: col.key,
        label: col.label,
        ops: col.ops,
        color: col.color,
        isNoise: !!col.isNoise,
        hasMeasure: !!col.hasMeasure,
        xStart: cursorX,
        width: width
      };
      cursorX += width + colGap;
      return layout;
    });

    const totalW = Math.max(660, cursorX + 16);
    const wireGap = 32;
    const topPad = 36;
    const totalH = topPad + nTotal * wireGap + 24;

    const svg = svgEl("svg", {
      viewBox: "0 0 " + totalW + " " + totalH,
      width: totalW,
      height: totalH,
      role: "img",
      "aria-label": t("circuit.title")
    });

    // Background
    svg.appendChild(svgEl("rect", { x: 0, y: 0, width: totalW, height: totalH, rx: 8, fill: "#0e1729" }));

    // Stage highlight boxes
    colLayouts.forEach(function (col) {
      const isActive = col.key === activeStageKey;
      svg.appendChild(svgEl("rect", {
        x: col.xStart - 6,
        y: 8,
        width: col.width + 12,
        height: totalH - 16,
        rx: 6,
        fill: isActive ? "rgba(56, 189, 248, 0.14)" : "rgba(22, 34, 56, 0.45)",
        stroke: isActive ? col.color : "#253552",
        "stroke-width": isActive ? "1.8" : "1"
      }));
      svg.appendChild(svgEl("text", {
        x: col.xStart + col.width / 2,
        y: 23,
        "text-anchor": "middle",
        fill: isActive ? "#f8fafc" : "#9fb0cb",
        "font-size": "11",
        "font-weight": "700"
      }, col.label));
    });

    // Qubit wires and labels
    for (let q = 0; q < nTotal; q++) {
      const y = topPad + q * wireGap + 16;
      const isAncilla = q >= nData;
      const label = isAncilla
        ? t("circuit.ancilla_qubit", { idx: q - nData, q: q })
        : t("circuit.data_qubit", { idx: q });

      svg.appendChild(svgEl("text", {
        x: 10,
        y: y + 4,
        fill: isAncilla ? "#c4b5fd" : "#e2e8f0",
        "font-size": "11",
        "font-family": "ui-monospace, monospace",
        "font-weight": "600"
      }, label));

      svg.appendChild(svgEl("line", {
        x1: leftPad - 10,
        y1: y,
        x2: totalW - 14,
        y2: y,
        stroke: isAncilla ? "#4c1d95" : "#374f78",
        "stroke-width": "1.5",
        "stroke-dasharray": isAncilla ? "4,2" : "none"
      }));
    }

    // Draw gates in each column
    colLayouts.forEach(function (col) {
      if (col.ops.length === 0) {
        svg.appendChild(svgEl("text", {
          x: col.xStart + col.width / 2,
          y: topPad + (nTotal * wireGap) / 2 + 12,
          "text-anchor": "middle",
          fill: "#64748b",
          "font-size": "11",
          "font-family": "ui-monospace, monospace"
        }, col.isNoise ? "I (no-op)" : "—"));
      }

      col.ops.forEach(function (op, idx) {
        const gx = col.xStart + 16 + idx * gateStep;
        const targets = op.targets || [];
        const controls = op.controls || [];
        const allQ = controls.concat(targets);
        if (allQ.length === 0) return;

        const ys = allQ.map(function (q) { return topPad + q * wireGap + 16; });
        const minY = Math.min.apply(null, ys);
        const maxY = Math.max.apply(null, ys);

        if (allQ.length > 1) {
          svg.appendChild(svgEl("line", {
            x1: gx,
            y1: minY,
            x2: gx,
            y2: maxY,
            stroke: col.color,
            "stroke-width": "1.6"
          }));
        }

        controls.forEach(function (cq) {
          const cy = topPad + cq * wireGap + 16;
          svg.appendChild(svgEl("circle", {
            cx: gx,
            cy: cy,
            r: 4,
            fill: col.color
          }));
        });

        targets.forEach(function (tq) {
          const ty = topPad + tq * wireGap + 16;
          const upName = String(op.name || "").toUpperCase();
          if ((upName === "CNOT" || upName === "CX" || upName === "CCNOT" || upName === "TOFFOLI") && controls.length > 0) {
            svg.appendChild(svgEl("circle", {
              cx: gx,
              cy: ty,
              r: 8,
              fill: "#0b1220",
              stroke: col.color,
              "stroke-width": "1.8"
            }));
            svg.appendChild(svgEl("line", { x1: gx - 5, y1: ty, x2: gx + 5, y2: ty, stroke: col.color, "stroke-width": "1.6" }));
            svg.appendChild(svgEl("line", { x1: gx, y1: ty - 5, x2: gx, y2: ty + 5, stroke: col.color, "stroke-width": "1.6" }));
          } else {
            const shortLabel = upName.length > 4 ? upName.slice(0, 4) : upName;
            svg.appendChild(svgEl("rect", {
              x: gx - 11,
              y: ty - 10,
              width: 22,
              height: 20,
              rx: 4,
              fill: col.isNoise ? "#4c0519" : "#162238",
              stroke: col.color,
              "stroke-width": "1.4"
            }));
            svg.appendChild(svgEl("text", {
              x: gx,
              y: ty + 3.5,
              "text-anchor": "middle",
              fill: "#f8fafc",
              "font-size": "9.5",
              "font-family": "ui-monospace, monospace",
              "font-weight": "700"
            }, shortLabel));
          }
        });
      });

      if (col.hasMeasure) {
        const mx = col.xStart + col.width - 18;
        for (let aq = nData; aq < nTotal; aq++) {
          const ay = topPad + aq * wireGap + 16;
          svg.appendChild(svgEl("rect", {
            x: mx - 10,
            y: ay - 10,
            width: 20,
            height: 20,
            rx: 4,
            fill: "#2e1065",
            stroke: "#a78bfa",
            "stroke-width": "1.4"
          }));
          svg.appendChild(svgEl("text", {
            x: mx,
            y: ay + 3.5,
            "text-anchor": "middle",
            fill: "#ede9fe",
            "font-size": "9.5",
            "font-weight": "700"
          }, "M"));
        }
      }
    });

    container.appendChild(svg);
  }

  function renderStageDetails(detailEl, syndromeEl, pauliEl, stepData, activeStageKey, onSelectSyndrome) {
    const t = global.QECC_I18N ? global.QECC_I18N.t : function (k) { return k; };
    if (!stepData || !Array.isArray(stepData.stages)) {
      if (detailEl) detailEl.textContent = "";
      return;
    }

    const stage = stepData.stages.find(function (s) { return s.key === activeStageKey; }) || stepData.stages[stepData.stages.length - 1];
    const isDfs = stepData.code === "dfs_2";

    if (detailEl) {
      detailEl.textContent = "";
      const title = document.createElement("h4");
      title.textContent = t("stage." + stage.key + ".title");
      detailEl.appendChild(title);

      const desc = document.createElement("p");
      desc.style.margin = "0 0 8px";
      let descKey = "stage." + stage.key + ".desc";
      if (isDfs && (stage.key === "syndrome_extracted" || stage.key === "recovered")) {
        descKey = "stage." + stage.key + ".desc_dfs";
      }
      desc.textContent = t(descKey);
      detailEl.appendChild(desc);

      const metricsGrid = document.createElement("div");
      metricsGrid.className = "stage-metrics-grid";

      const blochStr = Array.isArray(stage.logical_bloch)
        ? "(" + stage.logical_bloch.map(function (v) { return Number(v).toFixed(4); }).join(", ") + ")"
        : "—";

      const chips = [
        { k: t("stage.metric.fidelity"), v: Number(stage.fidelity_vs_input).toFixed(6) },
        { k: t("stage.metric.subspace_prob"), v: Number(stage.code_subspace_prob).toFixed(6) },
        { k: t("stage.metric.abs_rho01"), v: Number(stage.logical_abs_rho01).toFixed(6) },
        { k: t("stage.metric.logical_bloch"), v: blochStr }
      ];

      if (stage.selected_syndrome !== undefined && stage.selected_syndrome !== null) {
        chips.push({
          k: t("stage.metric.selected_syndrome"),
          v: isDfs ? t("stage.metric.dfs_empty_syndrome") : stage.selected_syndrome
        });
      }

      chips.forEach(function (ch) {
        const card = document.createElement("div");
        card.className = "metric-chip";
        const kSpan = document.createElement("span");
        kSpan.className = "k";
        kSpan.textContent = ch.k;
        const vSpan = document.createElement("span");
        vSpan.className = "v";
        vSpan.textContent = ch.v;
        card.appendChild(kSpan);
        card.appendChild(vSpan);
        metricsGrid.appendChild(card);
      });
      detailEl.appendChild(metricsGrid);

      // Gates applied
      const opsToShow = stage.key === "recovered" && Array.isArray(stage.recovery_ops) && stage.recovery_ops.length
        ? stage.recovery_ops
        : stage.gate_ops;
      const gatesLine = document.createElement("div");
      gatesLine.style.fontSize = "0.8rem";
      gatesLine.style.marginBottom = "8px";
      const gLabel = document.createElement("strong");
      gLabel.textContent = t("stage.metric.gates_applied") + ": ";
      const gVal = document.createElement("span");
      gVal.className = "mono";
      gVal.textContent = opsToShow && opsToShow.length ? formatGateList(opsToShow) : t("stage.metric.none");
      gatesLine.appendChild(gLabel);
      gatesLine.appendChild(gVal);
      detailEl.appendChild(gatesLine);

      // Non-zero amplitudes
      if (Array.isArray(stage.statevector_nonzero) && stage.statevector_nonzero.length) {
        const ampTitle = document.createElement("div");
        ampTitle.style.fontSize = "0.78rem";
        ampTitle.style.fontWeight = "600";
        ampTitle.style.color = "#9fb0cb";
        ampTitle.textContent = t("stage.metric.nonzero_amplitudes");
        detailEl.appendChild(ampTitle);

        const ampWrap = document.createElement("div");
        ampWrap.className = "amp-chips";
        stage.statevector_nonzero.slice(0, 8).forEach(function (term) {
          const chip = document.createElement("span");
          chip.className = "amp-chip";
          const sign = term.im >= 0 ? "+" : "-";
          chip.textContent = "|" + term.basis + "⟩: " + Number(term.re).toFixed(3) + sign + Math.abs(Number(term.im)).toFixed(3) + "i (" + (Number(term.prob) * 100).toFixed(1) + "%)";
          ampWrap.appendChild(chip);
        });
        detailEl.appendChild(ampWrap);
      }
    }

    // Syndrome probabilities from syndrome_extracted stage
    const synStage = stepData.stages.find(function (s) { return s.key === "syndrome_extracted"; });
    if (syndromeEl) {
      syndromeEl.textContent = "";
      const probs = synStage && synStage.syndrome_probabilities;
      if (probs && typeof probs === "object") {
        const h = document.createElement("h4");
        h.className = "subsection-title";
        h.style.marginBottom = "6px";
        h.textContent = t("stage.metric.syndrome_probs");
        syndromeEl.appendChild(h);

        const bars = document.createElement("div");
        bars.className = "syn-bars";
        const activeSyn = synStage.selected_syndrome;
        Object.keys(probs).forEach(function (synKey) {
          const pVal = Number(probs[synKey]) || 0;
          const row = document.createElement("button");
          row.type = "button";
          row.className = "syn-bar-row" + (synKey === activeSyn ? " selected" : "");
          row.disabled = pVal <= 1e-15 || isDfs;
          row.addEventListener("click", function () {
            if (pVal > 1e-15 && typeof onSelectSyndrome === "function") {
              onSelectSyndrome(synKey);
            }
          });

          const lbl = document.createElement("span");
          lbl.className = "mono";
          lbl.textContent = synKey === "" ? '""' : synKey;

          const track = document.createElement("div");
          track.className = "syn-bar-track";
          const fill = document.createElement("div");
          fill.className = "syn-bar-fill";
          fill.style.width = Math.max(0, Math.min(100, pVal * 100)).toFixed(1) + "%";
          track.appendChild(fill);

          const pct = document.createElement("span");
          pct.className = "mono";
          pct.style.textAlign = "right";
          pct.textContent = (pVal * 100).toFixed(2) + "%";

          row.appendChild(lbl);
          row.appendChild(track);
          row.appendChild(pct);
          bars.appendChild(row);
        });
        syndromeEl.appendChild(bars);
      }
    }

    // Pauli coefficients box (for su2_rotation / coherent_collective)
    if (pauliEl) {
      const pc = stepData.error_applied && stepData.error_applied.pauli_coefficients;
      pauliEl.textContent = "";
      if (pc && typeof pc === "object") {
        pauliEl.classList.remove("hidden");
        const h = document.createElement("h4");
        h.className = "subsection-title";
        h.style.marginBottom = "6px";
        h.textContent = t("stage.metric.pauli_coeffs");
        pauliEl.appendChild(h);

        const tblWrap = document.createElement("div");
        tblWrap.className = "table-scroll";
        const tbl = document.createElement("table");
        tbl.className = "data-table";
        const thead = document.createElement("thead");
        const hr = document.createElement("tr");
        ["Pauli", "Re(c_P)", "Im(c_P)", "|c_P|²"].forEach(function (col) {
          const th = document.createElement("th");
          th.textContent = col;
          hr.appendChild(th);
        });
        thead.appendChild(hr);
        tbl.appendChild(thead);

        const tbody = document.createElement("tbody");
        ["I", "X", "Y", "Z"].forEach(function (pName) {
          const item = pc[pName];
          if (!item) return;
          const tr = document.createElement("tr");
          [pName, Number(item.re).toFixed(6), Number(item.im).toFixed(6), Number(item.prob).toFixed(6)].forEach(function (val) {
            const td = document.createElement("td");
            td.className = "mono";
            td.textContent = val;
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
        tbl.appendChild(tbody);
        tblWrap.appendChild(tbl);
        pauliEl.appendChild(tblWrap);
      } else {
        pauliEl.classList.add("hidden");
      }
    }
  }

  global.QECC_CIRCUIT = {
    renderCircuitSvg: renderCircuitSvg,
    renderStageDetails: renderStageDetails
  };
})(window);
