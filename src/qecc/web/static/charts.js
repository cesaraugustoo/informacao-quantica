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

  const SERIES_COLORS = {
    "bit_flip_3/bit_flip": "#38bdf8",
    "phase_flip_3/phase_flip": "#a78bfa",
    "five_qubit/depolarizing": "#fbbf24",
    "five_qubit/bit_flip": "#fb923c",
    "five_qubit/phase_flip": "#f472b6",
    "dfs_2/collective_dephasing": "#34d399",
    "dfs_2/phase_flip": "#64748b"
  };

  function getSeriesColor(key, idx) {
    if (SERIES_COLORS[key]) return SERIES_COLORS[key];
    const palette = ["#38bdf8", "#a78bfa", "#fbbf24", "#34d399", "#f472b6", "#60a5fa", "#fb923c"];
    return palette[idx % palette.length];
  }

  function pickDisplayKeys(analyticalCurves, activeCode, activeChannel) {
    const preferred = [
      "bit_flip_3/bit_flip",
      "phase_flip_3/phase_flip",
      "five_qubit/depolarizing",
      "five_qubit/bit_flip",
      "dfs_2/collective_dephasing",
      "dfs_2/phase_flip"
    ];
    const keys = [];
    const activeKey = activeCode + "/" + activeChannel;
    if (analyticalCurves[activeKey]) {
      keys.push(activeKey);
    }
    preferred.forEach(function (k) {
      if (analyticalCurves[k] && keys.indexOf(k) === -1) {
        keys.push(k);
      }
    });
    if (keys.length === 0) {
      Object.keys(analyticalCurves).slice(0, 6).forEach(function (k) {
        keys.push(k);
      });
    }
    return keys;
  }

  function buildLineSvg(opts) {
    const w = opts.width || 720;
    const h = opts.height || 310;
    const padL = 62;
    const padR = 24;
    const padT = 26;
    const padB = 46;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const pGrid = opts.pGrid || [0, 0.5];
    const xMin = pGrid[0];
    const xMax = pGrid[pGrid.length - 1] > xMin ? pGrid[pGrid.length - 1] : xMin + 0.5;
    const yMin = opts.yMin;
    const yMax = opts.yMax;

    const mapX = function (x) {
      return padL + ((x - xMin) / (xMax - xMin)) * plotW;
    };
    const mapY = function (y) {
      const clamped = Math.max(yMin, Math.min(yMax, y));
      return padT + (1 - (clamped - yMin) / (yMax - yMin)) * plotH;
    };

    const svg = svgEl("svg", {
      viewBox: "0 0 " + w + " " + h,
      width: w,
      height: h,
      role: "img",
      "aria-label": opts.ariaLabel || "Chart"
    });

    svg.appendChild(svgEl("rect", { x: 0, y: 0, width: w, height: h, rx: 8, fill: "#0e1729" }));

    // Grid lines (5 Y ticks, 5 X ticks)
    const nTicks = 5;
    for (let i = 0; i <= nTicks; i++) {
      const frac = i / nTicks;
      const yVal = yMin + frac * (yMax - yMin);
      const yy = mapY(yVal);
      svg.appendChild(svgEl("line", {
        x1: padL,
        y1: yy,
        x2: w - padR,
        y2: yy,
        stroke: Math.abs(yVal) < 1e-12 ? "#475569" : "#1e293b",
        "stroke-width": Math.abs(yVal) < 1e-12 ? "1.4" : "1"
      }));
      svg.appendChild(svgEl("text", {
        x: padL - 8,
        y: yy + 3.5,
        "text-anchor": "end",
        fill: "#9fb0cb",
        "font-size": "10.5",
        "font-family": "ui-monospace, monospace"
      }, yVal.toFixed(2)));

      const xVal = xMin + frac * (xMax - xMin);
      const xx = mapX(xVal);
      svg.appendChild(svgEl("line", {
        x1: xx,
        y1: padT,
        x2: xx,
        y2: h - padB,
        stroke: "#1e293b",
        "stroke-width": "1"
      }));
      svg.appendChild(svgEl("text", {
        x: xx,
        y: h - padB + 16,
        "text-anchor": "middle",
        fill: "#9fb0cb",
        "font-size": "10.5",
        "font-family": "ui-monospace, monospace"
      }, xVal.toFixed(2)));
    }

    // Plot border
    svg.appendChild(svgEl("rect", {
      x: padL,
      y: padT,
      width: plotW,
      height: plotH,
      fill: "none",
      stroke: "#374f78",
      "stroke-width": "1.2"
    }));

    // Optional baseline curve
    if (Array.isArray(opts.baseline) && opts.baseline.length === pGrid.length) {
      const pts = pGrid.map(function (px, idx) {
        return mapX(px).toFixed(2) + "," + mapY(opts.baseline[idx]).toFixed(2);
      }).join(" ");
      svg.appendChild(svgEl("polyline", {
        points: pts,
        fill: "none",
        stroke: "#fb7185",
        "stroke-width": "2.2",
        "stroke-dasharray": "6,4"
      }));
    }

    // Series polylines + optional p* markers + MC error bars
    (opts.series || []).forEach(function (ser) {
      if (!Array.isArray(ser.values) || ser.values.length !== pGrid.length) return;
      const pts = pGrid.map(function (px, idx) {
        return mapX(px).toFixed(2) + "," + mapY(ser.values[idx]).toFixed(2);
      }).join(" ");

      svg.appendChild(svgEl("polyline", {
        points: pts,
        fill: "none",
        stroke: ser.color,
        "stroke-width": ser.highlighted ? "3.2" : "2",
        "stroke-opacity": ser.highlighted ? "1" : "0.88"
      }));

      if (ser.pStar && ser.pStar > xMin && ser.pStar < xMax) {
        const sx = mapX(ser.pStar);
        svg.appendChild(svgEl("line", {
          x1: sx,
          y1: padT,
          x2: sx,
          y2: h - padB,
          stroke: ser.color,
          "stroke-width": "1.2",
          "stroke-dasharray": "3,3",
          "stroke-opacity": "0.7"
        }));
      }

      if (Array.isArray(ser.mcPoints)) {
        ser.mcPoints.forEach(function (pt) {
          if (pt.p < xMin || pt.p > xMax) return;
          const cx = mapX(pt.p);
          const cy = mapY(pt.y);
          const err = Number(pt.se) || 0;
          if (err > 0) {
            const yTop = mapY(pt.y + 1.96 * err);
            const yBot = mapY(pt.y - 1.96 * err);
            svg.appendChild(svgEl("line", {
              x1: cx,
              y1: yTop,
              x2: cx,
              y2: yBot,
              stroke: ser.color,
              "stroke-width": "1.2"
            }));
          }
          svg.appendChild(svgEl("circle", {
            cx: cx,
            cy: cy,
            r: ser.highlighted ? "3.5" : "2.6",
            fill: ser.color,
            stroke: "#0b1220",
            "stroke-width": "0.9"
          }));
        });
      }
    });

    // Axis titles
    svg.appendChild(svgEl("text", {
      x: padL + plotW / 2,
      y: h - 10,
      "text-anchor": "middle",
      fill: "#e2e8f0",
      "font-size": "11.5",
      "font-weight": "600"
    }, opts.xLabel || "p"));

    svg.appendChild(svgEl("text", {
      x: 16,
      y: padT + plotH / 2,
      "text-anchor": "middle",
      transform: "rotate(-90 16 " + (padT + plotH / 2) + ")",
      fill: "#e2e8f0",
      "font-size": "11.5",
      "font-weight": "600"
    }, opts.yLabel || ""));

    return svg;
  }

  function clearChartsForLoading(curvesEl, deltaEl, cohEl) {
    [curvesEl, deltaEl, cohEl].forEach(function (el) {
      if (el) {
        el.textContent = "";
      }
    });
  }

  function renderSweepsCharts(els, sweepsData, activeCode, activeChannel) {
    const t = global.QECC_I18N ? global.QECC_I18N.t : function (k) { return k; };
    if (!sweepsData) return;

    const metric = sweepsData.metric || "entanglement";
    const pGrid = sweepsData.p_grid || [];
    const rawCurve = sweepsData.raw_curve || [];
    const analytical = sweepsData.analytical_curves || {};
    const mcSweeps = sweepsData.mc_sweeps || {};
    const cohCurves = sweepsData.coherence_curves || {};
    const thresholds = sweepsData.gain_thresholds || {};

    const keys = pickDisplayKeys(analytical, activeCode, activeChannel);
    const activeKey = activeCode + "/" + activeChannel;

    // 1. Main curves-chart
    if (els.curvesEl) {
      els.curvesEl.textContent = "";
      const series = keys.map(function (k, idx) {
        const entry = analytical[k];
        const mcRaw = mcSweeps[k] || [];
        const mcPts = mcRaw.map(function (row) {
          const yVal = row.converted_fidelity !== undefined ? Number(row.converted_fidelity) : Number(row.mean_fidelity);
          return { p: Number(row.p), y: yVal, se: Number(row.std_error) || 0 };
        });
        return {
          key: k,
          values: entry.fidelities,
          pStar: entry.p_star,
          mcPoints: mcPts,
          color: getSeriesColor(k, idx),
          highlighted: k === activeKey
        };
      });

      const allVals = rawCurve.slice();
      series.forEach(function (s) {
        allVals.push.apply(allVals, s.values);
      });
      const minV = Math.max(0, Math.min.apply(null, allVals) - 0.04);
      const yMin = Math.floor(minV * 10) / 10;

      const svg = buildLineSvg({
        width: 780,
        height: 320,
        pGrid: pGrid,
        baseline: rawCurve,
        series: series,
        yMin: yMin,
        yMax: 1.02,
        xLabel: t("curves.axis.p"),
        yLabel: metric === "state" ? t("curves.axis.fidelity_state") : t("curves.axis.fidelity_ent"),
        ariaLabel: t("curves.title")
      });
      els.curvesEl.appendChild(svg);

      // Legend below main SVG
      const legend = document.createElement("div");
      legend.className = "bloch-legend";
      const baseItem = document.createElement("div");
      baseItem.className = "legend-item";
      const baseDot = document.createElement("span");
      baseDot.className = "legend-dot";
      baseDot.style.backgroundColor = "#fb7185";
      const baseTxt = document.createElement("span");
      baseTxt.textContent = metric === "state" ? t("curves.raw_baseline_state") : t("curves.raw_baseline_ent");
      baseItem.appendChild(baseDot);
      baseItem.appendChild(baseTxt);
      legend.appendChild(baseItem);

      series.forEach(function (s) {
        const it = document.createElement("div");
        it.className = "legend-item";
        const dot = document.createElement("span");
        dot.className = "legend-dot";
        dot.style.backgroundColor = s.color;
        const txt = document.createElement("span");
        txt.className = "mono";
        const pStarTxt = s.pStar > 0 ? " (p*=" + Number(s.pStar).toFixed(4) + ")" : "";
        txt.textContent = s.key + pStarTxt;
        it.appendChild(dot);
        it.appendChild(txt);
        legend.appendChild(it);
      });
      els.curvesEl.appendChild(legend);
    }

    // 2. Delta F chart
    if (els.deltaEl) {
      els.deltaEl.textContent = "";
      const deltaSeries = keys.map(function (k, idx) {
        return {
          key: k,
          values: analytical[k].delta_f,
          pStar: analytical[k].p_star,
          color: getSeriesColor(k, idx),
          highlighted: k === activeKey
        };
      });
      let minD = -0.1;
      let maxD = 0.25;
      deltaSeries.forEach(function (s) {
        s.values.forEach(function (v) {
          if (v < minD) minD = v;
          if (v > maxD) maxD = v;
        });
      });
      const svgDelta = buildLineSvg({
        width: 520,
        height: 260,
        pGrid: pGrid,
        series: deltaSeries,
        yMin: Math.max(-1.05, Math.floor(minD * 20) / 20 - 0.02),
        yMax: Math.min(1.05, Math.ceil(maxD * 20) / 20 + 0.02),
        xLabel: t("curves.axis.p"),
        yLabel: t("curves.axis.delta_f"),
        ariaLabel: t("curves.delta_chart.title")
      });
      els.deltaEl.appendChild(svgDelta);
    }

    // 3. Coherence |rho_01|(p) chart
    if (els.cohEl) {
      els.cohEl.textContent = "";
      const cohKeys = keys.filter(function (k) { return !!cohCurves[k]; });
      const firstCoh = cohKeys.length ? cohCurves[cohKeys[0]] : null;
      const cohPGrid = firstCoh ? firstCoh.p_grid : pGrid;
      const cohBaseline = firstCoh ? firstCoh.abs_rho01_raw : [];
      const cohSeries = cohKeys.map(function (k, idx) {
        return {
          key: k,
          values: cohCurves[k].abs_rho01_analytical,
          color: getSeriesColor(k, idx),
          highlighted: k === activeKey
        };
      });
      const svgCoh = buildLineSvg({
        width: 520,
        height: 260,
        pGrid: cohPGrid,
        baseline: cohBaseline,
        series: cohSeries,
        yMin: 0.0,
        yMax: 0.52,
        xLabel: t("curves.axis.p"),
        yLabel: t("curves.axis.abs_rho01"),
        ariaLabel: t("curves.coh_chart.title")
      });
      els.cohEl.appendChild(svgCoh);
    }

    // 4. Gain threshold table
    if (els.pstarTable) {
      els.pstarTable.textContent = "";
      const thead = document.createElement("thead");
      const hr = document.createElement("tr");
      [
        t("curves.p_star.col_code"),
        t("curves.p_star.col_channel"),
        t("curves.p_star.col_p_star"),
        t("curves.p_star.col_region")
      ].forEach(function (lbl) {
        const th = document.createElement("th");
        th.textContent = lbl;
        hr.appendChild(th);
      });
      thead.appendChild(hr);
      els.pstarTable.appendChild(thead);

      const tbody = document.createElement("tbody");
      Object.keys(thresholds).forEach(function (pairKey) {
        const parts = pairKey.split("/");
        const cName = parts[0] || "";
        const chName = parts[1] || "";
        const val = Number(thresholds[pairKey]) || 0;
        const tr = document.createElement("tr");
        if (pairKey === activeKey) {
          tr.style.outline = "2px solid #38bdf8";
        }
        const tdC = document.createElement("td");
        tdC.className = "mono";
        tdC.textContent = cName;
        const tdCh = document.createElement("td");
        tdCh.className = "mono";
        tdCh.textContent = chName;
        const tdP = document.createElement("td");
        tdP.className = "mono";
        tdP.textContent = val > 0 ? val.toFixed(10) : t("curves.p_star.none");
        const tdR = document.createElement("td");
        tdR.className = "mono";
        tdR.textContent = val > 0 ? "0 < p < " + val.toFixed(4) : "—";
        tr.appendChild(tdC);
        tr.appendChild(tdCh);
        tr.appendChild(tdP);
        tr.appendChild(tdR);
        tbody.appendChild(tr);
      });
      els.pstarTable.appendChild(tbody);
    }
  }

  global.QECC_CHARTS = {
    clearChartsForLoading: clearChartsForLoading,
    renderSweepsCharts: renderSweepsCharts
  };
})(window);
