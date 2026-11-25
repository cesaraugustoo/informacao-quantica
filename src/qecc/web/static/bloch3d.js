(function (global) {
  "use strict";

  const DEFAULT_YAW = -0.62;
  const DEFAULT_PITCH = 0.36;

  function BlochRenderer(canvas, legendEl) {
    this.canvas = canvas;
    this.legendEl = legendEl;
    this.yaw = DEFAULT_YAW;
    this.pitch = DEFAULT_PITCH;
    this.dragging = false;
    this.lastX = 0;
    this.lastY = 0;
    this.data = null;
    this._bindEvents();
    this.render();
  }

  BlochRenderer.prototype._bindEvents = function () {
    if (!this.canvas) return;
    const self = this;

    this.canvas.addEventListener("pointerdown", function (e) {
      self.dragging = true;
      self.lastX = e.clientX;
      self.lastY = e.clientY;
      if (self.canvas.setPointerCapture) {
        try {
          self.canvas.setPointerCapture(e.pointerId);
        } catch (_err) {
          // Ignore in synthetic environments
        }
      }
    });

    this.canvas.addEventListener("pointermove", function (e) {
      if (!self.dragging) return;
      const dx = e.clientX - self.lastX;
      const dy = e.clientY - self.lastY;
      self.lastX = e.clientX;
      self.lastY = e.clientY;
      self.yaw += dx * 0.012;
      self.pitch = Math.max(-1.45, Math.min(1.45, self.pitch + dy * 0.012));
      self.render();
    });

    const endDrag = function () {
      self.dragging = false;
    };
    this.canvas.addEventListener("pointerup", endDrag);
    this.canvas.addEventListener("pointercancel", endDrag);
    this.canvas.addEventListener("dblclick", function () {
      self.resetView();
    });
  };

  BlochRenderer.prototype.resetView = function () {
    this.yaw = DEFAULT_YAW;
    this.pitch = DEFAULT_PITCH;
    this.render();
  };

  BlochRenderer.prototype.setData = function (data) {
    this.data = data || null;
    this.render();
  };

  BlochRenderer.prototype.clear = function () {
    this.data = null;
    this.render();
  };

  BlochRenderer.prototype._project = function (vx, vy, vz, cx, cy, radius) {
    // Standard quantum optics orientation: +Z up, +X forward-left, +Y right
    // Map Bloch (x, y, z) -> camera coords (X_cam, Y_cam, Z_cam)
    // Rotate around vertical Z-axis by yaw, then tilt by pitch
    const cosY = Math.cos(this.yaw);
    const sinY = Math.sin(this.yaw);
    const x1 = vx * cosY - vy * sinY;
    const y1 = vx * sinY + vy * cosY;
    const z1 = vz;

    const cosP = Math.cos(this.pitch);
    const sinP = Math.sin(this.pitch);
    const depth = y1 * cosP - z1 * sinP;
    const up = y1 * sinP + z1 * cosP;

    const persp = 1 / (1 - depth * 0.16);
    return {
      px: cx + x1 * radius * persp,
      py: cy - up * radius * persp,
      depth: depth
    };
  };

  BlochRenderer.prototype._drawCircle3D = function (ctx, cx, cy, radius, fnPoint, strokeStyle, dashed) {
    ctx.save();
    ctx.beginPath();
    if (dashed) {
      ctx.setLineDash([4, 4]);
    }
    const steps = 72;
    for (let i = 0; i <= steps; i++) {
      const a = (i / steps) * Math.PI * 2;
      const p3 = fnPoint(a);
      const p2 = this._project(p3[0], p3[1], p3[2], cx, cy, radius);
      if (i === 0) {
        ctx.moveTo(p2.px, p2.py);
      } else {
        ctx.lineTo(p2.px, p2.py);
      }
    }
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 1.1;
    ctx.stroke();
    ctx.restore();
  };

  BlochRenderer.prototype._drawArrow = function (ctx, cx, cy, radius, vec, color, label, dashed) {
    if (!Array.isArray(vec) || vec.length < 3) return;
    const vx = Number(vec[0]) || 0;
    const vy = Number(vec[1]) || 0;
    const vz = Number(vec[2]) || 0;

    const origin = this._project(0, 0, 0, cx, cy, radius);
    const tip = this._project(vx, vy, vz, cx, cy, radius);

    const dx = tip.px - origin.px;
    const dy = tip.py - origin.py;
    const len = Math.hypot(dx, dy);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = dashed ? 2 : 2.8;
    if (dashed) {
      ctx.setLineDash([5, 4]);
    }

    ctx.beginPath();
    ctx.moveTo(origin.px, origin.py);
    ctx.lineTo(tip.px, tip.py);
    ctx.stroke();

    if (len > 8) {
      const ang = Math.atan2(dy, dx);
      const head = 8;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(tip.px, tip.py);
      ctx.lineTo(tip.px - head * Math.cos(ang - 0.4), tip.py - head * Math.sin(ang - 0.4));
      ctx.lineTo(tip.px - head * Math.cos(ang + 0.4), tip.py - head * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();
    }

    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(tip.px, tip.py, 4.5, 0, Math.PI * 2);
    ctx.fill();

    if (label) {
      ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = color;
      ctx.fillText(label, tip.px + 7, tip.py - 6);
    }
    ctx.restore();
  };

  BlochRenderer.prototype.render = function () {
    if (!this.canvas || !this.canvas.getContext) return;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;

    const w = this.canvas.width > 0 ? this.canvas.width : 440;
    const h = this.canvas.height > 0 ? this.canvas.height : 360;
    ctx.clearRect(0, 0, w, h);

    // Paint sphere background
    const bgGrad = ctx.createRadialGradient(w * 0.45, h * 0.4, 20, w * 0.5, h * 0.5, Math.max(w, h) * 0.65);
    bgGrad.addColorStop(0, "#192946");
    bgGrad.addColorStop(1, "#0b1220");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.52;
    const radius = Math.min(w, h) * 0.35;

    // Outer sphere silhouette
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(56, 189, 248, 0.05)";
    ctx.fill();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.restore();

    // Wireframe circles: equator (XY), XZ, YZ
    this._drawCircle3D(ctx, cx, cy, radius, function (a) { return [Math.cos(a), Math.sin(a), 0]; }, "rgba(148, 163, 184, 0.35)", false);
    this._drawCircle3D(ctx, cx, cy, radius, function (a) { return [Math.cos(a), 0, Math.sin(a)]; }, "rgba(148, 163, 184, 0.2)", true);
    this._drawCircle3D(ctx, cx, cy, radius, function (a) { return [0, Math.cos(a), Math.sin(a)]; }, "rgba(148, 163, 184, 0.2)", true);

    // Axes lines & cardinal pole labels
    const self = this;
    const axes = [
      { a: [-1.15, 0, 0], b: [1.15, 0, 0], la: "|-⟩", lb: "|+⟩ (X)" },
      { a: [0, -1.15, 0], b: [0, 1.15, 0], la: "|-i⟩", lb: "|+i⟩ (Y)" },
      { a: [0, 0, -1.15], b: [0, 0, 1.15], la: "|1⟩", lb: "|0⟩ (Z)" }
    ];
    ctx.save();
    axes.forEach(function (ax) {
      const pA = self._project(ax.a[0], ax.a[1], ax.a[2], cx, cy, radius);
      const pB = self._project(ax.b[0], ax.b[1], ax.b[2], cx, cy, radius);
      ctx.beginPath();
      ctx.moveTo(pA.px, pA.py);
      ctx.lineTo(pB.px, pB.py);
      ctx.strokeStyle = "rgba(148, 163, 184, 0.32)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.font = "600 11px system-ui, -apple-system, sans-serif";
      ctx.fillStyle = "#cbd5e1";
      ctx.fillText(ax.lb, pB.px + 4, pB.py + 3);
      ctx.fillStyle = "#94a3b8";
      ctx.fillText(ax.la, pA.px + 4, pA.py + 3);
    });
    ctx.restore();

    if (this.data) {
      const d = this.data;
      if (d.input) {
        this._drawArrow(ctx, cx, cy, radius, d.input, "#fbbf24", "|ψ⟩", true);
      }
      if (d.raw) {
        this._drawArrow(ctx, cx, cy, radius, d.raw, "#fb7185", "raw", false);
      }
      if (d.dfs) {
        this._drawArrow(ctx, cx, cy, radius, d.dfs, "#34d399", "dfs_2", false);
      }
      if (d.qecc) {
        this._drawArrow(ctx, cx, cy, radius, d.qecc, "#38bdf8", d.codeLabel || "QECC", false);
      }
    }

    this._renderLegend();
  };

  BlochRenderer.prototype._renderLegend = function () {
    if (!this.legendEl) return;
    const t = global.QECC_I18N ? global.QECC_I18N.t : function (k) { return k; };
    const d = this.data || {};
    const fmt = function (v) {
      if (!Array.isArray(v) || v.length < 3) return "—";
      const r = Math.hypot(v[0], v[1], v[2]);
      return "(" + Number(v[0]).toFixed(2) + ", " + Number(v[1]).toFixed(2) + ", " + Number(v[2]).toFixed(2) + ") |r|=" + r.toFixed(3);
    };
    const items = [
      { color: "#fbbf24", label: t("bloch.legend.input"), val: fmt(d.input) },
      { color: "#fb7185", label: t("bloch.legend.raw"), val: fmt(d.raw) },
      { color: "#38bdf8", label: t("bloch.legend.qecc", { code: d.code || "qecc" }) + (d.stageKey ? " [" + d.stageKey + "]" : ""), val: fmt(d.qecc) },
      { color: "#34d399", label: t("bloch.legend.dfs"), val: fmt(d.dfs) }
    ];

    this.legendEl.textContent = "";
    items.forEach(function (it) {
      const row = document.createElement("div");
      row.className = "legend-item";
      const dot = document.createElement("span");
      dot.className = "legend-dot";
      dot.style.backgroundColor = it.color;
      const txt = document.createElement("div");
      const title = document.createElement("div");
      title.textContent = it.label;
      title.style.fontWeight = "600";
      const coords = document.createElement("div");
      coords.className = "mono";
      coords.style.fontSize = "0.71rem";
      coords.style.color = "#9fb0cb";
      coords.textContent = it.val;
      txt.appendChild(title);
      txt.appendChild(coords);
      row.appendChild(dot);
      row.appendChild(txt);
      this.legendEl.appendChild(row);
    }, this);
  };

  global.QECC_BLOCH = {
    BlochRenderer: BlochRenderer
  };
})(window);
