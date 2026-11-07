/* ==========================================================================
   FieldPlot — a small, dependency-free canvas field plotter.

   Renders geophysical grids (heat map + contour lines) and point sets,
   with latitude/longitude axes, a vertical colour bar and hover readout.
   No external libraries, no tiles, no network access.
   ========================================================================== */
(function (global) {
  'use strict';

  /* Two palettes matching the CSS themes. */
  const THEMES = {
    light: {
      paper: '#FFFFFF',
      ink: '#3B3B3B',
      ink2: '#616161',
      ink3: '#8B8B8B',
      line: '#CECECE',
      gridLine: 'rgba(20,20,20,0.06)',
      contour: 'rgba(20,20,20,0.34)',
      crosshair: 'rgba(20,20,20,0.35)',
      markerRing: '#FFFFFF',
      pointStroke: 'rgba(255,255,255,0.9)',
    },
    dark: {
      paper: '#1F1F1F',
      ink: '#CCCCCC',
      ink2: '#B0B0B0',
      ink3: '#8B8B8B',
      line: '#3C3C3C',
      gridLine: 'rgba(210,210,210,0.07)',
      contour: 'rgba(210,210,210,0.30)',
      crosshair: 'rgba(210,210,210,0.35)',
      markerRing: '#1F1F1F',
      pointStroke: 'rgba(255,255,255,0.85)',
    },
  };
  let C = THEMES.light;
  let themeName = 'light';

  /* diverging (blue → white → red) and sequential (light → deep blue) ramps */
  const DIVERGING = [
    [0.00, 33, 55, 100],
    [0.25, 84, 146, 204],
    [0.50, 246, 248, 251],
    [0.75, 213, 106, 88],
    [1.00, 128, 26, 24],
  ];
  const SEQUENTIAL = [
    [0.00, 240, 246, 252],
    [0.25, 176, 208, 235],
    [0.50, 98, 160, 208],
    [0.75, 38, 106, 168],
    [1.00, 12, 52, 100],
  ];

  function rampColor(stops, t) {
    const x = Math.max(0, Math.min(1, t));
    for (let i = 0; i < stops.length - 1; i++) {
      const a = stops[i], b = stops[i + 1];
      if (x <= b[0]) {
        const f = (x - a[0]) / (b[0] - a[0] || 1);
        return [
          Math.round(a[1] + f * (b[1] - a[1])),
          Math.round(a[2] + f * (b[2] - a[2])),
          Math.round(a[3] + f * (b[3] - a[3])),
        ];
      }
    }
    const last = stops[stops.length - 1];
    return [last[1], last[2], last[3]];
  }

  function niceTicks(min, max, target) {
    if (!(isFinite(min) && isFinite(max)) || min === max) return [min];
    const span = max - min;
    const raw = span / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const out = [];
    const start = Math.ceil(min / step) * step;
    for (let v = start; v <= max + step * 1e-6; v += step) {
      out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
    }
    return out;
  }

  function formatValue(v, span) {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(2);
    const digits = span >= 100 ? 0 : span >= 10 ? 1 : span >= 1 ? 2 : 3;
    return v.toFixed(digits);
  }

  class FieldPlot {
    constructor(canvas, tooltipEl) {
      this.canvas = canvas;
      this.tooltip = tooltipEl;
      this.data = null;
      this.hover = null;
      this.pad = { l: 58, r: 88, t: 30, b: 42 };
      this._bind();
      if (global.ResizeObserver && canvas.parentElement) {
        this._ro = new ResizeObserver(() => this.draw());
        this._ro.observe(canvas.parentElement);
      } else {
        global.addEventListener('resize', () => this.draw());
      }
    }

    _bind() {
      this.canvas.addEventListener('mousemove', (e) => {
        if (!this.data) return;
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const hit = this._pick(x, y);
        this.hover = hit;
        if (hit && this.tooltip) {
          this.tooltip.style.display = 'block';
          this.tooltip.textContent = hit.label;
          const tw = this.tooltip.offsetWidth || 120;
          const th = this.tooltip.offsetHeight || 30;
          let tx = x + 16, ty = y - th - 10;
          if (tx + tw > this.canvas.clientWidth - 8) tx = x - tw - 14;
          if (ty < 4) ty = y + 16;
          this.tooltip.style.left = tx + 'px';
          this.tooltip.style.top = ty + 'px';
        } else if (this.tooltip) {
          this.tooltip.style.display = 'none';
        }
        this.draw();
      });
      this.canvas.addEventListener('mouseleave', () => {
        this.hover = null;
        if (this.tooltip) this.tooltip.style.display = 'none';
        this.draw();
      });
    }

    clear() {
      this.data = null;
      this.draw();
    }

    /** Re-measure and repaint (after the canvas was hidden or the theme changed). */
    resize() {
      this.draw();
    }

    /**
     * @param {object} d
     *   kind    'grid' | 'points'
     *   lon     number[]
     *   lat     number[]
     *   values  2-D array (grid) or 1-D array (points)
     *   title   string
     *   unit    string
     */
    setData(d) {
      const nums = [];
      if (d.kind === 'grid') {
        for (const row of d.values) {
          for (const v of row) if (Number.isFinite(v)) nums.push(v);
        }
      } else {
        for (const v of d.values) if (Number.isFinite(v)) nums.push(v);
      }
      if (!nums.length) { this.data = null; this.draw(); return; }

      let min = Infinity, max = -Infinity, sum = 0;
      for (const v of nums) {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
      }
      const diverging = min < 0 && max > 0;
      this.data = {
        ...d,
        min,
        max,
        mean: sum / nums.length,
        span: max - min || 1,
        diverging,
        mid: diverging ? 0 : null,
      };
      this.draw();
    }

    _ramp(d) { return d.diverging ? DIVERGING : SEQUENTIAL; }

    _t(d, v) {
      if (d.diverging) {
        const m = 0;
        return v < m ? 0.5 * (v - d.min) / (m - d.min || 1)
                     : 0.5 + 0.5 * (v - m) / (d.max - m || 1);
      }
      return (v - d.min) / d.span;
    }

    draw() {
      const cv = this.canvas;
      const W = Math.max(120, cv.clientWidth);
      const H = Math.max(120, cv.clientHeight);
      const dpr = global.devicePixelRatio || 1;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
        cv.width = Math.round(W * dpr);
        cv.height = Math.round(H * dpr);
      }
      const ctx = cv.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = C.paper;
      ctx.fillRect(0, 0, W, H);

      const d = this.data;
      if (!d) return;

      const pad = this.pad;
      const pw = W - pad.l - pad.r;
      const ph = H - pad.t - pad.b;
      if (pw < 40 || ph < 40) return;

      const lonMin = Math.min(...d.lon), lonMax = Math.max(...d.lon);
      const latMin = Math.min(...d.lat), latMax = Math.max(...d.lat);
      const lonSpan = lonMax - lonMin || 1;
      const latSpan = latMax - latMin || 1;

      const xOf = (lon) => pad.l + ((lon - lonMin) / lonSpan) * pw;
      const yOf = (lat) => pad.t + (1 - (lat - latMin) / latSpan) * ph;

      ctx.save();
      ctx.beginPath();
      ctx.rect(pad.l, pad.t, pw, ph);
      ctx.clip();

      if (d.kind === 'grid') {
        this._drawGrid(ctx, d, pad, pw, ph);
        if (d.values.length <= 420 && d.values[0].length <= 420) {
          this._drawContours(ctx, d, pad, pw, ph, xOf, yOf);
        }
      } else {
        this._drawPoints(ctx, d, pad, pw, ph, xOf, yOf);
      }

      // hover crosshair
      if (this.hover && this.hover.px !== undefined) {
        ctx.strokeStyle = C.crosshair;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(this.hover.px, pad.t);
        ctx.lineTo(this.hover.px, pad.t + ph);
        ctx.moveTo(pad.l, this.hover.py);
        ctx.lineTo(pad.l + pw, this.hover.py);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(this.hover.px, this.hover.py, 3.2, 0, Math.PI * 2);
        ctx.fillStyle = C.markerRing;
        ctx.fill();
        ctx.strokeStyle = C.ink;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }

      ctx.restore();

      this._drawAxes(ctx, d, pad, pw, ph, lonMin, lonMax, latMin, latMax);
      this._drawColorbar(ctx, d, W, H, pad, ph);
      this._drawTitle(ctx, d, pad, pw);
    }

    _drawGrid(ctx, d, pad, pw, ph) {
      const rows = d.values.length;
      const cols = d.values[0].length;
      const off = document.createElement('canvas');
      off.width = cols;
      off.height = rows;
      const octx = off.getContext('2d');
      const img = octx.createImageData(cols, rows);
      const stops = this._ramp(d);

      for (let j = 0; j < rows; j++) {
        const row = d.values[j];
        for (let i = 0; i < cols; i++) {
          const k = (j * cols + i) * 4;
          const v = row[i];
          if (!Number.isFinite(v)) {
            img.data[k] = 255; img.data[k + 1] = 255; img.data[k + 2] = 255; img.data[k + 3] = 0;
            continue;
          }
          const [r, g, b] = rampColor(stops, this._t(d, v));
          img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(off, pad.l, pad.t, pw, ph);
    }

    _drawContours(ctx, d, pad, pw, ph, xOf, yOf) {
      const values = d.values;
      const rows = values.length;
      const cols = values[0].length;
      const levels = niceTicks(d.min, d.max, 7).filter((v) => v > d.min && v < d.max);
      if (!levels.length) return;

      ctx.strokeStyle = C.contour;
      ctx.lineWidth = 0.7;
      ctx.beginPath();

      for (const level of levels) {
        for (let j = 0; j < rows - 1; j++) {
          for (let i = 0; i < cols - 1; i++) {
            const v0 = values[j][i];
            const v1 = values[j][i + 1];
            const v2 = values[j + 1][i + 1];
            const v3 = values[j + 1][i];
            if (![v0, v1, v2, v3].every(Number.isFinite)) continue;

            const x0 = xOf(d.lon[i]);
            const x1 = xOf(d.lon[i + 1]);
            const y0 = yOf(d.lat[j]);
            const y1 = yOf(d.lat[j + 1]);

            const pts = [];
            const edge = (ax, ay, av, bx, by, bv) => {
              const t = (level - av) / (bv - av || 1e-12);
              if (t >= 0 && t <= 1) pts.push([ax + (bx - ax) * t, ay + (by - ay) * t]);
            };
            if ((v0 > level) !== (v1 > level)) edge(x0, y0, v0, x1, y0, v1);
            if ((v1 > level) !== (v2 > level)) edge(x1, y0, v1, x1, y1, v2);
            if ((v2 > level) !== (v3 > level)) edge(x1, y1, v2, x0, y1, v3);
            if ((v3 > level) !== (v0 > level)) edge(x0, y1, v3, x0, y0, v0);

            if (pts.length === 2) {
              ctx.moveTo(pts[0][0], pts[0][1]);
              ctx.lineTo(pts[1][0], pts[1][1]);
            } else if (pts.length === 4) {
              ctx.moveTo(pts[0][0], pts[0][1]);
              ctx.lineTo(pts[1][0], pts[1][1]);
              ctx.moveTo(pts[2][0], pts[2][1]);
              ctx.lineTo(pts[3][0], pts[3][1]);
            }
          }
        }
      }
      ctx.stroke();
    }

    _drawPoints(ctx, d, pad, pw, ph, xOf, yOf) {
      const stops = this._ramp(d);
      const n = d.values.length;
      const radius = n > 400 ? 2.4 : n > 120 ? 3.4 : n > 30 ? 4.6 : 6;
      const labels = n <= 30;

      for (let i = 0; i < n; i++) {
        const v = d.values[i];
        if (!Number.isFinite(v)) continue;
        const x = xOf(d.lon[i]);
        const y = yOf(d.lat[i]);
        const [r, g, b] = rampColor(stops, this._t(d, v));
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fill();
        ctx.strokeStyle = C.pointStroke;
        ctx.lineWidth = 1.2;
        ctx.stroke();
        if (labels) {
          ctx.font = '10px ui-monospace, Consolas, monospace';
          ctx.fillStyle = C.ink2;
          ctx.textAlign = 'center';
          ctx.fillText(`P${i + 1}`, x, y - radius - 4);
        }
      }
    }

    _drawAxes(ctx, d, pad, pw, ph, lonMin, lonMax, latMin, latMax) {
      ctx.font = '10px ui-monospace, Consolas, monospace';
      ctx.fillStyle = C.ink3;
      ctx.strokeStyle = C.gridLine;
      ctx.lineWidth = 1;

      const lonTicks = niceTicks(lonMin, lonMax, 7);
      const latTicks = niceTicks(latMin, latMax, 6);
      const lonSpan = lonMax - lonMin || 1;
      const latSpan = latMax - latMin || 1;

      ctx.textAlign = 'center';
      for (const t of lonTicks) {
        const x = pad.l + ((t - lonMin) / lonSpan) * pw;
        if (x < pad.l - 1 || x > pad.l + pw + 1) continue;
        ctx.beginPath();
        ctx.moveTo(x, pad.t);
        ctx.lineTo(x, pad.t + ph);
        ctx.stroke();
        ctx.fillText(`${formatValue(t, lonSpan)}°`, x, pad.t + ph + 15);
      }

      ctx.textAlign = 'right';
      for (const t of latTicks) {
        const y = pad.t + (1 - (t - latMin) / latSpan) * ph;
        if (y < pad.t - 1 || y > pad.t + ph + 1) continue;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + pw, y);
        ctx.stroke();
        ctx.fillText(`${formatValue(t, latSpan)}°`, pad.l - 8, y + 3.5);
      }

      ctx.strokeStyle = C.line;
      ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, pw - 1, ph - 1);

      ctx.textAlign = 'left';
      ctx.fillStyle = C.ink3;
      ctx.fillText('LONGITUDE', pad.l, pad.t + ph + 30);
      ctx.save();
      ctx.translate(pad.l - 40, pad.t + ph);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('LATITUDE', 0, 0);
      ctx.restore();
    }

    _drawColorbar(ctx, d, W, H, pad, ph) {
      const x = W - pad.r + 26;
      const y = pad.t;
      const w = 12;
      const h = ph;
      const grad = ctx.createLinearGradient(0, y + h, 0, y);

      if (d.diverging) {
        grad.addColorStop(0, 'rgb(33,55,100)');
        grad.addColorStop(0.25, 'rgb(84,146,204)');
        grad.addColorStop(0.5, 'rgb(246,248,251)');
        grad.addColorStop(0.75, 'rgb(213,106,88)');
        grad.addColorStop(1, 'rgb(128,26,24)');
      } else {
        grad.addColorStop(0, 'rgb(240,246,252)');
        grad.addColorStop(0.5, 'rgb(98,160,208)');
        grad.addColorStop(1, 'rgb(12,52,100)');
      }

      ctx.fillStyle = grad;
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = C.line;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

      const ticks = niceTicks(d.min, d.max, 6);
      ctx.font = '10px ui-monospace, Consolas, monospace';
      ctx.fillStyle = C.ink2;
      ctx.textAlign = 'left';
      for (const t of ticks) {
        if (t < d.min - d.span * 0.001 || t > d.max + d.span * 0.001) continue;
        const frac = d.diverging
          ? (t < 0 ? 0.5 * (t - d.min) / (0 - d.min || 1)
                   : 0.5 + 0.5 * t / (d.max || 1))
          : (t - d.min) / d.span;
        const yt = y + (1 - frac) * h;
        ctx.beginPath();
        ctx.moveTo(x + w, yt);
        ctx.lineTo(x + w + 3, yt);
        ctx.strokeStyle = C.ink3;
        ctx.stroke();
        ctx.fillText(formatValue(t, d.span), x + w + 5, yt + 3.5);
      }
    }

    _drawTitle(ctx, d, pad, pw) {
      if (!d.title) return;
      ctx.font = '500 12px -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif';
      ctx.fillStyle = C.ink;
      ctx.textAlign = 'left';
      ctx.fillText(d.title, pad.l, pad.t - 10);
      if (d.unit) {
        const tw = ctx.measureText(d.title).width;
        ctx.font = '10px ui-monospace, Consolas, monospace';
        ctx.fillStyle = C.ink3;
        ctx.fillText(`[${d.unit}]`, pad.l + tw + 7, pad.t - 10);
      }
    }

    _pick(x, y) {
      const d = this.data;
      const pad = this.pad;
      const W = this.canvas.clientWidth;
      const H = this.canvas.clientHeight;
      const pw = W - pad.l - pad.r;
      const ph = H - pad.t - pad.b;
      if (x < pad.l || x > pad.l + pw || y < pad.t || y > pad.t + ph) return null;

      const lonMin = Math.min(...d.lon), lonMax = Math.max(...d.lon);
      const latMin = Math.min(...d.lat), latMax = Math.max(...d.lat);
      const lon = lonMin + ((x - pad.l) / pw) * (lonMax - lonMin);
      const lat = latMax - ((y - pad.t) / ph) * (latMax - latMin);

      let value = null;
      if (d.kind === 'grid') {
        const rows = d.values.length;
        const cols = d.values[0].length;
        const ci = Math.round(((lon - lonMin) / (lonMax - lonMin || 1)) * (cols - 1));
        const ri = Math.round(((latMax - lat) / (latMax - latMin || 1)) * (rows - 1));
        if (ri >= 0 && ri < rows && ci >= 0 && ci < cols) value = d.values[ri][ci];
      } else {
        let best = Infinity;
        for (let i = 0; i < d.lon.length; i++) {
          const dx = d.lon[i] - lon;
          const dy = d.lat[i] - lat;
          const dist = dx * dx + dy * dy;
          if (dist < best) { best = dist; value = d.values[i]; }
        }
      }

      const label = `${lat.toFixed(4)}°N  ${lon.toFixed(4)}°E\n` +
        (Number.isFinite(value) ? `value  ${formatValue(value, d.span)}${d.unit ? ' ' + d.unit : ''}` : 'value  —');
      return { px: x, py: y, label, lon, lat, value };
    }
  }

  global.FieldPlot = FieldPlot;
  global.FieldPlotUtil = {
    niceTicks, formatValue, rampColor, DIVERGING, SEQUENTIAL,
    setTheme(name) {
      themeName = name === 'dark' ? 'dark' : 'light';
      C = THEMES[themeName];
    },
    getTheme() { return themeName; },
  };
})(window);
