import { useState, useEffect, useRef } from "react";
import {
  parseCSV,
  getValidRows,
  trainLinearRegression,
  predictPrice,
  FEATURE_KEYS,
} from "./regression";
import "./App.css";

const FEATURES = [
  { key: "Engine_size",    label: "Engine Size",    unit: "L",    step: 0.1, decimals: 1 },
  { key: "Horsepower",     label: "Horsepower",     unit: "HP",   step: 1,   decimals: 0 },
  { key: "Wheelbase",      label: "Wheelbase",      unit: "in",   step: 0.1, decimals: 1 },
  { key: "Width",          label: "Width",          unit: "in",   step: 0.1, decimals: 1 },
  { key: "Length",         label: "Length",         unit: "in",   step: 0.1, decimals: 1 },
  { key: "Curb_weight",    label: "Curb Weight",    unit: "tons", step: 0.01,decimals: 2 },
  { key: "Fuel_capacity",  label: "Fuel Capacity",  unit: "gal",  step: 0.1, decimals: 1 },
  { key: "Fuel_efficiency",label: "Fuel Efficiency",unit: "mpg",  step: 1,   decimals: 0 },
];

/* ── Segment helper ─────────────────────── */
function getPriceSegment(val) {
  if (val < 15)  return { label: "Entry Level",   icon: "◌" };
  if (val < 30)  return { label: "Mid Range",     icon: "◎" };
  if (val < 55)  return { label: "Premium",       icon: "◉" };
  return              { label: "Luxury",          icon: "⬡" };
}

/* ── NumberInput ────────────────────────── */
function NumberInput({ feature, value, min, max, onChange }) {
  const increment = () => {
    const next = parseFloat((parseFloat(value) + feature.step).toFixed(feature.decimals + 1));
    onChange(Math.min(next, max));
  };
  const decrement = () => {
    const next = parseFloat((parseFloat(value) - feature.step).toFixed(feature.decimals + 1));
    onChange(Math.max(next, min));
  };
  return (
    <div className="input-group">
      <label className="input-label">{feature.label}</label>
      <div className="input-wrapper">
        <input
          type="number"
          className="num-input"
          value={value}
          min={min}
          max={max}
          step={feature.step}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v)) onChange(v);
          }}
        />
        <span className="input-unit">{feature.unit}</span>
        <div className="spin-btns">
          <button className="spin-btn" onClick={increment} tabIndex={-1}>▲</button>
          <button className="spin-btn" onClick={decrement} tabIndex={-1}>▼</button>
        </div>
      </div>
    </div>
  );
}

/* ── AnimatedPrice ──────────────────────── */
function AnimatedPrice({ value }) {
  const [display, setDisplay] = useState(value);
  const prevRef = useRef(value);
  useEffect(() => {
    const start = prevRef.current;
    const end = value;
    prevRef.current = value;
    const duration = 800;
    const startTime = performance.now();
    const tick = (now) => {
      const p = Math.min((now - startTime) / duration, 1);
      const ease = 1 - Math.pow(1 - p, 4);
      setDisplay(start + (end - start) * ease);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [value]);
  return "$" + (display * 1000).toLocaleString("en-US", { maximumFractionDigits: 0 });
}

/* ── BentoSummary ───────────────────────── */
function BentoSummary({ predicted, inputs }) {
  const seg = getPriceSegment(predicted);
  const priceUSD = predicted * 1000;
  const priceIDR = (priceUSD * 16200).toLocaleString("id-ID");
  const hp  = inputs["Horsepower"]      ?? 0;
  const eng = inputs["Engine_size"]     ?? 0;
  const eff = inputs["Fuel_efficiency"] ?? 0;
  const wt  = inputs["Curb_weight"]     ?? 0;
  const pwRatio = hp > 0 && wt > 0 ? (hp / wt).toFixed(1) : "—";

  return (
    <div className="bento-grid">
      {/* Segmen pasar — wide */}
      <div className="bento-cell wide highlight">
        <span className="bento-icon">{seg.icon}</span>
        <span className="bento-label">Segmen Pasar</span>
        <span className="bento-value">{seg.label}</span>
        <span className="bento-badge">Berdasarkan prediksi model</span>
      </div>

      {/* Estimasi IDR */}
      <div className="bento-cell">
        <span className="bento-icon">Rp</span>
        <span className="bento-label">Est. IDR</span>
        <span className="bento-value sm">±{priceIDR.slice(0, 10)}…</span>
      </div>

      {/* Tenaga */}
      <div className="bento-cell">
        <span className="bento-icon">⚡</span>
        <span className="bento-label">Tenaga</span>
        <span className="bento-value">{hp} <span style={{fontSize:11,color:'var(--text-dim)'}}>HP</span></span>
      </div>

      {/* Mesin */}
      <div className="bento-cell">
        <span className="bento-icon">⬡</span>
        <span className="bento-label">Mesin</span>
        <span className="bento-value">{eng}<span style={{fontSize:11,color:'var(--text-dim)'}}>L</span></span>
      </div>

      {/* Power/Weight */}
      <div className="bento-cell">
        <span className="bento-icon">◈</span>
        <span className="bento-label">HP / ton</span>
        <span className="bento-value">{pwRatio}</span>
      </div>

      {/* Efisiensi */}
      <div className="bento-cell">
        <span className="bento-icon">⛽</span>
        <span className="bento-label">Efisiensi</span>
        <span className="bento-value">{eff}<span style={{fontSize:11,color:'var(--text-dim)'}}>mpg</span></span>
      </div>
    </div>
  );
}

/* ── App ────────────────────────────────── */
export default function App() {
  const [model,     setModel]     = useState(null);
  const [inputs,    setInputs]    = useState({});
  const [predicted, setPredicted] = useState(null);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);
  const [predicting,setPredicting]= useState(false);
  const [csvInfo,   setCsvInfo]   = useState({ total: 0, valid: 0 });

  useEffect(() => {
    fetch("/Car_sales.csv")
      .then((r) => {
        if (!r.ok) throw new Error("Car_sales.csv tidak ditemukan di folder public/");
        return r.text();
      })
      .then((text) => {
        const rows  = parseCSV(text);
        const valid = getValidRows(rows);
        setCsvInfo({ total: rows.length, valid: valid.length });
        const trained = trainLinearRegression(valid);
        setModel(trained);
        const defaults = {};
        FEATURES.forEach((f) => {
          const s = trained.featureStats[f.key];
          defaults[f.key] = parseFloat(s.mean.toFixed(f.decimals));
        });
        setInputs(defaults);
        setLoading(false);
      })
      .catch((err) => { setError(err.message); setLoading(false); });
  }, []);

  const handleChange  = (key, val) => setInputs((p) => ({ ...p, [key]: val }));

  const handlePredict = () => {
    if (!model) return;
    setPredicting(true);
    setTimeout(() => {
      setPredicted(predictPrice(inputs, model.theta));
      setPredicting(false);
    }, 600);
  };

  const handleReset = () => {
    if (!model) return;
    const defaults = {};
    FEATURES.forEach((f) => {
      const s = model.featureStats[f.key];
      defaults[f.key] = parseFloat(s.mean.toFixed(f.decimals));
    });
    setInputs(defaults);
    setPredicted(null);
  };

  /* ── Splash states ── */
  if (loading)
    return (
      <div className="splash">
        <div className="glow-orb" />
        <div className="splash-card">
          <div className="loader-ring" />
          <p className="splash-title">MEMUAT MODEL</p>
          <p className="splash-sub">Membaca dataset & menghitung koefisien…</p>
        </div>
      </div>
    );

  if (error)
    return (
      <div className="splash">
        <div className="glow-orb" />
        <div className="splash-card">
          <p className="splash-title" style={{ color: "#ff4444" }}>ERROR</p>
          <p className="splash-sub">{error}</p>
        </div>
      </div>
    );

  /* ── Main render ── */
  return (
    <div className="app">
      <div className="orb orb-1" />
      <div className="orb orb-2" />
      <div className="orb orb-3" />

      {/* NAV */}
      <nav className="navbar">
        <div className="nav-logo">
          <span className="logo-icon">◈</span>
          <span className="logo-text">CARPREDICT</span>
        </div>
      </nav>

      {/* HERO */}
      <section className="hero">
        <h1 className="hero-title">
          SMART CAR
          <br />
          <span className="hero-accent">PRICE PREDICTION</span>
        </h1>
        <p className="hero-sub">
          Analyze vehicle specifications and find the best price estimate in
          seconds.
        </p>
      </section>

      {/* MAIN CARD */}
      <main className="main-card">
        {/* LEFT */}
        <div className="left-panel">
          <div className="section-label">⬡ Spesifikasi Kendaraan</div>
          <div className="inputs-grid">
            {FEATURES.map((f) => {
              const stats = model?.featureStats[f.key] ?? {
                min: 0,
                max: 100,
                mean: 50,
              };
              return (
                <NumberInput
                  key={f.key}
                  feature={f}
                  value={
                    inputs[f.key] ?? parseFloat(stats.mean.toFixed(f.decimals))
                  }
                  min={parseFloat(stats.min.toFixed(f.decimals))}
                  max={parseFloat(stats.max.toFixed(f.decimals))}
                  onChange={(val) => handleChange(f.key, val)}
                />
              );
            })}
          </div>

          <div className="btn-row">
            <button
              className="btn-primary"
              onClick={handlePredict}
              disabled={predicting}
            >
              {predicting ? (
                <>
                  <span className="spinner" /> Menghitung…
                </>
              ) : (
                "Hitung Harga"
              )}
            </button>
            <button className="btn-ghost" onClick={handleReset}>
              Reset
            </button>
          </div>
        </div>

        {/* DIVIDER */}
        <div className="divider-v" />

        {/* RIGHT */}
        <div className="right-panel">
          <div className="section-label">◎ Estimasi Harga Pasar</div>

          {/* Price card */}
          <div className="price-card">
            <div className="price-glow" />
            {predicted !== null ? (
              <>
                <div className="price-tag">USD · ESTIMASI</div>
                <div className="price-big">
                  <AnimatedPrice value={predicted} />
                </div>
                <div className="price-sub-text">
                  ${predicted.toFixed(2)}k · ribuan dolar Amerika
                </div>
                <div className="price-meter">
                  <div
                    className="price-meter-fill"
                    style={{
                      width: `${Math.min((predicted / 85.5) * 100, 100)}%`,
                    }}
                  />
                </div>
                <div className="price-meter-labels">
                  <span>$9k</span>
                  <span>$85k+</span>
                </div>
              </>
            ) : (
              <div className="price-empty">
                <div className="empty-icon">◎</div>
                <div className="empty-text">
                  Masukkan spesifikasi lalu
                  <br />
                  tekan{" "}
                  <strong style={{ color: "var(--orange)" }}>
                    Hitung Harga
                  </strong>
                </div>
              </div>
            )}
          </div>

          {/* Bento Summary — tampil setelah prediksi */}
          {predicted !== null && (
            <BentoSummary predicted={predicted} inputs={inputs} />
          )}

          {/* Credit box */}
          <div className="credit-card">
            <div className="credit-title">◌ Dibuat Oleh</div>

            <div className="credit-field">
              <span className="credit-key">Nama</span>
              <span className="credit-sep">:</span>
              <span className="credit-value">Ikhwan Kurniawan Julianto</span>
            </div>

            <div className="credit-field">
              <span className="credit-key">NPM</span>
              <span className="credit-sep">:</span>
              <span className="credit-value">237006102</span>
            </div>
          </div>
        </div>
      </main>

      {/* SITE FOOTER */}
      <footer className="site-footer">
        <div className="footer-inner">
          {/* Brand */}
          <div className="footer-brand">
            <span className="footer-brand-name">CARPREDICT</span>
            <span className="footer-brand-desc">
              Car Price Prediction System
            </span>
          </div>

          <div className="footer-sep" />

          {/* Credits */}
          <div className="footer-credits">
            <div className="footer-credit-row">
              <span className="footer-credit-label">Nama</span>
              <span className="footer-credit-val">
                Ikhwan Kurniawan Julianto
              </span>
            </div>
            <div className="footer-credit-row">
              <span className="footer-credit-label">NPM</span>
              <span className="footer-credit-val">237006102</span>
            </div>
            <div className="footer-credit-row">
              <span className="footer-credit-label">MK</span>
              <span className="footer-credit-val">Data Science</span>
            </div>
          </div>

          <div className="footer-sep" />

          {/* Tech stack */}
          <div className="footer-tech">
            <span className="footer-tech-label">Teknologi</span>
            <div className="footer-tech-pills">
              <span className="tech-pill orange">React</span>
              <span className="tech-pill">Vite</span>
              <span className="tech-pill orange">Linear Regression</span>
              <span className="tech-pill">Car Sales Dataset</span>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="footer-bottom">
          <span className="footer-copy">
            Dataset: Car Sales Dataset (Kaggle)
          </span>
          <span className="footer-dot" />
          <span className="footer-copy">
            Model aktif · {new Date().getFullYear()}
          </span>
        </div>
      </footer>
    </div>
  );
}
