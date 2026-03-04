import { useState, useEffect, useRef, useCallback } from "react";

// ── Simulation Engine ──────────────────────────────────────────────────────

const DELTA = 0.018;

function sha256Mock(input) {
  let h = 0;
  for (let i = 0; i < input.length; i++) {
    h = (Math.imul(31, h) + input.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(16).padStart(8, "0") +
    Math.abs(h * 0x9e3779b9 >>> 0).toString(16).padStart(8, "0") +
    Math.abs(h * 0x6c62272e >>> 0).toString(16).padStart(8, "0") +
    Math.abs(h * 0x517cc1b7 >>> 0).toString(16).padStart(8, "0");
}

function computeLambda1(epoch, phase, noiseLevel = 0.003) {
  const noise = (Math.random() - 0.5) * noiseLevel;
  if (phase === "collapse") return -0.005 + noise * 0.5;
  if (phase === "critical") return 0.008 + noise;
  const base = 0.04 + Math.sin(epoch * 0.15) * 0.01;
  return Math.max(0.001, base + noise);
}

function getOracleDecision(lambda1) {
  if (lambda1 > DELTA) return "NOMINAL";
  if (lambda1 > 0) return "ALERT";
  return "HALT_AND_ROLLBACK";
}

function generateEvent(type) {
  const events = {
    transaction: [
      { id: `TXN-${Math.floor(Math.random()*99999)}`, amount: `$${(Math.random()*50000+1000).toFixed(2)}`, from: "ACCT-8821", to: "ACCT-3347", flag: Math.random() > 0.85 ? "⚠ AML" : "✓" },
      { id: `TXN-${Math.floor(Math.random()*99999)}`, amount: `$${(Math.random()*200000+5000).toFixed(2)}`, from: "CORP-4492", to: "OFFSHR-001", flag: Math.random() > 0.7 ? "⚠ SUSPICIOUS" : "✓" },
      { id: `TXN-${Math.floor(Math.random()*99999)}`, amount: `$${(Math.random()*5000+100).toFixed(2)}`, from: "ACCT-2219", to: "ACCT-8874", flag: "✓" },
    ],
    fraud: [
      { id: `FRD-${Math.floor(Math.random()*9999)}`, score: (Math.random()*0.4+0.6).toFixed(3), card: `****${Math.floor(Math.random()*9999)}`, merchant: "UNKNOWN-VENDOR", action: "BLOCK" },
      { id: `FRD-${Math.floor(Math.random()*9999)}`, score: (Math.random()*0.3).toFixed(3), card: `****${Math.floor(Math.random()*9999)}`, merchant: "AMAZON", action: "PASS" },
    ],
    log: [
      { id: `LOG-${Math.floor(Math.random()*99999)}`, type: "VPC_FLOW", src: `10.${Math.floor(Math.random()*255)}.${Math.floor(Math.random()*255)}.1`, thermal: (Math.random()*0.8+0.1).toFixed(4) },
      { id: `LOG-${Math.floor(Math.random()*99999)}`, type: "FW_DENY", src: `192.168.${Math.floor(Math.random()*255)}.1`, thermal: (Math.random()*0.3).toFixed(4) },
    ]
  };
  const arr = events[type];
  return arr[Math.floor(Math.random() * arr.length)];
}

function generateAgentDecision(lambda1) {
  const decisions = [
    { type: "FRAUD_ALERT", confidence: 0.94, path: "ToT→GoT", rq: (Math.random()*0.2+0.1).toFixed(4), wdvv: "✓ PASS", action: "BLOCK TXN-" + Math.floor(Math.random()*99999) },
    { type: "AML_FLAG", confidence: 0.87, path: "GoT", rq: (Math.random()*0.3+0.05).toFixed(4), wdvv: "✓ PASS", action: "ESCALATE ACCT-" + Math.floor(Math.random()*9999) },
    { type: "RISK_SCORE", confidence: 0.71, path: "CoT", rq: (Math.random()*0.15+0.02).toFixed(4), wdvv: "✓ PASS", action: "SCORE " + (Math.random()*40+60).toFixed(1) },
    { type: "APPROVED", confidence: 0.99, path: "CoT", rq: (Math.random()*0.05+0.01).toFixed(4), wdvv: "✓ PASS", action: "APPROVE TXN-" + Math.floor(Math.random()*99999) },
  ];
  if (lambda1 < DELTA) {
    return { type: "ABSTAIN", confidence: 0, path: "WDVV→∞", rq: "∞", wdvv: "✗ FAIL", action: "STRUCTURED ABSTENTION" };
  }
  return decisions[Math.floor(Math.random() * decisions.length)];
}

// ── Sparkline Component ────────────────────────────────────────────────────
function Sparkline({ data, width = 200, height = 48, delta = DELTA }) {
  if (!data.length) return null;
  const max = Math.max(...data, delta * 2, 0.06);
  const min = Math.min(...data, -0.01);
  const range = max - min || 0.01;
  const pts = data.map((v, i) => {
    const x = (i / Math.max(data.length - 1, 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  }).join(" ");
  const deltaY = height - ((delta - min) / range) * height;
  const zeroY  = height - ((0 - min) / range) * height;
  const last = data[data.length - 1];
  const color = last > delta ? "#00ff9d" : last > 0 ? "#ffb300" : "#ff3d3d";
  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <line x1={0} y1={deltaY} x2={width} y2={deltaY} stroke="#ffb300" strokeWidth={0.8} strokeDasharray="3,3" opacity={0.6} />
      <line x1={0} y1={zeroY}  x2={width} y2={zeroY}  stroke="#ff3d3d" strokeWidth={0.8} strokeDasharray="2,2" opacity={0.4} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} />
      <circle cx={(data.length-1)/(Math.max(data.length-1,1))*width} cy={height-((last-min)/range)*height} r={3} fill={color} />
    </svg>
  );
}

// ── Phase Badge ────────────────────────────────────────────────────────────
function PhaseBadge({ decision }) {
  const cfg = {
    NOMINAL:          { bg: "#00ff9d22", border: "#00ff9d", text: "#00ff9d", label: "PHASE I · NOMINAL" },
    ALERT:            { bg: "#ffb30022", border: "#ffb300", text: "#ffb300", label: "PHASE II · ALERT" },
    HALT_AND_ROLLBACK:{ bg: "#ff3d3d22", border: "#ff3d3d", text: "#ff3d3d", label: "PHASE III · HALT" },
  }[decision] || { bg: "#ffffff11", border: "#888", text: "#888", label: decision };
  return (
    <span style={{
      background: cfg.bg, border: `1px solid ${cfg.border}`,
      color: cfg.text, borderRadius: 3, padding: "2px 8px",
      fontSize: 10, fontFamily: "monospace", letterSpacing: 1,
      animation: decision === "HALT_AND_ROLLBACK" ? "pulse 0.8s infinite" : "none",
    }}>{cfg.label}</span>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [running, setRunning]         = useState(false);
  const [tick, setTick]               = useState(0);
  const [simPhase, setSimPhase]       = useState("nominal");

  // Core metrics
  const [lambda1, setLambda1]         = useState(0.045);
  const [lambda1History, setL1Hist]   = useState([0.045]);
  const [oracleDecision, setOracle]   = useState("NOMINAL");
  const [epoch, setEpoch]             = useState(0);
  const [trainLoss, setTrainLoss]     = useState(0.42);
  const [valLoss, setValLoss]         = useState(0.48);

  // Pipeline data
  const [kafkaEvents, setKafkaEvents] = useState([]);
  const [flinkFeats, setFlinkFeats]   = useState([]);
  const [agentDecisions, setAgentDec] = useState([]);
  const [ledgerEntries, setLedger]    = useState([]);
  const [rollbackLog, setRollback]    = useState([]);

  // Bridge metrics
  const [qStar, setQstar]             = useState(2.718);
  const [lrOpt, setLrOpt]             = useState(0.00089);
  const [kappa, setKappa]             = useState(1.42);
  const [pruneCount, setPrune]        = useState(0);
  const [wdvvRes, setWdvv]            = useState(0.0000003);
  const [dH, setDh]                   = useState(5.2);
  const [beta0, setBeta0]             = useState(1);
  const [prevHash, setPrevHash]       = useState("0".repeat(32));

  // C1–C10 gate
  const [gatePass, setGatePass]       = useState(Array(10).fill(true));

  const tickRef  = useRef(0);
  const epochRef = useRef(0);

  const step = useCallback(() => {
    tickRef.current += 1;
    const t = tickRef.current;
    epochRef.current = Math.floor(t / 3);

    const phase =
      simPhase === "collapse" ? "collapse" :
      simPhase === "critical" ? "critical" : "nominal";

    const lam = computeLambda1(epochRef.current, phase);
    const dec = getOracleDecision(lam);

    setLambda1(lam);
    setL1Hist(h => [...h.slice(-79), lam]);
    setOracle(dec);
    setEpoch(epochRef.current);
    setTrainLoss(v => Math.max(0.01, v + (Math.random()-0.52)*0.008));
    setValLoss(v  => Math.max(0.01, v + (Math.random()-0.51)*0.006));

    // Kafka: new events
    if (t % 1 === 0) {
      const types = ["transaction","transaction","transaction","fraud","log"];
      const type  = types[Math.floor(Math.random()*types.length)];
      const ev    = generateEvent(type);
      const thermal = Math.random();
      const qstar   = 2.718 + (Math.random()-0.5)*0.3;
      const thresh  = Math.log(qstar) / (2*Math.PI);
      setKafkaEvents(arr => [{
        ...ev, type, thermal: thermal.toFixed(4),
        passed: thermal > thresh,
        ts: new Date().toISOString().substr(11,12)
      }, ...arr].slice(0,12));
    }

    // Flink features every 2 ticks
    if (t % 2 === 0) {
      setFlinkFeats(arr => [{
        feature: ["txn_freq_5m","velocity_1h","cross_border_ratio","card_present_ratio","acct_age_score"][Math.floor(Math.random()*5)],
        value: (Math.random()*10).toFixed(4),
        window: ["5m","1h","24h"][Math.floor(Math.random()*3)],
        ts: new Date().toISOString().substr(11,12)
      }, ...arr].slice(0,8));
    }

    // Agent decisions every 3 ticks
    if (t % 3 === 0) {
      const dec2 = generateAgentDecision(lam);
      setAgentDec(arr => [{ ...dec2, ts: new Date().toISOString().substr(11,12) }, ...arr].slice(0,10));
    }

    // Landau bridges
    const qs = 2.718 + Math.sin(t*0.1)*0.2;
    const kp = 1.4  + Math.sin(t*0.07)*0.3;
    setQstar(qs);
    setKappa(kp);
    setLrOpt(0.001 * Math.log(qs) / kp);
    setPrune(Math.floor(Math.random()*15 + 80));
    setWdvv(1e-7 + Math.random()*1e-8);
    setDh(5.2 + (Math.random()-0.5)*0.4);
    setBeta0(1);

    // Ledger
    if (t % 3 === 0) {
      const hash = sha256Mock(`${lam}${beta0}${dH}${prevHash}${t}`);
      setPrevHash(hash);
      setLedger(arr => [{
        epoch: epochRef.current,
        lambda1: lam.toFixed(6),
        decision: getOracleDecision(lam),
        dH: (5.2 + (Math.random()-0.5)*0.4).toFixed(3),
        wdvv: (1e-7 + Math.random()*1e-8).toExponential(2),
        hash: hash.slice(0,16) + "…",
        ts: new Date().toISOString().substr(11,12)
      }, ...arr].slice(0,15));
    }

    // Rollback event
    if (dec === "HALT_AND_ROLLBACK") {
      setRollback(arr => [{
        epoch: epochRef.current,
        lambda1: lam.toFixed(6),
        restoredTo: Math.max(0, epochRef.current - 8),
        ts: new Date().toISOString().substr(11,12)
      }, ...arr].slice(0,5));
    }

    // C1–C10 gate
    const newGate = [
      lam > DELTA,                          // C1 spectral
      lam > 0,                              // C2 painlevé
      (1e-7 + Math.random()*1e-8) < 1e-6,  // C3 wdvv
      true,                                 // C4 ph_sp
      true,                                 // C5 hausdorff
      true,                                 // C6 ledger
      true,                                 // C7 london
      true,                                 // C8 lld
      true,                                 // C9 lktl
      true,                                 // C10 cssg
    ];
    setGatePass(newGate);
    setTick(t);
  }, [simPhase, beta0, dH, prevHash]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(step, 600);
    return () => clearInterval(id);
  }, [running, step]);

  const oracleColor = { NOMINAL: "#00ff9d", ALERT: "#ffb300", HALT_AND_ROLLBACK: "#ff3d3d" }[oracleDecision] || "#888";
  const gateLabels  = ["C1 spectral","C2 painlevé","C3 wdvv","C4 ph_sp","C5 hausdorff","C6 ledger","C7 london","C8 lld","C9 lktl","C10 cssg"];
  const allPass     = gatePass.every(Boolean);

  return (
    <div style={{
      fontFamily: "'Courier New', monospace",
      background: "#050a0f",
      minHeight: "100vh",
      color: "#c8d8e8",
      padding: "16px",
      fontSize: 11,
    }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes scanline { 0%{transform:translateY(-100%)} 100%{transform:translateY(100vh)} }
        @keyframes blink { 0%,100%{opacity:1} 49%{opacity:1} 50%{opacity:0} }
        ::-webkit-scrollbar{width:4px;background:#0a1520}
        ::-webkit-scrollbar-thumb{background:#1e3a5f}
        .card{background:#070e18;border:1px solid #1e3a5f;border-radius:4px;padding:10px;position:relative}
        .card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#1e6fb4,transparent)}
        .mono{font-family:'Courier New',monospace}
        .label{color:#4a7aa8;font-size:9px;letter-spacing:1px;text-transform:uppercase;margin-bottom:3px}
        .val{font-size:18px;font-weight:bold;letter-spacing:1px}
        .small-val{font-size:12px;letter-spacing:0.5px}
        .row{display:flex;gap:8px;margin-bottom:8px}
        .col{flex:1;min-width:0}
        .tag{display:inline-block;padding:1px 5px;border-radius:2px;font-size:9px;letter-spacing:1px}
        .scroll-list{max-height:160px;overflow-y:auto}
        .entry{padding:3px 0;border-bottom:1px solid #0d1e2e;display:flex;gap:6px;align-items:flex-start;font-size:10px}
        .ts{color:#3a5a7a;flex-shrink:0;font-size:9px}
        .glow-green{text-shadow:0 0 8px #00ff9d55}
        .glow-amber{text-shadow:0 0 8px #ffb30055}
        .glow-red{text-shadow:0 0 8px #ff3d3d55}
        .btn{background:#0d1e2e;border:1px solid #1e6fb4;color:#6ab4f0;padding:6px 16px;border-radius:3px;cursor:pointer;font-family:'Courier New',monospace;font-size:11px;letter-spacing:1px;transition:all 0.15s}
        .btn:hover{background:#1e3a5f;color:#a0d4ff}
        .btn.danger{border-color:#ff3d3d55;color:#ff8080}
        .btn.danger:hover{background:#2a0808}
        .btn.warn{border-color:#ffb30055;color:#ffcc66}
        .btn.warn:hover{background:#1a1200}
        .btn.active{background:#1e3a5f;border-color:#4a9eff;color:#a0d4ff}
        .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}
        .grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
      `}</style>

      {/* Header */}
      <div style={{ borderBottom: "1px solid #1e3a5f", paddingBottom: 10, marginBottom: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 13, color: "#4a9eff", letterSpacing: 3, fontWeight: "bold" }}>
            ◈ JORDAN-LIOUVILLE PRODUCTION AI SYSTEM
          </div>
          <div style={{ fontSize: 9, color: "#3a5a7a", letterSpacing: 2, marginTop: 2 }}>
            TIER-1 BANKING · SPECTRAL STABILITY ORACLE · OPERATOR: SYMMETRIZED EMPIRICAL FISHER
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <div style={{ textAlign: "right", marginRight: 8 }}>
            <div style={{ fontSize: 9, color: "#3a5a7a" }}>EPOCH</div>
            <div style={{ fontSize: 16, color: "#4a9eff", fontWeight: "bold" }}>{String(epoch).padStart(4,"0")}</div>
          </div>
          <div style={{ textAlign: "right", marginRight: 8 }}>
            <div style={{ fontSize: 9, color: "#3a5a7a" }}>TICKS</div>
            <div style={{ fontSize: 16, color: "#4a9eff", fontWeight: "bold" }}>{String(tick).padStart(5,"0")}</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <button className={`btn ${running ? "active" : ""}`} onClick={() => setRunning(r => !r)}>
              {running ? "⏸ PAUSE" : "▶ RUN SIM"}
            </button>
            <div style={{ display: "flex", gap: 4 }}>
              <button className="btn warn" style={{ fontSize: 9, padding: "3px 8px" }} onClick={() => setSimPhase("critical")}>⚡ CRITICAL</button>
              <button className="btn danger" style={{ fontSize: 9, padding: "3px 8px" }} onClick={() => setSimPhase("collapse")}>☠ COLLAPSE</button>
              <button className="btn" style={{ fontSize: 9, padding: "3px 8px" }} onClick={() => setSimPhase("nominal")}>↺ RESET</button>
            </div>
          </div>
        </div>
      </div>

      {/* TOP ROW: Oracle + Sparkline + Gate */}
      <div className="row">
        {/* Spectral Oracle */}
        <div className="card col" style={{ flex: "0 0 220px" }}>
          <div className="label">Spectral Oracle  𝓛_JL</div>
          <div style={{ marginBottom: 6 }}>
            <div style={{ fontSize: 9, color: "#3a5a7a" }}>λ₁ = λ_min(sym(Fisher))</div>
            <div className="val" style={{ color: oracleColor, animation: oracleDecision === "HALT_AND_ROLLBACK" ? "pulse 0.8s infinite" : "none" }}>
              {lambda1 >= 0 ? "+" : ""}{lambda1.toFixed(6)}
            </div>
          </div>
          <div style={{ marginBottom: 8 }}><PhaseBadge decision={oracleDecision} /></div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, fontSize: 10 }}>
            <div><span style={{ color: "#3a5a7a" }}>δ threshold:</span> <span style={{ color: "#6ab4f0" }}>{DELTA}</span></div>
            <div><span style={{ color: "#3a5a7a" }}>margin:</span> <span style={{ color: (lambda1 - DELTA) > 0 ? "#00ff9d" : "#ff3d3d" }}>{(lambda1 - DELTA >= 0 ? "+" : "")}{(lambda1 - DELTA).toFixed(5)}</span></div>
            <div><span style={{ color: "#3a5a7a" }}>train loss:</span> <span style={{ color: "#c8d8e8" }}>{trainLoss.toFixed(4)}</span></div>
            <div><span style={{ color: "#3a5a7a" }}>val loss:</span>   <span style={{ color: "#c8d8e8" }}>{valLoss.toFixed(4)}</span></div>
          </div>
          {oracleDecision === "HALT_AND_ROLLBACK" && (
            <div style={{ marginTop: 8, background: "#1a000033", border: "1px solid #ff3d3d44", borderRadius: 3, padding: 6, fontSize: 9, color: "#ff8080", animation: "pulse 1s infinite" }}>
              ⚠ AUTOMATED ROLLBACK TRIGGERED<br/>Restoring last λ₁ &gt; milestone checkpoint
            </div>
          )}
          {oracleDecision === "ALERT" && (
            <div style={{ marginTop: 8, background: "#1a0d0033", border: "1px solid #ffb30044", borderRadius: 3, padding: 6, fontSize: 9, color: "#ffcc66" }}>
              ⚡ PHASE II DETECTED — MARGIN BELOW δ<br/>Trend monitoring: active
            </div>
          )}
        </div>

        {/* Lambda1 sparkline */}
        <div className="card col">
          <div className="label">λ₁ History — Fisher Ground Eigenvalue</div>
          <div style={{ marginBottom: 4 }}>
            <Sparkline data={lambda1History} width={380} height={56} />
          </div>
          <div style={{ display: "flex", gap: 16, fontSize: 9, color: "#3a5a7a", marginTop: 2 }}>
            <span><span style={{ color: "#ffb300" }}>— — </span>δ = {DELTA} (calibrated threshold)</span>
            <span><span style={{ color: "#ff3d3d" }}>— — </span>λ₁ = 0 (collapse boundary)</span>
            <span style={{ marginLeft: "auto", color: "#6ab4f0" }}>float64 · 10⁻¹⁵ resolution · Lanczos if d&gt;1000</span>
          </div>

          <div style={{ marginTop: 8, display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 6 }}>
            {[["WDVV Residual", wdvvRes.toExponential(2), wdvvRes < 1e-6 ? "#00ff9d" : "#ff3d3d"],
              ["Hausdorff d_H", dH.toFixed(3), "#6ab4f0"],
              ["β₀ (Betti-0)", beta0, "#6ab4f0"],
              ["Prunable θ", pruneCount + "%", pruneCount < 90 ? "#00ff9d" : "#ffb300"]
            ].map(([l,v,c]) => (
              <div key={l} style={{ background: "#0d1e2e55", padding: "5px 8px", borderRadius: 3 }}>
                <div style={{ fontSize: 8, color: "#3a5a7a", marginBottom: 2 }}>{l}</div>
                <div style={{ fontSize: 13, color: c, fontWeight: "bold" }}>{v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Twenty-Language Gate */}
        <div className="card" style={{ flex: "0 0 180px" }}>
          <div className="label">Twenty-Language Gate</div>
          <div style={{ marginBottom: 6 }}>
            <span className="tag" style={{
              background: allPass ? "#00ff9d22" : "#ff3d3d22",
              border: `1px solid ${allPass ? "#00ff9d" : "#ff3d3d"}`,
              color: allPass ? "#00ff9d" : "#ff3d3d",
              fontSize: 10, padding: "2px 8px",
              animation: !allPass ? "pulse 1s infinite" : "none"
            }}>
              {allPass ? "✓ PROMOTE" : "✗ BLOCK"}
            </span>
          </div>
          {gateLabels.map((label, i) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 2 }}>
              <span style={{ color: gatePass[i] ? "#00ff9d" : "#ff3d3d", fontSize: 10 }}>
                {gatePass[i] ? "✓" : "✗"}
              </span>
              <span style={{ fontSize: 9, color: gatePass[i] ? "#6ab4f0" : "#ff8080" }}>{label}</span>
              {i === 5 && <span style={{ fontSize: 8, color: "#00ff9d88", marginLeft: "auto" }}>PROVED</span>}
            </div>
          ))}
        </div>
      </div>

      {/* MIDDLE ROW: Pipeline */}
      <div className="row" style={{ alignItems: "stretch" }}>

        {/* Layer 1: Kafka + LKTL */}
        <div className="card col">
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <div className="label">① KAFKA + LKTL</div>
            <span className="tag" style={{ background: "#1e3a5f", color: "#4a9eff", border: "1px solid #1e6fb4" }}>INGESTION</span>
          </div>
          <div style={{ fontSize: 9, color: "#3a5a7a", marginBottom: 6 }}>
            Landau thermal gate · q*={qStar.toFixed(3)} · threshold=ln(q*)/2π={( Math.log(qStar)/(2*Math.PI) ).toFixed(4)}
          </div>
          <div className="scroll-list">
            {kafkaEvents.map((ev, i) => (
              <div key={i} className="entry">
                <span className="ts">{ev.ts}</span>
                <span className="tag" style={{ background: ev.passed ? "#00ff9d11" : "#ff3d3d11", border: `1px solid ${ev.passed ? "#00ff9d33" : "#ff3d3d33"}`, color: ev.passed ? "#00ff9d" : "#ff3d3d", fontSize: 8 }}>
                  {ev.passed ? "PASS" : "DAMP"}
                </span>
                <span style={{ color: "#6ab4f0", flexShrink: 0 }}>{ev.type?.toUpperCase()}</span>
                <span style={{ color: "#c8d8e8", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {ev.id} {ev.amount || ev.score || ""} {ev.flag || ev.action || ""}
                </span>
                <span style={{ marginLeft: "auto", color: "#3a5a7a", flexShrink: 0 }}>ε={ev.thermal}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Layer 2: Flink Features */}
        <div className="card col">
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <div className="label">② FLINK — Feature Engineering</div>
            <span className="tag" style={{ background: "#1e3a5f", color: "#4a9eff", border: "1px solid #1e6fb4" }}>TRANSFORM</span>
          </div>
          <div style={{ fontSize: 9, color: "#3a5a7a", marginBottom: 6 }}>
            Real-time features · distributed Fisher shards · global λ₁=min(shards)
          </div>
          <div className="scroll-list">
            {flinkFeats.map((f, i) => (
              <div key={i} className="entry">
                <span className="ts">{f.ts}</span>
                <span style={{ color: "#6ab4f0", flexShrink: 0 }}>{f.feature}</span>
                <span style={{ color: "#c8d8e8" }}>= {f.value}</span>
                <span className="tag" style={{ background: "#1e3a5f44", border: "1px solid #1e6fb422", color: "#4a7aa8", marginLeft: "auto", flexShrink: 0 }}>
                  window={f.window}
                </span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 6, borderTop: "1px solid #0d1e2e", paddingTop: 6 }}>
            <div style={{ fontSize: 9, color: "#3a5a7a" }}>Fisher precision strategy</div>
            <div style={{ display: "flex", gap: 8, marginTop: 3 }}>
              <span style={{ fontSize: 9 }}><span style={{ color: "#ffb300" }}>weights:</span> <span style={{ color: "#c8d8e8" }}>float32</span></span>
              <span style={{ fontSize: 9 }}><span style={{ color: "#00ff9d" }}>Fisher+λ₁:</span> <span style={{ color: "#c8d8e8" }}>float64 · 10⁻¹⁵</span></span>
            </div>
          </div>
        </div>

        {/* Layer 3: LangGraph Agent Decisions */}
        <div className="card col">
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <div className="label">③ LANGGRAPH — Agent Decisions</div>
            <span className="tag" style={{ background: "#1e3a5f", color: "#4a9eff", border: "1px solid #1e6fb4" }}>REASONING</span>
          </div>
          <div style={{ fontSize: 9, color: "#3a5a7a", marginBottom: 6 }}>
            WDVV gate (learned F(t)) → Rayleigh Quotient min → Analyst→Critic loop
          </div>
          <div className="scroll-list">
            {agentDecisions.map((d, i) => (
              <div key={i} className="entry" style={{ flexWrap: "wrap" }}>
                <span className="ts">{d.ts}</span>
                <span className="tag" style={{
                  background: d.type === "ABSTAIN" ? "#ff3d3d22" : d.type === "FRAUD_ALERT" || d.type === "AML_FLAG" ? "#ffb30022" : "#00ff9d22",
                  border: `1px solid ${d.type === "ABSTAIN" ? "#ff3d3d44" : d.type === "FRAUD_ALERT" || d.type === "AML_FLAG" ? "#ffb30044" : "#00ff9d44"}`,
                  color: d.type === "ABSTAIN" ? "#ff8080" : d.type === "FRAUD_ALERT" || d.type === "AML_FLAG" ? "#ffcc66" : "#00ff9d",
                }}>
                  {d.type}
                </span>
                <span style={{ color: "#c8d8e8", marginLeft: 4 }}>{d.action}</span>
                <div style={{ width: "100%", display: "flex", gap: 10, marginTop: 2, paddingLeft: 44 }}>
                  <span style={{ color: "#3a5a7a" }}>path=<span style={{ color: "#6ab4f0" }}>{d.path}</span></span>
                  <span style={{ color: "#3a5a7a" }}>RQ=<span style={{ color: d.rq === "∞" ? "#ff8080" : "#6ab4f0" }}>{d.rq}</span></span>
                  <span style={{ color: "#3a5a7a" }}>WDVV=<span style={{ color: d.wdvv === "✓ PASS" ? "#00ff9d" : "#ff3d3d" }}>{d.wdvv}</span></span>
                  <span style={{ color: "#3a5a7a" }}>conf=<span style={{ color: "#c8d8e8" }}>{d.confidence}</span></span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* BOTTOM ROW: Four Bridges + SHA-256 Ledger + Rollbacks + K8s */}
      <div className="row" style={{ alignItems: "stretch" }}>

        {/* Four Landau Bridges */}
        <div className="card col">
          <div className="label">Four Landau Bridges — Calibration Laws</div>
          <div style={{ fontSize: 8, color: "#3a5a7a", marginBottom: 8 }}>All are calibration hypotheses · subject to empirical ablation · never hand-tuned</div>

          {/* Bridge 1 */}
          <div style={{ marginBottom: 8, borderBottom: "1px solid #0d1e2e", paddingBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ color: "#4a9eff", fontSize: 10, fontWeight: "bold" }}>H1 · KINETIC → LR Scaling</span>
              <span className="tag" style={{ background: "#1e3a5f33", color: "#3a7abf", border: "1px solid #1e3a5f", fontSize: 8 }}>Landau kinetic theory</span>
            </div>
            <div style={{ fontSize: 9 }}>
              <span style={{ color: "#3a5a7a" }}>lr*(t) = lr₀ × ln(q*)/κ(t) = </span>
              <span style={{ color: "#00ff9d", fontWeight: "bold" }}>{lrOpt.toFixed(6)}</span>
            </div>
            <div style={{ fontSize: 9, color: "#3a5a7a" }}>
              q*={qStar.toFixed(3)} (Farey) · κ(t)={kappa.toFixed(3)} (Hessian Frobenius)
            </div>
          </div>

          {/* Bridge 2 */}
          <div style={{ marginBottom: 8, borderBottom: "1px solid #0d1e2e", paddingBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ color: "#4a9eff", fontSize: 10, fontWeight: "bold" }}>H2 · THIN-FILM → Architecture</span>
              <span className="tag" style={{ background: "#1e3a5f33", color: "#3a7abf", border: "1px solid #1e3a5f", fontSize: 8 }}>LLD Law Ca^(2/3)</span>
            </div>
            <div style={{ fontSize: 9 }}>
              <span style={{ color: "#3a5a7a" }}>Δ ≈ A × (d_intrinsic/n_params)^(2/3) · d_H=</span>
              <span style={{ color: "#6ab4f0" }}>{dH.toFixed(3)}</span>
            </div>
            <div style={{ fontSize: 9, color: "#3a5a7a" }}>A fit from validation split · bootstrap CI required</div>
          </div>

          {/* Bridge 3 */}
          <div style={{ marginBottom: 8, borderBottom: "1px solid #0d1e2e", paddingBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ color: "#4a9eff", fontSize: 10, fontWeight: "bold" }}>H3 · SUPERCONDUCTIVITY → Pruning</span>
              <span className="tag" style={{ background: "#1e3a5f33", color: "#3a7abf", border: "1px solid #1e3a5f", fontSize: 8 }}>London depth</span>
            </div>
            <div style={{ fontSize: 9 }}>
              <span style={{ color: "#3a5a7a" }}>C_P(i) = |∂λ₁/∂θᵢ| · prunable=</span>
              <span style={{ color: pruneCount < 90 ? "#00ff9d" : "#ffb300" }}>{pruneCount}%</span>
            </div>
            <div style={{ fontSize: 9, color: "#3a5a7a" }}>ε_prune = 0.01 × mean(C_P) · calibrated</div>
          </div>

          {/* Bridge 4 */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ color: "#4a9eff", fontSize: 10, fontWeight: "bold" }}>H4 · CSSG → Grokking Control</span>
              <span className="tag" style={{ background: "#1e3a5f33", color: "#3a7abf", border: "1px solid #1e3a5f", fontSize: 8 }}>Schulze-Hardy z⁻⁶</span>
            </div>
            <div style={{ fontSize: 9 }}>
              <span style={{ color: "#3a5a7a" }}>rate ~ order⁻⁶ · L2→L4 = </span>
              <span style={{ color: "#00ff9d", fontWeight: "bold" }}>64× slower grokking</span>
            </div>
            <div style={{ display: "flex", gap: 6, marginTop: 3 }}>
              {[1,2,3,4,5].map(o => (
                <div key={o} style={{ background: o === 2 ? "#1e3a5f" : "#0d1e2e", border: `1px solid ${o === 2 ? "#1e6fb4" : "#0d1e2e"}`, borderRadius: 2, padding: "1px 5px", fontSize: 8 }}>
                  L{o} <span style={{ color: "#6ab4f0" }}>{(o**-6 / 2**-6).toFixed(2)}×</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* SHA-256 Ledger */}
        <div className="card col">
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <div className="label">④ SHA-256 Topology Ledger</div>
            <span className="tag" style={{ background: "#00ff9d11", color: "#00ff9d", border: "1px solid #00ff9d33", fontSize: 8 }}>CRYPTOGRAPHICALLY PROVED</span>
          </div>
          <div style={{ fontSize: 9, color: "#3a5a7a", marginBottom: 6 }}>
            HASH_t = SHA-256(λ₁‖β_k‖d_H‖HASH_{{"{t-1}"}}) · tamper-evident · millisecond granularity
          </div>
          <div className="scroll-list">
            {ledgerEntries.map((e, i) => (
              <div key={i} className="entry" style={{ flexWrap: "wrap" }}>
                <span className="ts">{e.ts}</span>
                <span style={{
                  color: e.decision === "NOMINAL" ? "#00ff9d" : e.decision === "ALERT" ? "#ffb300" : "#ff3d3d",
                  fontSize: 9, flexShrink: 0
                }}>●</span>
                <span style={{ color: "#4a7aa8", fontSize: 9 }}>ep={e.epoch}</span>
                <span style={{ color: "#c8d8e8", fontSize: 9 }}>λ₁={e.lambda1}</span>
                <span style={{ color: "#3a5a7a", fontSize: 9, marginLeft: "auto" }}>d_H={e.dH}</span>
                <div style={{ width: "100%", paddingLeft: 44 }}>
                  <span style={{ color: "#1e6fb4", fontSize: 8, fontFamily: "monospace" }}>{e.hash}</span>
                  <span style={{ color: "#3a5a7a", fontSize: 8, marginLeft: 8 }}>wdvv={e.wdvv}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* K8s + Rollback */}
        <div className="card col" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {/* K8s */}
          <div>
            <div className="label">⑤ KUBERNETES — Spectral Autoscaling</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 6 }}>
              {[
                ["HPA Pods", Math.max(3, Math.min(50, Math.round(3 + (1/(Math.max(lambda1,0.001))) * 0.1))),""],
                ["λ₁ inverse", (1/Math.max(lambda1,0.001)*100).toFixed(1), "HPA metric"],
                ["Probe", "/health/fisher_λ₁", ""],
                ["Blue/Green", oracleDecision === "NOMINAL" ? "BLUE ✓" : "CUTTING", oracleDecision !== "NOMINAL" ? "⚠" : ""],
              ].map(([l,v,s]) => (
                <div key={l} style={{ background: "#0d1e2e55", padding: "4px 7px", borderRadius: 3 }}>
                  <div style={{ fontSize: 8, color: "#3a5a7a" }}>{l} {s && <span style={{ color: "#ffb300" }}>{s}</span>}</div>
                  <div style={{ fontSize: 11, color: "#6ab4f0", marginTop: 1 }}>{v}</div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: 9, color: "#3a5a7a", borderTop: "1px solid #0d1e2e", paddingTop: 4 }}>
              Docker container env: JL_OPERATOR=empirical_fisher · JL_EIGENVALUE_DTYPE=float64
            </div>
          </div>

          {/* Rollback log */}
          <div style={{ flex: 1 }}>
            <div className="label" style={{ color: rollbackLog.length ? "#ff8080" : "#4a7aa8" }}>
              ⑥ BCP — Rollback History {rollbackLog.length > 0 && `(${rollbackLog.length})`}
            </div>
            {rollbackLog.length === 0 ? (
              <div style={{ fontSize: 9, color: "#1e3a5f", padding: "6px 0" }}>No rollbacks. System stable.</div>
            ) : (
              <div className="scroll-list" style={{ maxHeight: 100 }}>
                {rollbackLog.map((r, i) => (
                  <div key={i} className="entry" style={{ animation: i === 0 ? "pulse 2s 3" : "none" }}>
                    <span className="ts">{r.ts}</span>
                    <span style={{ color: "#ff8080", fontSize: 9 }}>ROLLBACK</span>
                    <span style={{ color: "#c8d8e8", fontSize: 9 }}>ep={r.epoch} λ₁={r.lambda1} → ep={r.restoredTo}</span>
                  </div>
                ))}
              </div>
            )}
            <div style={{ marginTop: 6, fontSize: 9, color: "#3a5a7a" }}>
              Geometric checkpoints at λ₁ &gt; 0.50 / 0.25 / 0.10 / 0.05<br/>
              Every saved checkpoint: spectrally certified · no human required
            </div>
          </div>

          {/* Multi-region */}
          <div style={{ borderTop: "1px solid #0d1e2e", paddingTop: 6 }}>
            <div className="label">Multi-Region · global λ₁ = min(shards)</div>
            <div style={{ display: "flex", gap: 6 }}>
              {["us-east-1","eastus","us-central1"].map(r => (
                <div key={r} style={{ flex: 1, background: "#0d1e2e55", padding: "3px 5px", borderRadius: 3, textAlign: "center" }}>
                  <div style={{ fontSize: 7, color: "#3a5a7a" }}>{r}</div>
                  <div style={{ fontSize: 9, color: oracleDecision === "NOMINAL" ? "#00ff9d" : "#ffb300" }}>
                    {(lambda1 * (0.95 + Math.random()*0.1)).toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{ borderTop: "1px solid #0d1e2e", paddingTop: 8, marginTop: 4, display: "flex", justifyContent: "space-between", fontSize: 8, color: "#1e3a5f" }}>
        <span>𝓛_JL(θ) = (F(θ) + F(θ)ᵀ)/2  ∈  Sym_n(ℝ) · Jordan product A∘B=(AB+BA)/2 · Albert algebra H₃(𝕆): extension roadmap only</span>
        <span>SHA-256 ledger: {ledgerEntries.length} entries · chain integrity: {ledgerEntries.length > 0 ? "✓ VALID" : "—"}</span>
        <span>© Jordan-Liouville Production AI System · Operator: Symmetrized Empirical Fisher</span>
      </div>
    </div>
  );
}
