"use client";

import { useState, useEffect, useCallback } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

interface PredictResponse {
  linear_regression: number;
  random_forest: number;
  xgboost: number;
  ensemble_avg: number;
  unit: string;
}

interface OverfitAnalysis {
  train_test_r2_gap: number;
  is_overfit: boolean;
  explanation: string;
}

interface ModelMetrics {
  train: { r2: number; mae: number; rmse: number; mape: number };
  test: { r2: number; mae: number; rmse: number; mape: number };
  cv_r2_mean: number;
  cv_r2_std: number;
  overfit_analysis?: OverfitAnalysis;
}

interface MetricsResponse {
  linear_regression: ModelMetrics;
  random_forest: ModelMetrics;
  xgboost: ModelMetrics;
}

interface FeatureImportance {
  feature: string;
  linear_regression: number;
  random_forest: number;
  xgboost: number;
}

interface DatasetStats {
  total_rows: number;
  real_rows: number;
  synthetic_rows: number;
  target: { mean: number; std: number; min: number; max: number };
}

interface AugModelMetrics {
  r2: number;
  mae: number;
  rmse: number;
  mape: number;
}

interface AugModelComparison {
  real_only: AugModelMetrics;
  augmented: AugModelMetrics;
  r2_change: number;
  mae_change: number;
}

interface AugmentationData {
  description: string;
  splits: {
    real_only_train: number;
    augmented_train: number;
    synthetic_added: number;
    test_set: number;
  };
  comparison: {
    linear_regression: AugModelComparison;
    random_forest: AugModelComparison;
    xgboost: AugModelComparison;
  };
}

// ─── Short display names for features ─────────────────────────────────────────

const SHORT_NAMES: Record<string, string> = {
  "Main engine shaft RPM": "RPM",
  "Main engine shaft power (KW)": "Power",
  "Mean draft (m)": "Draft",
  "Wave height (m)": "Waves",
  "Wind speed (m/s)": "Wind",
  "Current speed (m/s)": "Current",
  "Speed through water (kn)": "Speed",
  "Swell height (m)": "Swell",
  "Trim (m)": "Trim",
};

// ─── Parameter definitions ────────────────────────────────────────────────────

interface Param {
  key: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  default: number;
  description: string;
}

const PARAMS: Param[] = [
  { key: "rpm",                label: "Engine RPM",            unit: "RPM",  min: 0,   max: 120,  step: 1,    default: 80,   description: "Main engine shaft speed" },
  { key: "power",              label: "Shaft Power",           unit: "kW",   min: 0,   max: 12000, step: 100, default: 4500, description: "Main engine shaft power output" },
  { key: "mean_draft",         label: "Mean Draft",            unit: "m",    min: 5,   max: 22,   step: 0.1,  default: 14,   description: "Average vessel draft" },
  { key: "wave_height",        label: "Wave Height",           unit: "m",    min: 0,   max: 8,    step: 0.1,  default: 1.5,  description: "Significant wave height" },
  { key: "wind_speed",         label: "Wind Speed",            unit: "m/s",  min: 0,   max: 30,   step: 0.5,  default: 8,    description: "Wind speed at deck level" },
  { key: "current_speed",      label: "Current Speed",         unit: "m/s",  min: 0,   max: 3,    step: 0.05, default: 0.3,  description: "Ocean current speed" },
  { key: "speed_through_water",label: "Speed Through Water",   unit: "kn",   min: 0,   max: 22,   step: 0.1,  default: 12,   description: "Vessel speed relative to water" },
  { key: "swell_height",       label: "Swell Height",          unit: "m",    min: 0,   max: 8,    step: 0.1,  default: 1.2,  description: "Ocean swell height" },
  { key: "trim",               label: "Trim",                  unit: "m",    min: -3,  max: 3,    step: 0.1,  default: 0.5,  description: "Stern minus bow draft (positive = stern deeper)" },
];

// ─── Utility ──────────────────────────────────────────────────────────────────

function MetricBadge({ value, label }: { value: string; label: string }) {
  return (
    <div className="flex flex-col items-center bg-slate-800/60 rounded-xl px-4 py-3 min-w-[80px]">
      <span className="text-lg font-bold text-sky-400">{value}</span>
      <span className="text-xs text-slate-400 mt-0.5">{label}</span>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xl font-semibold text-slate-100 mb-4 flex items-center gap-2">
      {children}
    </h2>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [values, setValues] = useState<Record<string, number>>(
    Object.fromEntries(PARAMS.map((p) => [p.key, p.default]))
  );
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [augData, setAugData] = useState<AugmentationData | null>(null);
  const [activeTab, setActiveTab] = useState<"rf" | "xgb" | "lr">("xgb");

  // Fetch static data on mount
  useEffect(() => {
    fetch(`${API_URL}/metrics`).then((r) => r.json()).then(setMetrics).catch(() => {});
    fetch(`${API_URL}/feature-importance`).then((r) => r.json()).then(setFeatureImportance).catch(() => {});
    fetch(`${API_URL}/stats`).then((r) => r.json()).then(setStats).catch(() => {});
    fetch(`${API_URL}/augmentation-comparison`).then((r) => r.json()).then(setAugData).catch(() => {});
  }, []);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      const data = await res.json();
      setPrediction(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect to API");
    } finally {
      setLoading(false);
    }
  }, [values]);

  // Feature importance chart data
  const fiChartData = featureImportance.map((fi) => ({
    name: SHORT_NAMES[fi.feature] || fi.feature,
    RF: +(fi.random_forest * 100).toFixed(1),
    XGB: +(fi.xgboost * 100).toFixed(1),
  }));

  // Model comparison chart data
  const modelCompData = metrics
    ? [
        {
          metric: "R² Score",
          "Linear Reg": +(metrics.linear_regression.test.r2 * 100).toFixed(1),
          "Random Forest": +(metrics.random_forest.test.r2 * 100).toFixed(1),
          "XGBoost": +(metrics.xgboost.test.r2 * 100).toFixed(1),
        },
        {
          metric: "Accuracy (100-MAPE)",
          "Linear Reg": +(100 - metrics.linear_regression.test.mape).toFixed(1),
          "Random Forest": +(100 - metrics.random_forest.test.mape).toFixed(1),
          "XGBoost": +(100 - metrics.xgboost.test.mape).toFixed(1),
        },
      ]
    : [];

  const modelNames = ["linear_regression", "random_forest", "xgboost"] as const;
  const modelLabels = { linear_regression: "Linear Regression", random_forest: "Random Forest", xgboost: "XGBoost" };
  const modelColors = { linear_regression: "text-violet-400", random_forest: "text-emerald-400", xgboost: "text-amber-400" };
  const modelBg    = { linear_regression: "bg-violet-500/10 border-violet-500/30", random_forest: "bg-emerald-500/10 border-emerald-500/30", xgboost: "bg-amber-500/10 border-amber-500/30" };

  return (
    <main className="min-h-screen bg-slate-950">
      {/* ── Header ── */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">⚓</span>
            <div>
              <h1 className="text-lg font-bold text-slate-100 leading-none">ShipFuel AI</h1>
              <p className="text-xs text-slate-400">Engine Consumption Predictor</p>
            </div>
          </div>
          {stats && (
            <div className="hidden sm:flex items-center gap-4 text-xs text-slate-400">
              <span><span className="text-sky-400 font-semibold">{stats.total_rows.toLocaleString()}</span> records</span>
              <span><span className="text-emerald-400 font-semibold">{stats.real_rows.toLocaleString()}</span> real</span>
              <span><span className="text-amber-400 font-semibold">{stats.synthetic_rows.toLocaleString()}</span> synthetic</span>
            </div>
          )}
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-10">

        {/* ── Hero ── */}
        <div className="text-center space-y-3">
          <h2 className="text-3xl sm:text-4xl font-bold text-slate-100">
            Main Engine Fuel Consumption
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Set vessel operating parameters below and get instant predictions from three
            independently trained ML models — trained on{" "}
            <span className="text-sky-400 font-medium">92,567 real ship telemetry records</span>.
          </p>
        </div>

        {/* ── Input + Results grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Input Panel */}
          <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5">
            <SectionTitle>🎛️ Operating Parameters</SectionTitle>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {PARAMS.map((param) => (
                <div key={param.key} className="space-y-2">
                  <div className="flex justify-between items-baseline">
                    <label className="text-sm font-medium text-slate-300">{param.label}</label>
                    <span className="text-sm font-mono font-bold text-sky-400">
                      {values[param.key].toFixed(param.step < 1 ? 2 : 0)}{" "}
                      <span className="text-slate-500 font-normal text-xs">{param.unit}</span>
                    </span>
                  </div>
                  <input
                    type="range"
                    min={param.min}
                    max={param.max}
                    step={param.step}
                    value={values[param.key]}
                    onChange={(e) =>
                      setValues((v) => ({ ...v, [param.key]: parseFloat(e.target.value) }))
                    }
                  />
                  <p className="text-xs text-slate-500">{param.description}</p>
                </div>
              ))}
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full mt-2 py-3.5 rounded-xl bg-sky-500 hover:bg-sky-400 disabled:bg-sky-900 disabled:text-sky-600 text-slate-950 font-bold text-base transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              {loading ? "Predicting..." : "⚡ Predict Fuel Consumption"}
            </button>

            {error && (
              <div className="bg-red-900/30 border border-red-700/50 text-red-300 rounded-xl px-4 py-3 text-sm">
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <SectionTitle>📊 Predictions</SectionTitle>

              {!prediction ? (
                <div className="flex flex-col items-center justify-center h-40 text-slate-500 text-sm gap-3">
                  <span className="text-4xl">🚢</span>
                  <p>Adjust sliders and hit Predict</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {modelNames.map((model) => (
                    <div
                      key={model}
                      className={`border rounded-xl px-4 py-3 flex justify-between items-center ${modelBg[model]}`}
                    >
                      <div>
                        <p className={`text-xs font-medium ${modelColors[model]}`}>{modelLabels[model]}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold text-slate-100 font-mono">
                          {prediction[model].toFixed(0)}
                        </p>
                        <p className="text-xs text-slate-400">L/hr</p>
                      </div>
                    </div>
                  ))}

                  <div className="border border-sky-500/30 bg-sky-500/10 rounded-xl px-4 py-3 flex justify-between items-center">
                    <p className="text-xs font-medium text-sky-400">Ensemble Average</p>
                    <div className="text-right">
                      <p className="text-3xl font-bold text-sky-400 font-mono">
                        {prediction.ensemble_avg.toFixed(0)}
                      </p>
                      <p className="text-xs text-slate-400">L/hr</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Dataset composition */}
            {stats && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <SectionTitle>🗄️ Dataset</SectionTitle>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Total records</span>
                    <span className="font-mono text-slate-200">{stats.total_rows.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-emerald-400">Real telemetry</span>
                    <span className="font-mono text-slate-200">{stats.real_rows.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-amber-400">Synthetic (augmented)</span>
                    <span className="font-mono text-slate-200">{stats.synthetic_rows.toLocaleString()}</span>
                  </div>

                  {/* Composition bar */}
                  <div className="w-full h-3 bg-amber-500/40 rounded-full overflow-hidden mt-1">
                    <div
                      className="h-full bg-emerald-500 rounded-full"
                      style={{ width: `${(stats.real_rows / stats.total_rows) * 100}%` }}
                    />
                  </div>
                  <p className="text-xs text-slate-500">
                    {((stats.real_rows / stats.total_rows) * 100).toFixed(0)}% real · {((stats.synthetic_rows / stats.total_rows) * 100).toFixed(0)}% synthetic
                  </p>

                  <div className="pt-2 border-t border-slate-800 space-y-1">
                    <p className="text-xs text-slate-400 font-medium">Target: Main Engine Consumption</p>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">Mean</span>
                      <span className="font-mono">{stats.target.mean} L/hr</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">Range</span>
                      <span className="font-mono">{stats.target.min} – {stats.target.max} L/hr</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Model Metrics ── */}
        {metrics && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <SectionTitle>🏆 Model Performance</SectionTitle>

            {/* Tab selector */}
            <div className="flex gap-2 mb-6">
              {(["xgb", "rf", "lr"] as const).map((tab) => {
                const m = tab === "xgb" ? "xgboost" : tab === "rf" ? "random_forest" : "linear_regression";
                return (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
                      activeTab === tab
                        ? `${modelBg[m]} ${modelColors[m]} border`
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {modelLabels[m]}
                  </button>
                );
              })}
            </div>

            {/* Active model detail */}
            {(() => {
              const m = activeTab === "xgb" ? "xgboost" : activeTab === "rf" ? "random_forest" : "linear_regression";
              const d = metrics[m];
              return (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <MetricBadge value={d.test.r2.toFixed(4)} label="Test R²" />
                    <MetricBadge value={`${d.test.mae.toFixed(1)} L/hr`} label="MAE" />
                    <MetricBadge value={`${d.test.rmse.toFixed(1)} L/hr`} label="RMSE" />
                    <MetricBadge value={`${d.test.mape.toFixed(2)}%`} label="MAPE" />
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    <div className="bg-slate-800/50 rounded-lg px-3 py-2 flex justify-between">
                      <span className="text-slate-400">Train R²</span>
                      <span className="font-mono">{d.train.r2.toFixed(4)}</span>
                    </div>
                    <div className="bg-slate-800/50 rounded-lg px-3 py-2 flex justify-between">
                      <span className="text-slate-400">Gap</span>
                      <span className="font-mono text-emerald-400">{(d.train.r2 - d.test.r2).toFixed(4)}</span>
                    </div>
                    <div className="bg-slate-800/50 rounded-lg px-3 py-2 flex justify-between">
                      <span className="text-slate-400">5-CV R²</span>
                      <span className="font-mono">{d.cv_r2_mean.toFixed(4)} ±{d.cv_r2_std.toFixed(4)}</span>
                    </div>
                  </div>
                  {d.overfit_analysis ? (
                    <p className={`text-xs ${d.overfit_analysis.is_overfit ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {d.overfit_analysis.is_overfit ? '⚠️' : '✅'}{' '}
                      {d.overfit_analysis.is_overfit ? 'Potential overfitting' : 'No overfitting'} — train/test R² gap: {d.overfit_analysis.train_test_r2_gap.toFixed(4)}
                    </p>
                  ) : (
                    <p className={`text-xs ${
                      (d.train.r2 - d.test.r2) > 0.02 ? 'text-amber-400' : 'text-emerald-400'
                    }`}>
                      {(d.train.r2 - d.test.r2) > 0.02 ? '⚠️ Potential overfitting' : '✅ No overfitting'} — train/test gap: {(d.train.r2 - d.test.r2).toFixed(4)}
                    </p>
                  )}
                </div>
              );
            })()}

            {/* Comparison chart */}
            <div className="mt-6">
              <p className="text-sm text-slate-400 mb-3">R² Score comparison (test set)</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={modelCompData} layout="vertical" margin={{ left: 80, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" domain={[95, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: "#94a3b8", fontSize: 12 }} />
                  <YAxis type="category" dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                    labelStyle={{ color: "#94a3b8" }}
                    formatter={(v) => [`${v}%`]}
                  />
                  <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
                  <Bar dataKey="Linear Reg"    fill="#a78bfa" radius={4} />
                  <Bar dataKey="Random Forest" fill="#34d399" radius={4} />
                  <Bar dataKey="XGBoost"       fill="#fbbf24" radius={4} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ── Feature Importance ── */}
        {featureImportance.length > 0 && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <SectionTitle>📌 Feature Importance</SectionTitle>
            <p className="text-sm text-slate-400 mb-4">
              Which parameters drive fuel consumption most? (percentage of total importance)
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={fiChartData} margin={{ top: 4, right: 20, left: 10, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <YAxis tickFormatter={(v) => `${v}%`} tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                  formatter={(v) => [`${v}%`]}
                />
                <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
                <Bar dataKey="RF"  name="Random Forest" fill="#34d399" radius={[4,4,0,0]} />
                <Bar dataKey="XGB" name="XGBoost"       fill="#fbbf24" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>

            {/* Top feature callouts */}
            <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3">
              {featureImportance.slice(0, 3).map((fi, i) => (
                <div key={fi.feature} className="bg-slate-800/50 rounded-xl p-3 flex items-start gap-2">
                  <span className="text-lg">{i === 0 ? "🥇" : i === 1 ? "🥈" : "🥉"}</span>
                  <div>
                    <p className="text-xs font-medium text-slate-200">{fi.feature.replace(" (m)", "").replace(" (KW)", "").replace(" (kn)", "").replace(" (m/s)", "")}</p>
                    <p className="text-xs text-amber-400">{(fi.xgboost * 100).toFixed(1)}% (XGB)</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Radar chart — model strengths ── */}
        {metrics && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <SectionTitle>🎯 Model Accuracy Radar</SectionTitle>
            <p className="text-sm text-slate-400 mb-4">
              Higher is better for all axes. All values normalized to 0–100 scale.
            </p>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={[
                { metric: "R² ×100",   LR: +(metrics.linear_regression.test.r2*100).toFixed(1),  RF: +(metrics.random_forest.test.r2*100).toFixed(1),  XGB: +(metrics.xgboost.test.r2*100).toFixed(1) },
                { metric: "Accuracy",  LR: +(100-metrics.linear_regression.test.mape).toFixed(1), RF: +(100-metrics.random_forest.test.mape).toFixed(1), XGB: +(100-metrics.xgboost.test.mape).toFixed(1) },
                { metric: "Low MAE",   LR: +(100-metrics.linear_regression.test.mae/5).toFixed(1), RF: +(100-metrics.random_forest.test.mae/5).toFixed(1), XGB: +(100-metrics.xgboost.test.mae/5).toFixed(1) },
                { metric: "CV Score",  LR: +(metrics.linear_regression.cv_r2_mean*100).toFixed(1), RF: +(metrics.random_forest.cv_r2_mean*100).toFixed(1), XGB: +(metrics.xgboost.cv_r2_mean*100).toFixed(1) },
                { metric: "No Overfit",LR: +(100-(metrics.linear_regression.train.r2-metrics.linear_regression.test.r2)*1000).toFixed(1), RF: +(100-(metrics.random_forest.train.r2-metrics.random_forest.test.r2)*1000).toFixed(1), XGB: +(100-(metrics.xgboost.train.r2-metrics.xgboost.test.r2)*1000).toFixed(1) },
              ]}>
                <PolarGrid stroke="#1e293b" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <PolarRadiusAxis domain={[90, 100]} tick={{ fill: "#475569", fontSize: 10 }} />
                <Radar name="Linear Regression" dataKey="LR"  stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.15} />
                <Radar name="Random Forest"     dataKey="RF"  stroke="#34d399" fill="#34d399" fillOpacity={0.15} />
                <Radar name="XGBoost"           dataKey="XGB" stroke="#fbbf24" fill="#fbbf24" fillOpacity={0.15} />
                <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* ── Augmentation Effect ── */}
        {augData && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <SectionTitle>🧪 Effect of Synthetic Data Augmentation</SectionTitle>
            <p className="text-sm text-slate-400 mb-2">
              Does adding synthetic data improve predictions? We trained each model twice —
              once on <span className="text-emerald-400 font-medium">real data only</span> ({augData.splits.real_only_train.toLocaleString()} rows)
              and once on <span className="text-amber-400 font-medium">real + synthetic</span> ({augData.splits.augmented_train.toLocaleString()} rows) —
              then tested on a held-out <span className="text-sky-400 font-medium">real-only test set</span> ({augData.splits.test_set.toLocaleString()} rows).
            </p>
            <p className="text-xs text-red-400/80 mb-5">
              ⚠️ Augmentation reduced accuracy for all models — synthetic data introduces distribution mismatch.
            </p>

            {/* R² comparison chart */}
            <p className="text-sm text-slate-400 mb-3">R² Score — Real-Only vs Augmented Training</p>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart
                data={[
                  {
                    model: "Linear Reg",
                    "Real Only": +(augData.comparison.linear_regression.real_only.r2 * 100).toFixed(2),
                    "With Synthetic": +(augData.comparison.linear_regression.augmented.r2 * 100).toFixed(2),
                  },
                  {
                    model: "Random Forest",
                    "Real Only": +(augData.comparison.random_forest.real_only.r2 * 100).toFixed(2),
                    "With Synthetic": +(augData.comparison.random_forest.augmented.r2 * 100).toFixed(2),
                  },
                  {
                    model: "XGBoost",
                    "Real Only": +(augData.comparison.xgboost.real_only.r2 * 100).toFixed(2),
                    "With Synthetic": +(augData.comparison.xgboost.augmented.r2 * 100).toFixed(2),
                  },
                ]}
                margin={{ top: 4, right: 20, left: 10, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="model" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <YAxis domain={[96, 99]} tickFormatter={(v) => `${v}%`} tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                  formatter={(v) => [`${v}%`]}
                />
                <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
                <Bar dataKey="Real Only" fill="#34d399" radius={[4,4,0,0]} />
                <Bar dataKey="With Synthetic" fill="#f87171" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>

            {/* Detail cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-5">
              {(["linear_regression", "random_forest", "xgboost"] as const).map((m) => {
                const d = augData.comparison[m];
                return (
                  <div key={m} className={`border rounded-xl p-4 space-y-2 ${modelBg[m]}`}>
                    <p className={`text-xs font-semibold ${modelColors[m]}`}>{modelLabels[m]}</p>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400">R² (real only)</span>
                        <span className="font-mono text-emerald-400">{d.real_only.r2.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">R² (augmented)</span>
                        <span className="font-mono text-red-400">{d.augmented.r2.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between border-t border-slate-700 pt-1">
                        <span className="text-slate-400">R² change</span>
                        <span className="font-mono text-red-400">{d.r2_change > 0 ? "+" : ""}{(d.r2_change * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">MAE change</span>
                        <span className="font-mono text-red-400">+{d.mae_change.toFixed(2)} L/hr</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Footer ── */}
        <footer className="text-center text-xs text-slate-600 pb-6 space-y-1">
          <p>ShipFuel AI · ML models: Linear Regression, Random Forest, XGBoost</p>
          <p>Trained on 92,567 ship telemetry records · 3 features explain &gt;80% of variance</p>
        </footer>
      </div>
    </main>
  );
}
