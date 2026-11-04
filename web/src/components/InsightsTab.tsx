import React, { useEffect, useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Info
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid
} from 'recharts';
import {
  api,
  type CalibrationData,
  type DriftReport,
  type SensitivityData,
  type AuditLogData
} from '../api/client';

export const InsightsTab: React.FC = () => {
  const [calibration, setCalibration] = useState<CalibrationData | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [sensitivity, setSensitivity] = useState<SensitivityData | null>(null);
  const [auditLog, setAuditLog] = useState<AuditLogData | null>(null);
  const [decisionFilter, setDecisionFilter] = useState<string>('ALL');
  const [expandedRecord, setExpandedRecord] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    loadAuditTrail();
  }, [decisionFilter]);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [cal, drf, sens, aud] = await Promise.all([
        api.getCalibration(),
        api.getDriftReport(),
        api.getSensitivity(),
        api.getAuditLog(decisionFilter, 25),
      ]);
      setCalibration(cal);
      setDrift(drf);
      setSensitivity(sens);
      setAuditLog(aud);
    } catch (err: any) {
      setError(err.message || 'Failed to load insights data');
    } finally {
      setLoading(false);
    }
  };

  const loadAuditTrail = async () => {
    try {
      const aud = await api.getAuditLog(decisionFilter, 25);
      setAuditLog(aud);
    } catch (err) {
      console.error(err);
    }
  };

  const toggleRecord = (txnId: string) => {
    setExpandedRecord(expandedRecord === txnId ? null : txnId);
  };

  if (loading && !calibration) {
    return (
      <div className="p-8 flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-3 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs font-semibold text-slate-500">Loading Governance &amp; Insights...</p>
        </div>
      </div>
    );
  }

  const gridData = calibration?.development_grid_search?.map((pt) => ({
    threshold: pt.threshold,
    expectedNet: pt.expected_net_value,
    realizedRevenue: pt.realized_recovered_revenue,
  })) || [];

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}
      {/* 1. THRESHOLD COMPARISON & GRID SEARCH CURVE */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Table */}
        <div className="lg:col-span-6 bg-white p-6 rounded-xl border border-slate-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-bold text-slate-900 text-sm">Policy Threshold Calibration Comparison</h3>
              <span className="text-[11px] font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                Optimal τ = 0.35
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Comparing industry standard (τ=0.50) vs RecoverAI EV-optimized (τ=0.35) on untouched test holdout
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3">Policy Policy</th>
                    <th className="py-2.5 px-3">Recovered</th>
                    <th className="py-2.5 px-3">Action Cost</th>
                    <th className="py-2.5 px-3">Net Profit</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {calibration?.threshold_comparison.map((row, idx) => {
                    const isUplift = idx === 2;
                    return (
                      <tr key={idx} className={isUplift ? 'bg-emerald-50/40 font-bold' : ''}>
                        <td className="py-2.5 px-3 font-semibold text-slate-800">{row.policy}</td>
                        <td className={`py-2.5 px-3 ${isUplift ? 'text-emerald-700' : 'text-slate-700'}`}>
                          {row.formatted_recovered_revenue}
                        </td>
                        <td className="py-2.5 px-3 text-slate-600">
                          {row.formatted_action_cost}
                        </td>
                        <td className={`py-2.5 px-3 ${isUplift ? 'text-emerald-700 font-black' : 'text-slate-900 font-bold'}`}>
                          {row.formatted_net_profit}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500">
            *Expected Value Optimization mathematically proves that threshold 0.35 captures recoverable revenue without overspending on action costs.
          </div>
        </div>

        {/* Expected Net Value vs Threshold Curve */}
        <div className="lg:col-span-6 bg-white p-6 rounded-xl border border-slate-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-bold text-slate-900 text-sm">Expected Net Value Calibration Curve</h3>
              <span className="text-[11px] font-semibold text-slate-500">Grid τ: 0.10 &rarr; 0.90</span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Search curve illustrating profit peak at optimal policy threshold
            </p>

            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={gridData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                  <XAxis
                    dataKey="threshold"
                    tick={{ fontSize: 11, fill: '#64748B' }}
                    label={{ value: 'Threshold (τ)', position: 'insideBottomRight', offset: -5, fontSize: 10 }}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: '#64748B' }}
                    tickFormatter={(val) => `₹${(val / 100000).toFixed(1)}L`}
                  />
                  <Tooltip
                    formatter={(val: any) => [`₹${Number(val).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`, 'Expected Net']}
                    contentStyle={{ borderRadius: 8, fontSize: 12, borderColor: '#CBD5E1' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="expectedNet"
                    stroke="#2563EB"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#2563EB' }}
                    activeDot={{ r: 5 }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-600">
            <span>Threshold Range: <b>0.10 to 0.90</b></span>
            <span>Policy Setting: <b className="text-blue-700">τ = 0.35</b></span>
          </div>
        </div>
      </section>

      {/* 2. PSI FEATURE DRIFT MONITOR */}
      <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-bold text-slate-900 text-sm">
                Population Stability Index (PSI) Drift Monitor
              </h3>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                drift?.system_status.includes('HEALTHY')
                  ? 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                  : 'bg-amber-100 text-amber-800 border border-amber-200'
              }`}>
                {drift?.system_status}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Continuously measures distribution divergence between training baseline and incoming transactions
            </p>
          </div>

          <div className="text-xs font-semibold text-slate-600">
            Significant Drift: <b className="text-slate-900">{drift?.significant_drift_count ?? 0}</b> | Moderate Drift: <b className="text-slate-900">{drift?.moderate_drift_count ?? 0}</b>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3">Feature</th>
                <th className="py-2.5 px-3">Type</th>
                <th className="py-2.5 px-3">PSI Score</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3">Operational Recommendation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {drift?.features.map((f, idx) => (
                <tr key={idx} className="hover:bg-slate-50/50 transition-colors">
                  <td className="py-2.5 px-3 font-semibold text-slate-800 font-mono">{f.feature}</td>
                  <td className="py-2.5 px-3 text-slate-500 capitalize">{f.type}</td>
                  <td className="py-2.5 px-3 font-bold text-slate-800">{f.psi.toFixed(4)}</td>
                  <td className="py-2.5 px-3">
                    <span className="bg-emerald-50 text-emerald-700 font-bold text-[10px] px-2 py-0.5 rounded border border-emerald-200">
                      {f.status}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-600 font-medium">{f.recommendation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 3. 6-SCENARIO SENSITIVITY GRID */}
      <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <div>
            <h3 className="font-bold text-slate-900 text-sm">
              6-Scenario Financial Sensitivity Analysis Grid
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Verified stress tests varying action costs (±50%), threshold shifts, and predicted probability miscalibration (-0.10)
            </p>
          </div>
          <span className="text-xs font-semibold text-slate-500 bg-slate-100 px-2.5 py-1 rounded">
            Source: `evaluation/sensitivity_results.csv`
          </span>
        </div>

        <div className="overflow-x-auto mt-4">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3">Scenario</th>
                <th className="py-2.5 px-3">Total Risk</th>
                <th className="py-2.5 px-3">Default Net</th>
                <th className="py-2.5 px-3">EV Realized Net Value (not causal)</th>
                <th className="py-2.5 px-3">Gross Uplift</th>
                <th className="py-2.5 px-3">True Net Uplift (₹)</th>
                <th className="py-2.5 px-3">Uplift (%)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {sensitivity?.scenarios.map((sc, idx) => (
                <tr key={idx} className="hover:bg-slate-50/50 transition-colors">
                  <td className="py-2.5 px-3 font-semibold text-slate-900">{sc.Scenario}</td>
                  <td className="py-2.5 px-3 text-slate-500">{sc['Total Revenue at Risk']}</td>
                  <td className="py-2.5 px-3 text-slate-600">{sc['Default Net Value']}</td>
                  <td className="py-2.5 px-3 font-bold text-slate-800">{sc['EV Realized Net Value (not causal)']}</td>
                  <td className="py-2.5 px-3 text-emerald-700 font-semibold">{sc['Gross Uplift']}</td>
                  <td className="py-2.5 px-3 text-emerald-700 font-black">{sc['True Net Uplift (₹)']}</td>
                  <td className="py-2.5 px-3 text-emerald-700 font-black">{sc['True Net Uplift (%)']}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1">
          <div className="flex items-start gap-1.5">
            <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
            <p>
              <b>Accounting Note:</b> Realized net value of the selected payments stays between ₹914,150.49 and ₹918,659.94 across all scenarios (this is realized value, not causal lift; the causal estimate is reported separately in the Scientific Limitations).
            </p>
          </div>
          <div className="flex items-start gap-1.5">
            <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
            <p>
              <b>Scenario 6 Interpretability:</b> Scenario 6 uplift rises because the default policy (tau=0.50) loses more payments than the EV policy under a downward shift; it does not mean a worse model is better.
            </p>
          </div>
        </div>
      </section>

      {/* 4. IMMUTABLE DECISION AUDIT LOG */}
      <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h3 className="font-bold text-slate-900 text-sm">
              Immutable Governance Audit Trail
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Decisions captured with full feature contributions and policy justifications (`logs/decision_audit.jsonl`)
            </p>
          </div>

          {/* Filter Pills */}
          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg">
            {['ALL', 'ACT', 'ESCALATE', 'REFUSE'].map((f) => (
              <button
                key={f}
                onClick={() => setDecisionFilter(f)}
                className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${
                  decisionFilter === f
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {/* Audit Records List */}
        <div className="space-y-3">
          {auditLog?.records.length === 0 ? (
            <div className="p-8 text-center text-slate-400 text-xs">
              No audit records matching filter: {decisionFilter}
            </div>
          ) : (
            auditLog?.records.map((rec) => {
              const isExpanded = expandedRecord === rec.transaction_id;
              const dec = rec.safety?.decision || 'REFUSE';

              let borderColor = 'border-l-emerald-600';
              let badgeBg = 'bg-emerald-100 text-emerald-800';
              if (dec === 'ESCALATE') {
                borderColor = 'border-l-amber-500';
                badgeBg = 'bg-amber-100 text-amber-800';
              } else if (dec === 'REFUSE') {
                borderColor = 'border-l-red-600';
                badgeBg = 'bg-red-100 text-red-800';
              }

              return (
                <div
                  key={rec.transaction_id}
                  className={`border border-slate-200 rounded-lg overflow-hidden border-l-4 ${borderColor} bg-white transition-all`}
                >
                  <div
                    onClick={() => toggleRecord(rec.transaction_id)}
                    className="p-4 cursor-pointer hover:bg-slate-50/50 flex flex-wrap items-center justify-between gap-3 text-xs"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-mono font-bold text-slate-900">
                        {rec.transaction_id}
                      </span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${badgeBg}`}>
                        {dec}
                      </span>
                      <span className="text-slate-400">
                        {rec.timestamp}
                      </span>
                    </div>

                    <div className="flex items-center gap-4 text-slate-600">
                      <span>Amount: <b className="text-slate-800">₹{rec.amount?.toLocaleString('en-IN') || 0}</b></span>
                      <span>P(Rec): <b className="text-blue-700">{((rec.prediction?.recovery_probability || 0) * 100).toFixed(1)}%</b></span>
                      <div className="flex items-center gap-1 text-slate-400">
                        <span>{isExpanded ? 'Collapse' : 'Inspect'}</span>
                        {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                      </div>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="p-4 bg-slate-50/70 border-t border-slate-100 space-y-3 text-xs">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div className="bg-white p-3 rounded-lg border border-slate-200">
                          <span className="font-bold text-slate-700 block mb-1">Safety Justification</span>
                          <p className="text-slate-600 text-xs leading-relaxed">
                            {rec.safety?.justification || 'Evaluated against policy rules.'}
                          </p>
                          {rec.safety?.triggered_rules && rec.safety.triggered_rules.length > 0 && (
                            <div className="mt-2 text-[11px] text-red-700 font-mono">
                              Triggered: {rec.safety.triggered_rules.join(', ')}
                            </div>
                          )}
                        </div>

                        <div className="bg-white p-3 rounded-lg border border-slate-200">
                          <span className="font-bold text-slate-700 block mb-1">Model Explanation</span>
                          <p className="text-slate-600 text-xs leading-relaxed">
                            {rec.prediction?.explanation || 'Prediction generated via logistic regression.'}
                          </p>
                        </div>
                      </div>

                      <div className="bg-slate-900 p-3 rounded-lg text-slate-100 font-mono text-[11px] overflow-x-auto">
                        <pre>{JSON.stringify(rec, null, 2)}</pre>
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </section>
    </div>
  );
};
