import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowUpRight,
  Clock
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend
} from 'recharts';
import {
  api,
  type DashboardMetrics,
  type FunnelData,
  type DecisionDistribution,
  type RecentTransaction
} from '../api/client';

export const CommandCenterTab: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [funnel, setFunnel] = useState<FunnelData | null>(null);
  const [distribution, setDistribution] = useState<DecisionDistribution | null>(null);
  const [recentTxns, setRecentTxns] = useState<RecentTransaction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [m, f, d, r] = await Promise.all([
        api.getMetrics(),
        api.getFunnel(),
        api.getDecisionDistribution(),
        api.getRecentTransactions(10),
      ]);
      setMetrics(m);
      setFunnel(f);
      setDistribution(d);
      setRecentTxns(r.transactions);
    } catch (err: any) {
      setError(err.message || 'Failed to load command center data');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !metrics) {
    return (
      <div className="p-8 flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-3 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs font-semibold text-slate-500">Loading Recovery Command Center...</p>
        </div>
      </div>
    );
  }

  if (error && !metrics) {
    return (
      <div className="p-8">
        <div className="bg-amber-50 border border-amber-300 rounded-xl p-5 text-amber-900">
          <div className="flex items-center gap-2 font-bold text-sm">
            <AlertTriangle className="w-4 h-4 text-amber-600" />
            <span>Backend Connecting / Waking Up</span>
          </div>
          <p className="text-xs mt-1.5 text-amber-800 leading-relaxed">
            {error.includes('Failed to fetch') || error.includes('NetworkError') || error.includes('Load failed')
              ? 'Backend is waking up, please wait... (Render free instances sleep after inactivity and take approximately 30–60 seconds to spin up).'
              : error}
          </p>
          <button
            onClick={loadData}
            className="mt-3.5 px-3.5 py-1.5 bg-amber-600 text-white rounded-lg text-xs font-semibold hover:bg-amber-700 shadow-xs transition"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const chartData = metrics ? [
    {
      name: 'Default (τ=0.50)',
      Recovered: metrics.default_recovered,
      Cost: metrics.default_cost,
      NetProfit: metrics.default_net,
    },
    {
      name: 'RecoverAI EV (τ=0.35)',
      Recovered: metrics.opt_recovered,
      Cost: metrics.opt_cost,
      NetProfit: metrics.opt_net,
    },
  ] : [];

  return (
    <div className="p-4 sm:p-6 lg:p-8 space-y-6 sm:space-y-8 max-w-[1600px] mx-auto w-full min-w-0">
      {/* 1. TOP 5 KPI HERO SECTION */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            Canonical Financial Performance (Untouched Test Holdout N=633)
          </h2>
          <span className="text-[11px] text-slate-500 bg-slate-100 px-2 py-0.5 rounded font-medium">
            Seed 42 • Stratified Holdout
          </span>
        </div>

        <div className="kpi-responsive-grid">
          {/* Card 1: Total Risk */}
          <div className="kpi-col-top min-w-0 bg-white p-4 sm:p-5 rounded-xl border border-slate-200/90 shadow-xs hover:border-slate-300 transition-all flex flex-col justify-between">
            <div>
              <div className="min-h-[28px] sm:min-h-[30px] flex items-end">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide block leading-tight">
                  Total Revenue At Risk
                </span>
              </div>
              <div className="mt-2 text-xl sm:text-2xl min-[1360px]:text-[clamp(1.15rem,1.4vw,1.5rem)] font-black text-slate-900 tracking-tight tabular-nums whitespace-nowrap overflow-visible">
                {metrics?.formatted_total_risk ?? '—'}
              </div>
            </div>
            <div className="mt-2.5 flex items-center gap-1.5 text-xs text-slate-500">
              <span className="font-semibold text-slate-700">{metrics?.total_failed_count ?? '—'}</span> failed payments
            </div>
          </div>

          {/* Card 2: Default Recovered */}
          <div className="kpi-col-top min-w-0 bg-white p-4 sm:p-5 rounded-xl border border-slate-200/90 shadow-xs hover:border-slate-300 transition-all flex flex-col justify-between">
            <div>
              <div className="min-h-[28px] sm:min-h-[30px] flex items-end">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide block leading-tight">
                  Default Recovered (τ=0.50)
                </span>
              </div>
              <div className="mt-2 text-xl sm:text-2xl min-[1360px]:text-[clamp(1.15rem,1.4vw,1.5rem)] font-black text-slate-900 tracking-tight tabular-nums whitespace-nowrap overflow-visible">
                {metrics?.formatted_default_recovered ?? '—'}
              </div>
            </div>
            <div className="mt-2.5 flex items-center gap-1 text-xs text-slate-500">
              <span>Standard industry rule</span>
            </div>
          </div>

          {/* Card 3: EV Recovered */}
          <div className="kpi-col-top min-w-0 bg-white p-4 sm:p-5 rounded-xl border border-blue-200 shadow-xs bg-blue-50/20 hover:border-blue-300 transition-all flex flex-col justify-between">
            <div>
              <div className="min-h-[28px] sm:min-h-[30px] flex items-end">
                <span className="text-[11px] font-bold text-blue-700 uppercase tracking-wide block leading-tight">
                  EV Recovered (τ=0.35)
                </span>
              </div>
              <div className="mt-2 text-xl sm:text-2xl min-[1360px]:text-[clamp(1.15rem,1.4vw,1.5rem)] font-black text-blue-700 tracking-tight tabular-nums whitespace-nowrap overflow-visible">
                {metrics?.formatted_opt_recovered ?? '—'}
              </div>
            </div>
            <div className="mt-2.5 flex items-center gap-1 text-xs text-blue-700 font-semibold">
              <ArrowUpRight className="w-3.5 h-3.5 shrink-0" />
              <span>{metrics ? `${metrics.recovery_rate_pct.toFixed(2)}% Total Risk Captured` : '—'}</span>
            </div>
          </div>

          {/* Card 4: True Net Uplift */}
          <div className="kpi-col-bottom min-w-0 bg-white p-4 sm:p-5 rounded-xl border border-emerald-200 shadow-xs bg-emerald-50/30 hover:border-emerald-300 transition-all flex flex-col justify-between">
            <div>
              <div className="min-h-[28px] sm:min-h-[30px] flex items-end">
                <span className="text-[11px] font-bold text-emerald-800 uppercase tracking-wide block leading-tight">
                  True Net Uplift (Net-of-Costs)
                </span>
              </div>
              <div className="mt-2 text-xl sm:text-2xl min-[1360px]:text-[clamp(1.15rem,1.4vw,1.5rem)] font-black text-emerald-700 tracking-tight tabular-nums whitespace-nowrap overflow-visible">
                {metrics?.formatted_net_uplift_inr ?? '—'}
              </div>
            </div>
            <div className="mt-2.5 flex items-center gap-1 text-xs font-bold text-emerald-800">
              <span>{metrics?.formatted_net_uplift_pct ? `${metrics.formatted_net_uplift_pct} pure net profit` : '—'}</span>
            </div>
          </div>

          {/* Card 5: Pending Reviews */}
          <div className="kpi-col-bottom min-w-0 bg-white p-4 sm:p-5 rounded-xl border border-slate-200/90 shadow-xs hover:border-slate-300 transition-all flex flex-col justify-between">
            <div>
              <div className="min-h-[28px] sm:min-h-[30px] flex items-end">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide block leading-tight">
                  Pending Escalations
                </span>
              </div>
              <div className="mt-2 text-xl sm:text-2xl min-[1360px]:text-[clamp(1.15rem,1.4vw,1.5rem)] font-black text-slate-900 tracking-tight tabular-nums whitespace-nowrap overflow-visible">
                {metrics?.pending_review_count ?? 0}
              </div>
            </div>
            <div className="mt-2.5 flex items-center gap-1.5 text-xs text-amber-700 font-semibold">
              <Clock className="w-3.5 h-3.5 shrink-0" />
              <span>Human Approval Queue</span>
            </div>
          </div>
        </div>
      </section>

      {/* 2. ECONOMIC COMPARISON CHART & CONVERSION FUNNEL */}
      <section className="dashboard-middle-grid">
        {/* Comparison Economics Chart */}
        <div className="bg-white p-5 sm:p-6 rounded-xl border border-slate-200 shadow-xs flex flex-col justify-between min-w-0 overflow-hidden">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-bold text-slate-900 text-sm">Policy Economics Comparison</h3>
              <span className="text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                +₹30,774.87 Net
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Comparing industry standard (τ=0.50) vs RecoverAI EV-optimized (τ=0.35)
            </p>

            <div className="h-60 w-full min-w-0 overflow-hidden relative">
              <ResponsiveContainer width="99%" height={240} debounce={50}>
                <BarChart data={chartData} margin={{ top: 10, right: 25, left: -5, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748B' }} interval={0} />
                  <YAxis tick={{ fontSize: 11, fill: '#64748B' }} tickFormatter={(val) => `₹${(val / 1000).toFixed(0)}k`} />
                  <Tooltip
                    formatter={(val: any) => [`₹${Number(val).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`, '']}
                    contentStyle={{ borderRadius: 8, fontSize: 12, borderColor: '#CBD5E1' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, bottom: 0 }} />
                  <Bar dataKey="Recovered" fill="#2563EB" name="Recovered Revenue" radius={[4, 4, 0, 0]} maxBarSize={48} isAnimationActive={false} />
                  <Bar dataKey="NetProfit" fill="#16A34A" name="Net Realized Value" radius={[4, 4, 0, 0]} maxBarSize={48} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-600">
            <span>Incremental Action Cost: <b>{metrics ? `+₹${metrics.inc_action_cost.toFixed(2)}` : '—'}</b></span>
            <span>Marginal ROI: <b className="text-emerald-700">{metrics && metrics.inc_action_cost > 0 ? `${Math.round(metrics.net_uplift_inr / metrics.inc_action_cost)}x` : '—'}</b></span>
          </div>
        </div>

        {/* 5-Step End-to-End Recovery Funnel */}
        <div className="bg-white p-5 sm:p-6 rounded-xl border border-slate-200 shadow-xs flex flex-col justify-between min-w-0">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-bold text-slate-900 text-sm">End-to-End Recovery Funnel</h3>
              <span className="text-[11px] text-slate-500 font-medium">
                Holdout N = {funnel?.total_failed ?? '—'}
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Conversion progression from failure ingestion to settled funds
            </p>

            <div className="funnel-responsive-grid items-stretch">
              {funnel?.steps.map((s) => {
                let cardStyle = 'border-slate-200 bg-white';
                let valColor = 'text-slate-900';
                let pctColor = 'text-slate-500';

                if (s.step === 3) {
                  cardStyle = 'border-blue-200 bg-blue-50/50';
                  valColor = 'text-blue-700';
                  pctColor = 'text-blue-700';
                } else if (s.step === 4) {
                  valColor = 'text-slate-900';
                  pctColor = 'text-slate-500';
                } else if (s.step === 5) {
                  cardStyle = 'border-emerald-200 bg-emerald-50/50';
                  valColor = 'text-emerald-700';
                  pctColor = 'text-emerald-700';
                }

                return (
                  <div
                    key={s.step}
                    className={`p-2.5 sm:p-3 rounded-lg border ${cardStyle} text-center flex flex-col justify-between h-full min-w-0 shadow-2xs transition-all`}
                  >
                    <div className="min-h-[38px] flex items-center justify-center text-[10px] sm:text-[11px] font-bold text-slate-600 uppercase tracking-wider leading-snug text-center break-words px-0.5">
                      {s.step}. {s.name}
                    </div>
                    <div className={`text-xl sm:text-2xl font-black my-1.5 tabular-nums ${valColor}`}>
                      {s.count}
                    </div>
                    <div className={`text-[10px] sm:text-[11px] font-bold ${pctColor} leading-tight`}>
                      {s.badge}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-5 p-3 rounded-lg bg-slate-50 border border-slate-200 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div>
              <span className="font-bold text-red-600">
                {funnel ? `${funnel.summary.blocked_count} Blocked` : '—'}
              </span>
              <span className="text-slate-500 ml-1">(Negative EV / Fraud Risk)</span>
            </div>
            <div>
              <span className="font-bold text-amber-600">
                {funnel ? `${funnel.summary.escalated_count} Escalated` : '—'}
              </span>
              <span className="text-slate-500 ml-1">(High-Value HITL ₹8,500 or more)</span>
            </div>
            <div>
              <span className="font-bold text-emerald-700">
                {funnel ? `₹${Math.round(funnel.summary.recovered_revenue).toLocaleString('en-IN')} Recovered` : '—'}
              </span>
              <span className="text-slate-500 ml-1">
                {metrics ? `(${metrics.recovery_rate_pct.toFixed(2)}% Captured)` : ''}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* 3. POLICY DECISION DISTRIBUTION & FAILURE CATEGORY INSIGHTS */}
      <section className="dashboard-bottom-grid">
        {/* Decision Breakdown */}
        <div className="bg-white p-5 sm:p-6 rounded-xl border border-slate-200 shadow-xs min-w-0">
          <h3 className="font-bold text-slate-900 text-sm mb-1">
            Policy Decision Distribution (Test Set N=633)
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Deterministic governance breakdown preventing wasteful attempts
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {distribution?.decisions.map((d) => (
              <div
                key={d.key}
                className="p-3.5 rounded-lg border border-slate-200 bg-white"
                style={{ borderLeftWidth: 4, borderLeftColor: d.color }}
              >
                <div className="text-[11px] font-bold uppercase tracking-wider truncate" style={{ color: d.color }}>
                  {d.name}
                </div>
                <div className="text-xl font-black text-slate-900 mt-1">
                  {d.count}{' '}
                  <span className="text-xs font-semibold text-slate-500">
                    ({d.percentage.toFixed(1)}%)
                  </span>
                </div>
                <div className="text-[11px] text-slate-500 mt-1 line-clamp-1">
                  {d.description}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Failure Categories Breakdown */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs">
          <h3 className="font-bold text-slate-900 text-sm mb-1">
            Failure Category Distribution &amp; Recoverability
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Historical payment failure drivers from `transactions.csv`
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
                <tr>
                  <th className="py-2.5 px-3">Failure Category</th>
                  <th className="py-2.5 px-3">Share</th>
                  <th className="py-2.5 px-3">Recoverability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {distribution?.failure_categories.map((c, idx) => (
                  <tr key={idx} className="hover:bg-slate-50/60 transition-colors">
                    <td className="py-2.5 px-3 font-semibold text-slate-800">{c.category}</td>
                    <td className="py-2.5 px-3 text-slate-600 font-medium">{c.share_pct}%</td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        c.recoverability.includes('Very High')
                          ? 'bg-emerald-100 text-emerald-800'
                          : c.recoverability.includes('High')
                          ? 'bg-blue-100 text-blue-800'
                          : c.recoverability.includes('Blocked')
                          ? 'bg-red-100 text-red-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}>
                        {c.recoverability}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* 4. RECENT ACTIVITY STREAM (DATABASE PERSISTED) */}
      <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-bold text-slate-900 text-sm">
              Live Relational Database Activity Feed
            </h3>
            <p className="text-xs text-slate-500">
              Real-time payment records stored in SQLite / PostgreSQL
            </p>
          </div>
          <span className="text-xs font-semibold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md">
            Latest {recentTxns.length} Records
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3">Transaction ID</th>
                <th className="py-2.5 px-3">Customer</th>
                <th className="py-2.5 px-3">Amount</th>
                <th className="py-2.5 px-3">Failure Reason</th>
                <th className="py-2.5 px-3">P(Recovery)</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {recentTxns.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-6 text-center text-slate-400">
                    No active transactions found in database.
                  </td>
                </tr>
              ) : (
                recentTxns.map((t) => (
                  <tr key={t.id} className="hover:bg-slate-50/60 transition-colors">
                    <td className="py-2.5 px-3 font-mono font-bold text-blue-600">{t.transaction_id}</td>
                    <td className="py-2.5 px-3 font-mono text-slate-500">{t.customer_id}</td>
                    <td className="py-2.5 px-3 font-bold text-slate-800">
                      ₹{t.amount.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </td>
                    <td className="py-2.5 px-3 text-slate-600 capitalize">
                      {t.failure_reason.replace(/_/g, ' ')}
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-slate-700">
                      {t.recovery_probability !== null && t.recovery_probability !== undefined
                        ? `${(t.recovery_probability * 100).toFixed(1)}%`
                        : '—'}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        t.status === 'RECOVERED'
                          ? 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                          : t.status === 'AWAITING_APPROVAL'
                          ? 'bg-amber-100 text-amber-800 border border-amber-200'
                          : t.status === 'BLOCKED'
                          ? 'bg-red-100 text-red-800 border border-red-200'
                          : 'bg-slate-100 text-slate-700 border border-slate-200'
                      }`}>
                        {t.status}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px]">
                      {t.created_at ? new Date(t.created_at).toLocaleTimeString() : '—'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
