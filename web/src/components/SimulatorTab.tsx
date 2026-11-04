import React, { useState } from 'react';
import {
  Play,
  RotateCcw,

  AlertTriangle,

  ChevronDown,
  ChevronRight,
  ArrowRight

} from 'lucide-react';
import { api, type DemoScenario, type SimulationResult } from '../api/client';

interface SimulatorTabProps {
  scenarios: DemoScenario[];
  selectedScenarioId: string;
  setSelectedScenarioId: (id: string) => void;
  simulationResult: SimulationResult | null;
  setSimulationResult: (res: SimulationResult | null) => void;
  onNavigateToTrace?: () => void;
}

export const SimulatorTab: React.FC<SimulatorTabProps> = ({
  scenarios,
  selectedScenarioId,
  setSelectedScenarioId,
  simulationResult,
  setSimulationResult,
  onNavigateToTrace,
}) => {
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  const fallbackScenarios: DemoScenario[] = [
    {
      id: 'auto_recovery',
      name: 'Scenario A — Smart Auto Recovery',
      description: 'Transient network timeout on a valued customer. Agent analyzes, ML predicts high recovery probability, Policy ACT approves, and recovery action executes automatically.',
      expected_flow: 'Analysis -> High ML Probability -> LLM Recommendation -> Policy ACT -> Auto Execution -> Verification',
      expected_decision: 'ACT',
      expected_action: 'retry',
      amount: 2500.0,
      failure_reason: 'network_timeout',
      failure_category: 'transient',
    },
    {
      id: 'human_approval',
      name: 'Scenario B — High-Value Human Approval',
      description: 'Transaction amount (₹14,500) exceeds safety threshold (₹8,500.00). Policy halts automated execution and routes directly into the Merchant Approval Queue.',
      expected_flow: 'Analysis -> LLM Recommendation -> Policy ESCALATE -> Enters Approval Queue -> Merchant Decision',
      expected_decision: 'ESCALATE',
      expected_action: 'retry',
      amount: 14500.0,
      failure_reason: 'network_timeout',
      failure_category: 'transient',
    },
    {
      id: 'fraud_block',
      name: 'Scenario C — Fraud & Risk Policy Guard',
      description: 'Transaction flagged with IP risk score (0.85 > 0.70 threshold). Deterministic hard safety rule triggers immediate REFUSE, preventing chargeback risk.',
      expected_flow: 'Analysis -> Risk Flagged -> Policy REFUSE (Hard Safety) -> Execution Blocked -> Telemetry Audit',
      expected_decision: 'REFUSE',
      expected_action: 'none',
      amount: 3200.0,
      failure_reason: 'suspected_risk',
      failure_category: 'risk_related',
    },
    {
      id: 'low_probability',
      name: 'Scenario D — Low Recovery Probability',
      description: 'Repeated card decline with low predicted recovery probability (0.18 < 0.35 threshold). Expected Value optimization halts action to save merchant fees.',
      expected_flow: 'Analysis -> Low ML Probability (0.18) -> Policy REFUSE (Low EV) -> Action Saved -> Cost Avoided',
      expected_decision: 'REFUSE',
      expected_action: 'none',
      amount: 1800.0,
      failure_reason: 'invalid_card',
      failure_category: 'hard_decline',
    },
  ];

  const displayScenarios = scenarios.length > 0 ? scenarios : fallbackScenarios;
  const currentScenario = displayScenarios.find((s) => s.id === selectedScenarioId) || displayScenarios[0];

  const handleSimulate = async () => {
    setIsRunning(true);
    setError(null);
    try {
      const res = await api.simulateScenario(selectedScenarioId);
      setSimulationResult(res);
      // Automatically expand step 5 (Policy Decision)
      setExpandedStep(5);
    } catch (err: any) {
      setError(err.message || 'Simulation execution failed');
    } finally {
      setIsRunning(false);
    }
  };

  const handleReset = () => {
    setSimulationResult(null);
    setError(null);
    setExpandedStep(null);
  };

  const toggleStep = (stepNumber: number) => {
    setExpandedStep(expandedStep === stepNumber ? null : stepNumber);
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* 1. SCENARIO SELECTOR CARDS A-D */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            Select Buildathon Demo Scenario (Scenarios A - D)
          </h2>
          <span className="text-xs text-slate-500 font-medium">
            Pre-configured deterministic boundary tests
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {displayScenarios.map((sc) => {
            const isSelected = sc.id === selectedScenarioId;
            const isAct = sc.expected_decision === 'ACT';
            const isEscalate = sc.expected_decision === 'ESCALATE';

            return (
              <div
                key={sc.id}
                onClick={() => setSelectedScenarioId(sc.id)}
                className={`cursor-pointer p-4 rounded-xl border transition-all text-left flex flex-col justify-between ${
                  isSelected
                    ? 'border-blue-600 bg-blue-50/40 ring-2 ring-blue-600/20 shadow-sm'
                    : 'border-slate-200 bg-white hover:border-slate-300 hover:shadow-xs'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-1.5">
                      <span className={`w-2 h-2 rounded-full ${
                        isAct ? 'bg-emerald-500' : isEscalate ? 'bg-amber-500' : 'bg-red-500'
                      }`} />
                      <span className="font-bold text-xs text-slate-800">
                        {sc.id === 'auto_recovery'
                          ? 'Scenario A'
                          : sc.id === 'human_approval'
                          ? 'Scenario B'
                          : sc.id === 'fraud_block'
                          ? 'Scenario C'
                          : 'Scenario D'}
                      </span>
                    </div>

                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                      isAct
                        ? 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                        : isEscalate
                        ? 'bg-amber-100 text-amber-800 border border-amber-200'
                        : 'bg-red-100 text-red-800 border border-red-200'
                    }`}>
                      {sc.expected_decision}
                    </span>
                  </div>

                  <h4 className="font-bold text-sm text-slate-900 line-clamp-1">
                    {sc.name.split('—')[1]?.trim() || sc.name}
                  </h4>
                  <p className="text-xs text-slate-500 mt-1 line-clamp-3 leading-relaxed">
                    {sc.description}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-600 font-medium">
                  <span>Amount: <b>₹{sc.amount.toLocaleString('en-IN')}</b></span>
                  <span className="capitalize">{sc.failure_reason.replace(/_/g, ' ')}</span>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* 2. ACTION CONTROLS & TRIGGER */}
      <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold text-slate-900">
              Active Scenario: {currentScenario.name}
            </span>
            <span className="text-xs text-slate-500">
              (Expected: <b className="text-slate-800">{currentScenario.expected_decision}</b>)
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Triggers the full multi-stage LangGraph autonomous recovery lifecycle in dry-run test mode.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleReset}
            disabled={isRunning || !simulationResult}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-600 bg-white hover:bg-slate-50 hover:text-slate-900 disabled:opacity-40 transition-all"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset / Replay</span>
          </button>

          <button
            onClick={handleSimulate}
            disabled={isRunning}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-blue-600 text-white text-xs font-bold hover:bg-blue-700 disabled:opacity-60 shadow-sm shadow-blue-600/20 transition-all"
          >
            {isRunning ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Running Agent Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-white" />
                <span>Simulate Payment Failure &amp; Run Agent</span>
              </>
            )}
          </button>
        </div>
      </section>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-xs text-red-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="font-bold text-red-600 hover:text-red-800">
            Dismiss
          </button>
        </div>
      )}

      {/* 3. SIMULATION RESULT HERO & 7-STAGE PIPELINE TIMELINE */}
      {simulationResult && (
        <section className="space-y-6">
          {/* Decision Outcome Card */}
          <div className={`p-6 rounded-xl border ${
            simulationResult.policy_decision === 'ACT'
              ? 'bg-emerald-50/40 border-emerald-200'
              : simulationResult.policy_decision === 'ESCALATE'
              ? 'bg-amber-50/40 border-amber-200'
              : 'bg-red-50/40 border-red-200'
          }`}>
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-3">
                  <span className={`text-sm font-black px-3 py-1 rounded-md tracking-wider ${
                    simulationResult.policy_decision === 'ACT'
                      ? 'bg-emerald-600 text-white'
                      : simulationResult.policy_decision === 'ESCALATE'
                      ? 'bg-amber-500 text-white'
                      : 'bg-red-600 text-white'
                  }`}>
                    POLICY {simulationResult.policy_decision}
                  </span>
                  <span className="text-xs font-bold text-slate-700">
                    Transaction ID: <code className="bg-white px-2 py-0.5 rounded border border-slate-200">{simulationResult.transaction_id}</code>
                  </span>
                </div>
                <p className="text-xs font-medium text-slate-700 mt-2.5 max-w-3xl leading-relaxed">
                  {simulationResult.policy_reason}
                </p>
              </div>

              {onNavigateToTrace && (
                <button
                  onClick={onNavigateToTrace}
                  className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-white border border-slate-200 text-xs font-bold text-blue-600 hover:border-blue-300 hover:bg-blue-50/50 shadow-xs"
                >
                  <span>Inspect Decision Trace</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Quick Metrics Strip */}
            <div className="mt-4 pt-4 border-t border-slate-200/60 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <span className="text-slate-500 block text-[11px]">Amount</span>
                <span className="font-bold text-slate-800">
                  ₹{simulationResult.amount.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[11px]">P(Recovery)</span>
                <span className="font-bold text-blue-700">
                  {(simulationResult.recovery_probability * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[11px]">Action Selected</span>
                <span className="font-bold text-slate-800 uppercase font-mono">
                  {simulationResult.selected_action}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[11px]">Agent Status</span>
                <span className="font-bold text-slate-800">
                  {simulationResult.agent_status}
                </span>
              </div>
            </div>
          </div>

          {/* 7-Stage Pipeline Timeline */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-bold text-slate-900 text-sm">
                  7-Stage Autonomous Decision &amp; Execution Pipeline
                </h3>
                <p className="text-xs text-slate-500">
                  Trace each deterministic stage from failure detection through settlement
                </p>
              </div>
              <span className="text-xs font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                7 Stages Verified
              </span>
            </div>

            <div className="space-y-3">
              {simulationResult.timeline_steps.map((step) => {
                const isExpanded = expandedStep === step.step_number;
                const isDecisionStage = step.step_number === 5;
                const isExecutionStage = step.step_number === 6;

                let borderLeft = 'border-l-blue-600';
                if (isDecisionStage || isExecutionStage) {
                  if (simulationResult.policy_decision === 'ACT') borderLeft = 'border-l-emerald-600';
                  else if (simulationResult.policy_decision === 'ESCALATE') borderLeft = 'border-l-amber-500';
                  else borderLeft = 'border-l-red-600';
                }

                return (
                  <div
                    key={step.step_number}
                    className={`rounded-lg border border-slate-200/90 border-l-4 ${borderLeft} bg-white transition-all`}
                  >
                    <div
                      onClick={() => toggleStep(step.step_number)}
                      className="p-4 cursor-pointer flex items-center justify-between gap-3 hover:bg-slate-50/50"
                    >
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 rounded-full bg-slate-100 text-slate-700 font-bold text-xs flex items-center justify-center shrink-0">
                          {step.step_number}
                        </span>
                        <div>
                          <h4 className="text-xs font-bold text-slate-900">
                            {step.title}
                          </h4>
                          <p className="text-xs text-slate-600 mt-0.5">
                            {step.summary}
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-semibold text-slate-400">
                          {isExpanded ? 'Hide Details' : 'View Details'}
                        </span>
                        {isExpanded ? (
                          <ChevronDown className="w-4 h-4 text-slate-400" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-slate-400" />
                        )}
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="p-4 bg-slate-50/80 border-t border-slate-100 text-xs font-mono">
                        <div className="bg-white p-3 rounded border border-slate-200 overflow-x-auto text-slate-800">
                          <pre>{JSON.stringify(step.details, null, 2)}</pre>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
