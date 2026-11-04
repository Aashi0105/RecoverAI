import React, { useState } from 'react';
import {
  Cpu,
  ArrowRight,
  Zap,
  Check
} from 'lucide-react';
import { api, type SimulationResult } from '../api/client';

interface TraceTabProps {
  simulationResult: SimulationResult | null;
  onNavigateToSimulator: () => void;
  onSimulationUpdate: (res: SimulationResult) => void;
}

export const TraceTab: React.FC<TraceTabProps> = ({
  simulationResult,
  onNavigateToSimulator,
  onSimulationUpdate,
}) => {
  const [webhookRunning, setWebhookRunning] = useState<boolean>(false);
  const [webhookMessage, setWebhookMessage] = useState<string | null>(null);

  if (!simulationResult) {
    return (
      <div className="p-8 max-w-4xl mx-auto">
        <div className="bg-white border border-slate-200 rounded-xl p-10 text-center space-y-4 shadow-xs">
          <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center mx-auto">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-bold text-slate-900 text-base">No Active Decision Trace</h3>
            <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">
              Run a transaction scenario in the Demo Simulator to inspect detailed stage-by-stage reasoning, feature contributions, and execution telemetry.
            </p>
          </div>
          <button
            onClick={onNavigateToSimulator}
            className="inline-flex items-center gap-1.5 px-4 py-2 bg-blue-600 text-white rounded-lg text-xs font-bold hover:bg-blue-700 transition-all shadow-sm"
          >
            <span>Go to Demo Simulator</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    );
  }

  const decision = simulationResult.policy_decision;
  const isAct = decision === 'ACT';
  const isEscalate = decision === 'ESCALATE';

  const handleSimulateWebhook = async () => {
    setWebhookRunning(true);
    setWebhookMessage(null);
    try {
      const plinkId = `plink_demo_${simulationResult.transaction_id.slice(-6)}`;
      const payload = {
        event: 'payment_link.paid',
        payload: {
          payment_link: {
            entity: {
              id: plinkId,
              reference_id: simulationResult.transaction_id,
              status: 'paid',
              amount: Math.round(simulationResult.amount * 100),
              currency: 'INR',
            },
          },
        },
      };

      await api.simulateWebhook(payload);
      setWebhookMessage('Closed-loop webhook processed! Transaction settled as RECOVERED.');
      onSimulationUpdate({
        ...simulationResult,
        agent_status: 'RECOVERED',
        verification_status: 'VERIFIED_SETTLED',
        money_recovered: true,
      });
    } catch (err: any) {
      setWebhookMessage(`Webhook error: ${err.message}`);
    } finally {
      setWebhookRunning(false);
    }
  };

  return (
    <div className="p-8 space-y-6 max-w-6xl mx-auto">
      {/* 1. DECISION BANNER */}
      <div className={`p-6 rounded-xl border ${
        isAct
          ? 'bg-white border-l-4 border-l-emerald-600 border-slate-200'
          : isEscalate
          ? 'bg-white border-l-4 border-l-amber-500 border-slate-200'
          : 'bg-white border-l-4 border-l-red-600 border-slate-200'
      } shadow-xs`}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono font-bold text-slate-900 text-sm">
                Transaction ID: {simulationResult.transaction_id}
              </span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                isAct
                  ? 'bg-emerald-100 text-emerald-800'
                  : isEscalate
                  ? 'bg-amber-100 text-amber-800'
                  : 'bg-red-100 text-red-800'
              }`}>
                {decision}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Customer: <code className="text-slate-700">{simulationResult.customer_id}</code> •
              Failure Reason: <b className="text-slate-700 capitalize">{simulationResult.failure_reason.replace(/_/g, ' ')}</b>
            </p>
          </div>

          <div className="flex items-center gap-6 text-xs">
            <div>
              <span className="text-slate-400 block text-[10px]">Amount</span>
              <span className="font-bold text-slate-900">
                ₹{simulationResult.amount.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
              </span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px]">Recovery Probability</span>
              <span className="font-bold text-blue-600">
                {(simulationResult.recovery_probability * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px]">Agent Status</span>
              <span className={`font-bold ${simulationResult.agent_status === 'RECOVERED' ? 'text-emerald-700' : 'text-slate-800'}`}>
                {simulationResult.agent_status}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Webhook Simulator (Close Recovery Loop) */}
      {isAct && (
        <div className="bg-blue-50/60 border border-blue-200 rounded-xl p-5 flex items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 font-bold text-xs text-blue-900">
              <Zap className="w-4 h-4 text-blue-600" />
              <span>Razorpay Closed-Loop Webhook Simulator</span>
            </div>
            <p className="text-xs text-blue-800 mt-0.5">
              Simulate customer paying the generated Razorpay payment link. Verifies HMAC-SHA256 signature and settles transaction state.
            </p>
            {webhookMessage && (
              <p className="text-xs font-semibold text-emerald-700 mt-2 flex items-center gap-1">
                <Check className="w-3.5 h-3.5" />
                <span>{webhookMessage}</span>
              </p>
            )}
          </div>

          <button
            onClick={handleSimulateWebhook}
            disabled={webhookRunning || simulationResult.agent_status === 'RECOVERED'}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-xs font-bold hover:bg-blue-700 disabled:opacity-50 shrink-0 transition-all shadow-xs"
          >
            {simulationResult.agent_status === 'RECOVERED'
              ? '✓ Loop Settled (Paid)'
              : webhookRunning
              ? 'Processing Webhook...'
              : 'Simulate Customer Payment Webhook'}
          </button>
        </div>
      )}

      {/* 2. STAGE-BY-STAGE OBSERVABLE TELEMETRY */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            7-Stage Trace Logs &amp; Structured Evidence
          </h3>
          <span className="text-xs text-slate-500 font-medium">
            Zero CoT leakage • Structured JSON payloads
          </span>
        </div>

        {simulationResult.timeline_steps.map((step) => (
          <div key={step.step_number} className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
            <div className="bg-slate-50/70 px-5 py-3 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 font-bold text-xs flex items-center justify-center">
                  {step.step_number}
                </span>
                <span className="font-bold text-xs text-slate-800">{step.title}</span>
              </div>
              <span className="text-[11px] font-semibold text-slate-500 bg-white px-2 py-0.5 rounded border border-slate-200">
                Stage {step.step_number} Telemetry
              </span>
            </div>

            <div className="p-5 space-y-3">
              <p className="text-xs text-slate-700 font-medium leading-relaxed">
                {step.summary}
              </p>

              <div className="bg-slate-900 rounded-lg p-4 font-mono text-xs text-slate-100 overflow-x-auto">
                <pre className="text-[11px] leading-relaxed">
                  {JSON.stringify(step.details, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
