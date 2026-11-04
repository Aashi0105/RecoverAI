import React, { useEffect, useState } from 'react';
import {
  CheckSquare,
  Check,
  AlertCircle,
  Clock,
  RefreshCw
} from 'lucide-react';
import { api, type ApprovalItem } from '../api/client';

interface ApprovalsTabProps {
  onApprovalCountChange?: (count: number) => void;
}

export const ApprovalsTab: React.FC<ApprovalsTabProps> = ({ onApprovalCountChange }) => {
  const [approvals, setApprovals] = useState<ApprovalItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [actionNotes, setActionNotes] = useState<Record<string, string>>({});
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [actionFeedback, setActionFeedback] = useState<{ id: string; type: 'success' | 'error'; message: string } | null>(null);

  useEffect(() => {
    loadApprovals();
  }, []);

  const loadApprovals = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getPendingApprovals();
      setApprovals(data.approvals || []);
      if (onApprovalCountChange) {
        onApprovalCountChange(data.approvals?.length || 0);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load approvals queue');
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (txnId: string) => {
    setProcessingId(txnId);
    setActionFeedback(null);
    try {
      const notes = actionNotes[txnId] || 'Merchant approved via web command center';
      const res = await api.approveTransaction(txnId, notes);
      setActionFeedback({
        id: txnId,
        type: 'success',
        message: `Transaction ${txnId} approved successfully: ${res.message || 'Action executed'}`,
      });
      await loadApprovals();
    } catch (err: any) {
      setActionFeedback({
        id: txnId,
        type: 'error',
        message: err.message || 'Failed to approve transaction',
      });
    } finally {
      setProcessingId(null);
    }
  };

  const handleReject = async (txnId: string) => {
    setProcessingId(txnId);
    setActionFeedback(null);
    try {
      const notes = actionNotes[txnId] || 'Merchant rejected via web command center';
      const res = await api.rejectTransaction(txnId, notes);
      setActionFeedback({
        id: txnId,
        type: 'success',
        message: `Transaction ${txnId} rejected safely: ${res.message || 'Zero payments executed'}`,
      });
      await loadApprovals();
    } catch (err: any) {
      setActionFeedback({
        id: txnId,
        type: 'error',
        message: err.message || 'Failed to reject transaction',
      });
    } finally {
      setProcessingId(null);
    }
  };

  const handleNotesChange = (txnId: string, value: string) => {
    setActionNotes((prev) => ({ ...prev, [txnId]: value }));
  };

  return (
    <div className="p-8 space-y-6 max-w-5xl mx-auto">
      {/* Header Info */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-bold text-slate-900 tracking-tight">
              Human-in-the-Loop (HITL) Governance Queue
            </h2>
            <span className="bg-amber-100 text-amber-800 text-xs font-bold px-2 py-0.5 rounded-full border border-amber-200">
              {approvals.length} Pending
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Orders escalated due to high financial exposure (&gt; ₹8,500 threshold) or sensitive customer tier.
          </p>
        </div>

        <button
          onClick={loadApprovals}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-blue-600' : 'text-slate-500'}`} />
          <span>Refresh Queue</span>
        </button>
      </div>

      {/* Global Feedback Alert */}
      {actionFeedback && (
        <div className={`p-4 rounded-xl border text-xs flex items-center justify-between ${
          actionFeedback.type === 'success'
            ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
            : 'bg-red-50 border-red-200 text-red-800'
        }`}>
          <div className="flex items-center gap-2">
            {actionFeedback.type === 'success' ? (
              <Check className="w-4 h-4 text-emerald-600 shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
            )}
            <span className="font-medium">{actionFeedback.message}</span>
          </div>
          <button
            onClick={() => setActionFeedback(null)}
            className="font-bold ml-4 hover:underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-xs text-red-800 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Queue Items */}
      {loading && approvals.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center">
          <div className="w-7 h-7 border-3 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-xs font-semibold text-slate-500">Checking escalation queue...</p>
        </div>
      ) : approvals.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center space-y-3">
          <div className="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-full flex items-center justify-center mx-auto">
            <CheckSquare className="w-6 h-6" />
          </div>
          <h3 className="font-bold text-slate-900 text-sm">Escalation Queue is Empty</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto">
            All high-value recovery attempts have been resolved. Run Scenario B in Demo Simulator to inject a live escalated order.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {approvals.map((item) => {
            const isProcessing = processingId === item.transaction_id;
            const noteVal = actionNotes[item.transaction_id] || '';

            return (
              <div
                key={item.transaction_id}
                className="bg-white border border-slate-200 border-l-4 border-l-amber-500 rounded-xl p-6 shadow-xs space-y-4"
              >
                {/* Header Strip */}
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5">
                    <span className="font-mono font-bold text-sm text-slate-900">
                      {item.transaction_id}
                    </span>
                    <span className="bg-amber-100 text-amber-800 text-[10px] font-bold px-2 py-0.5 rounded border border-amber-200">
                      REQUIRES MERCHANT DECISION
                    </span>
                  </div>

                  <div className="flex items-center gap-4 text-xs font-medium text-slate-500">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      <span>{new Date(item.created_at).toLocaleString()}</span>
                    </span>
                  </div>
                </div>

                {/* Main Details Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 bg-slate-50/70 p-4 rounded-lg border border-slate-200/60 text-xs">
                  <div>
                    <span className="text-slate-400 block text-[10px] uppercase font-bold">Amount at Risk</span>
                    <span className="font-bold text-slate-900 text-sm">
                      ₹{item.amount.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px] uppercase font-bold">Customer</span>
                    <span className="font-semibold text-slate-700 font-mono">
                      {item.customer_id}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px] uppercase font-bold">ML Probability</span>
                    <span className="font-bold text-blue-700 text-sm">
                      {(item.recovery_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px] uppercase font-bold">Recommended Action</span>
                    <span className="font-bold text-slate-800 uppercase font-mono">
                      {item.recommended_action} (Cost: ₹{item.action_cost ?? 5})
                    </span>
                  </div>
                </div>

                {/* Escalation Justification */}
                <div className="text-xs text-slate-600 bg-amber-50/50 border border-amber-200/80 p-3 rounded-lg">
                  <span className="font-bold text-amber-900 block mb-0.5">Escalation Trigger:</span>
                  <span>{item.policy_reason || item.diagnosis_summary || item.escalation_reason || 'Transaction amount exceeds standard automated threshold (₹8,500.00).'}</span>
                </div>

                {/* Actions & Notes Input */}
                <div className="pt-2 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
                  <input
                    type="text"
                    value={noteVal}
                    onChange={(e) => handleNotesChange(item.transaction_id, e.target.value)}
                    placeholder="Optional merchant note (e.g., 'Approved via Slack confirmation')..."
                    className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/20 focus:border-blue-600"
                  />

                  <div className="flex items-center gap-2.5">
                    <button
                      onClick={() => handleReject(item.transaction_id)}
                      disabled={isProcessing}
                      className="px-4 py-2 rounded-lg border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 hover:border-red-300 text-xs font-bold transition-all disabled:opacity-50"
                    >
                      {isProcessing ? 'Processing...' : 'Reject Recovery Action'}
                    </button>

                    <button
                      onClick={() => handleApprove(item.transaction_id)}
                      disabled={isProcessing}
                      className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold shadow-xs transition-all disabled:opacity-50"
                    >
                      {isProcessing ? 'Processing...' : 'Approve Recovery Action'}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
