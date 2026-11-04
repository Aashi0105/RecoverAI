/**
 * RecoverAI API Client
 * Connects to FastAPI backend endpoints via configured VITE_API_BASE_URL.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface DashboardMetrics {
  total_risk: number;
  formatted_total_risk: string;
  default_recovered: number;
  formatted_default_recovered: string;
  default_cost: number;
  default_net: number;
  opt_recovered: number;
  formatted_opt_recovered: string;
  opt_cost: number;
  opt_net: number;
  gross_uplift_inr: number;
  gross_uplift_pct: number;
  net_uplift_inr: number;
  formatted_net_uplift_inr: string;
  net_uplift_pct: number;
  formatted_net_uplift_pct: string;
  inc_action_cost: number;
  recovery_rate_pct: number;
  total_failed_count: number;
  payments_selected: number;
  pending_review_count: number;
}

export interface FunnelStep {
  step: number;
  name: string;
  count: number;
  percentage: string;
  badge: string;
  description: string;
}

export interface FunnelData {
  total_failed: number;
  total_risk: number;
  steps: FunnelStep[];
  summary: {
    blocked_count: number;
    escalated_count: number;
    recovered_revenue: number;
    net_merchant_profit_uplift: number;
    net_merchant_profit_uplift_pct: number;
  };
}

export interface DecisionItem {
  key: string;
  name: string;
  count: number;
  percentage: number;
  description: string;
  color: string;
}

export interface FailureCategoryItem {
  category: string;
  share_pct: number;
  recoverability: string;
}

export interface DecisionDistribution {
  holdout_size: number;
  decisions: DecisionItem[];
  failure_categories: FailureCategoryItem[];
}

export interface RecentTransaction {
  id: string;
  transaction_id: string;
  customer_id: string;
  amount: number;
  currency: string;
  failure_code?: string | null;
  failure_reason: string;
  status: string;
  recovery_probability?: number | null;
  created_at?: string | null;
}

export interface RecentTransactionsData {
  total_records: number;
  total_recovered: number;
  total_blocked: number;
  total_pending_approval: number;
  transactions: RecentTransaction[];
}

export interface DriftFeature {
  feature: string;
  type: string;
  psi: number;
  status: string;
  recommendation: string;
}

export interface DriftReport {
  system_status: string;
  significant_drift_count: number;
  moderate_drift_count: number;
  features: DriftFeature[];
}

export interface CalibrationData {
  threshold_comparison: {
    policy: string;
    threshold: number | null;
    recovered_revenue: number;
    formatted_recovered_revenue: string;
    action_cost: number;
    formatted_action_cost: string;
    net_profit: number;
    formatted_net_profit: string;
  }[];
  default_threshold: number;
  optimal_threshold: number;
  development_grid_search: {
    threshold: number;
    payments_selected: number;
    intervention_rate_pct: number;
    expected_net_value: number;
    realized_recovered_revenue: number;
    precision: number;
    recall: number;
    f1: number;
  }[];
}

export interface SensitivityScenario {
  Scenario: string;
  'Total Revenue at Risk': string;
  'Default Gross Revenue': string;
  'Default Action Costs': string;
  'Default Net Value': string;
  'EV Gross Revenue': string;
  'EV Action Costs': string;
  'EV Realized Net Value (not causal)': string;
  'Gross Uplift': string;
  'True Net Uplift (₹)': string;
  'True Net Uplift (%)': string;
  'Positive True Net Uplift?': string;
}

export interface SensitivityData {
  source: string;
  scenario_count: number;
  scenarios: SensitivityScenario[];
}

export interface AuditRecord {
  transaction_id: string;
  timestamp: string;
  customer_id: string;
  amount: number;
  prediction?: {
    recovery_probability: number;
    model_type: string;
    explanation: string;
    top_positive_signals: { feature: string; contribution: number }[];
    top_negative_signals: { feature: string; contribution: number }[];
  };
  business_policy?: {
    threshold: number;
    recommended_action: string;
    action_cost: number;
    expected_gross_recovery_value: number;
  };
  safety?: {
    decision: 'ACT' | 'ESCALATE' | 'REFUSE';
    justification: string;
    triggered_rules: string[];
  };
  execution?: {
    execution_status: string;
    external_api_called: boolean;
    dry_run: boolean;
    reference_id?: string | null;
  };
}

export interface AuditLogData {
  total_records: number;
  act_count: number;
  escalate_count: number;
  refuse_count: number;
  decision_filter?: string | null;
  filtered_count: number;
  records: AuditRecord[];
}

export interface DemoScenario {
  id: string;
  name: string;
  description: string;
  expected_flow: string;
  expected_decision: 'ACT' | 'ESCALATE' | 'REFUSE';
  expected_action: string;
  amount: number;
  failure_reason: string;
  failure_category: string;
}

export interface TimelineStep {
  step_number: number;
  title: string;
  icon?: string;
  summary: string;
  details: Record<string, any>;
}

export interface SimulationResult {
  scenario: string;
  transaction_id: string;
  customer_id: string;
  merchant_id: string;
  amount: number;
  currency: string;
  failure_reason: string;
  failure_category: string;
  recovery_probability: number;
  expected_recovery_value: number;
  diagnosis?: string;
  diagnosis_source?: string;
  recommended_action: string;
  recommendation_source?: string;
  recommendation_confidence?: number;
  recommendation_factors?: string[];
  recommendation_expected_benefit?: string;
  policy_decision: 'ACT' | 'ESCALATE' | 'REFUSE';
  policy_reason: string;
  policy_violations?: string[];
  expected_decision: string;
  is_policy_matched?: boolean;
  business_title?: string;
  business_impact?: string;
  action_status: string;
  selected_action: string;
  action_reference?: string;
  verification_status: string;
  money_recovered: boolean;
  agent_status: string;
  timeline_steps: TimelineStep[];
  closed_loop_simulated?: boolean;
}

export interface ApprovalItem {
  id?: number;
  transaction_id: string;
  merchant_id: string;
  customer_id: string;
  customer_email?: string;
  amount: number;
  currency: string;
  failure_reason: string;
  failure_category?: string;
  recovery_probability: number;
  recommended_action: string;
  action_cost?: number;
  policy_reason?: string;
  diagnosis_summary?: string;
  escalation_reason?: string;
  status: string;
  human_decision?: string;
  created_at: string;
}

export interface PendingApprovalsData {
  total_pending: number;
  approvals: ApprovalItem[];
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers || {}),
    },
  });

  if (!res.ok) {
    let errorDetail = res.statusText;
    try {
      const errorJson = await res.json();
      errorDetail = errorJson.detail?.message || errorJson.detail || JSON.stringify(errorJson);
    } catch {}
    throw new Error(`API Error [${res.status}]: ${errorDetail}`);
  }

  return res.json();
}

export const api = {
  // System Health
  health: () => request<{ status: string }>('/health'),

  // Dashboard Endpoints
  getMetrics: () => request<DashboardMetrics>('/api/v1/dashboard/metrics'),
  getFunnel: () => request<FunnelData>('/api/v1/dashboard/funnel'),
  getDecisionDistribution: () => request<DecisionDistribution>('/api/v1/dashboard/decision-distribution'),
  getRecentTransactions: (limit = 10) => request<RecentTransactionsData>(`/api/v1/dashboard/recent-transactions?limit=${limit}`),
  getDriftReport: () => request<DriftReport>('/api/v1/dashboard/drift'),
  getCalibration: () => request<CalibrationData>('/api/v1/dashboard/calibration'),
  getSensitivity: () => request<SensitivityData>('/api/v1/dashboard/sensitivity'),
  getAuditLog: (decision?: string, limit = 50) => {
    const q = new URLSearchParams();
    if (decision && decision !== 'ALL') q.set('decision', decision);
    q.set('limit', limit.toString());
    return request<AuditLogData>(`/api/v1/dashboard/audit-log?${q.toString()}`);
  },

  // Demo Simulation Endpoints
  getScenarios: () => request<DemoScenario[]>('/api/v1/demo/scenarios'),
  simulateScenario: (scenario_id: string) =>
    request<SimulationResult>('/api/v1/demo/simulate', {
      method: 'POST',
      body: JSON.stringify({ scenario_id }),
    }),
  simulateWebhook: (payload: any) =>
    request<any>('/api/v1/demo/simulate-webhook', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Approval Endpoints
  getPendingApprovals: () => request<PendingApprovalsData>('/api/v1/approvals/pending'),
  approveTransaction: (transaction_id: string, notes?: string) =>
    request<{ message: string; status: string }>(`/api/v1/approvals/${transaction_id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ notes, dry_run: true }),
    }),
  rejectTransaction: (transaction_id: string, notes?: string) =>
    request<{ message: string; status: string }>(`/api/v1/approvals/${transaction_id}/reject`, {
      method: 'POST',
      body: JSON.stringify({ notes }),
    }),
};
