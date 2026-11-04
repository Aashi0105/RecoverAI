import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { CommandCenterTab } from './components/CommandCenterTab';
import { SimulatorTab } from './components/SimulatorTab';
import { TraceTab } from './components/TraceTab';
import { ApprovalsTab } from './components/ApprovalsTab';
import { InsightsTab } from './components/InsightsTab';
import { api, type DemoScenario, type SimulationResult } from './api/client';

export const App: React.FC = () => {
  const getInitialTab = () => {
    const hash = window.location.hash.replace('#', '');
    const valid = ['command-center', 'simulator', 'trace', 'approvals', 'insights'];
    return valid.includes(hash) ? hash : 'command-center';
  };

  const [activeTab, setActiveTabState] = useState<string>(getInitialTab);
  const [scenarios, setScenarios] = useState<DemoScenario[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('auto_recovery');
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [pendingCount, setPendingCount] = useState<number>(0);
  const [serverOnline, setServerOnline] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [refreshKey, setRefreshKey] = useState<number>(0);

  const setActiveTab = (tab: string) => {
    setActiveTabState(tab);
    window.location.hash = tab;
  };

  // Sync with browser hash changes
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      const valid = ['command-center', 'simulator', 'trace', 'approvals', 'insights'];
      if (valid.includes(hash)) {
        setActiveTabState(hash);
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Initial data loading
  useEffect(() => {
    checkHealth();
    loadScenarios();
    loadPendingCount();
    initSimulation();

    // Check health periodically
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const initSimulation = async () => {
    try {
      const res = await api.simulateScenario('auto_recovery');
      setSimulationResult(res);
    } catch (e) {
      console.warn('Initial simulation fallback:', e);
    }
  };

  const checkHealth = async () => {
    try {
      await api.health();
      setServerOnline(true);
    } catch (err) {
      setServerOnline(false);
    }
  };

  const loadScenarios = async () => {
    try {
      const data = await api.getScenarios();
      if (Array.isArray(data) && data.length > 0) {
        setScenarios(data);
      }
    } catch (err) {
      console.warn('Could not fetch scenarios from backend; fallback scenarios will be used');
    }
  };

  const loadPendingCount = async () => {
    try {
      const data = await api.getPendingApprovals();
      setPendingCount(data.total_pending || 0);
    } catch (err) {
      console.warn('Could not fetch pending approvals count');
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await checkHealth();
    await loadPendingCount();
    setRefreshKey((prev) => prev + 1);
    setTimeout(() => setIsRefreshing(false), 400);
  };

  const getHeaderInfo = () => {
    switch (activeTab) {
      case 'command-center':
        return {
          title: 'Recovery Command Center',
          description: 'Live telemetry, baseline vs optimal policy recovery performance, and conversion audit metrics',
        };
      case 'simulator':
        return {
          title: 'Payment Failure & Autonomous Recovery Simulator',
          description: 'Simulate realistic payment failure scenarios and observe the LangGraph recovery policy pipeline',
        };
      case 'trace':
        return {
          title: 'Live Decision Trace & Webhook Execution',
          description: 'Stage-by-stage execution trace, feature contributions, and closed-loop payment verification',
        };
      case 'approvals':
        return {
          title: 'Merchant Approval Queue (HITL)',
          description: 'Human-in-the-loop governance queue for high-value transactions requiring merchant authorization',
        };
      case 'insights':
        return {
          title: 'Recovery Insights & System Governance',
          description: 'Model calibration, population drift (PSI), sensitivity stress tests, and immutable audit logs',
        };
      default:
        return {
          title: 'RecoverAI Dashboard',
          description: 'Autonomous AI Revenue Recovery Agent',
        };
    }
  };

  const headerInfo = getHeaderInfo();

  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-900 font-sans antialiased">
      {/* Fixed Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        pendingCount={pendingCount}
        serverOnline={serverOnline}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header
          title={headerInfo.title}
          description={headerInfo.description}
          onRefresh={handleRefresh}
          isLoading={isRefreshing}
        />

        <main className="flex-1 overflow-y-auto">
          {activeTab === 'command-center' && (
            <CommandCenterTab key={`command-center-${refreshKey}`} />
          )}

          {activeTab === 'simulator' && (
            <SimulatorTab
              scenarios={scenarios}
              selectedScenarioId={selectedScenarioId}
              setSelectedScenarioId={setSelectedScenarioId}
              simulationResult={simulationResult}
              setSimulationResult={setSimulationResult}
              onNavigateToTrace={() => setActiveTab('trace')}
            />
          )}

          {activeTab === 'trace' && (
            <TraceTab
              simulationResult={simulationResult}
              onNavigateToSimulator={() => setActiveTab('simulator')}
              onSimulationUpdate={setSimulationResult}
            />
          )}

          {activeTab === 'approvals' && (
            <ApprovalsTab
              key={`approvals-${refreshKey}`}
              onApprovalCountChange={setPendingCount}
            />
          )}

          {activeTab === 'insights' && (
            <InsightsTab key={`insights-${refreshKey}`} />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
