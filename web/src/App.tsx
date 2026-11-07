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
  }, []);

  // Monitor server health: poll every 5s while waking up / offline, 30s when online
  useEffect(() => {
    const intervalTime = serverOnline ? 30000 : 5000;
    const interval = setInterval(async () => {
      try {
        await api.health();
        if (!serverOnline) {
          // Backend has finished waking up: refresh active views
          setServerOnline(true);
          loadScenarios();
          loadPendingCount();
          setRefreshKey((prev) => prev + 1);
        }
      } catch {
        if (serverOnline) {
          setServerOnline(false);
        }
      }
    }, intervalTime);
    return () => clearInterval(interval);
  }, [serverOnline]);

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

        {/* Cold Start / Waking Up Banner for Free Cloud Instances (Render) */}
        {!serverOnline && (
          <div className="mx-4 sm:mx-6 lg:mx-8 mt-4 bg-amber-50 border border-amber-300 rounded-xl p-4 text-amber-900 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 animate-pulse">
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-amber-600 border-t-transparent rounded-full animate-spin shrink-0" />
              <div>
                <p className="text-xs sm:text-sm font-bold text-amber-900">
                  Backend is waking up, please wait...
                </p>
                <p className="text-[11px] sm:text-xs text-amber-700 mt-0.5">
                  Render free tier instances sleep after inactivity and typically take 30–60 seconds to spin up. Reconnecting automatically...
                </p>
              </div>
            </div>
            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="px-3.5 py-1.5 bg-amber-600 hover:bg-amber-700 text-white rounded-lg text-xs font-semibold shadow-xs disabled:opacity-50 shrink-0 cursor-pointer transition"
            >
              {isRefreshing ? 'Checking...' : 'Retry Now'}
            </button>
          </div>
        )}

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
