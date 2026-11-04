import React from 'react';
import {
  Activity,
  PlayCircle,
  Cpu,
  CheckSquare,
  BarChart3,
  ShieldCheck,
  Lock
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  pendingCount: number;
  serverOnline: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  pendingCount,
  serverOnline,
}) => {
  const navItems = [
    { id: 'command-center', label: 'Recovery Command Center', icon: Activity },
    { id: 'simulator', label: 'Demo Simulator', icon: PlayCircle },
    { id: 'trace', label: 'Live Decision Trace', icon: Cpu },
    {
      id: 'approvals',
      label: 'Merchant Approval Queue',
      icon: CheckSquare,
      badge: pendingCount > 0 ? pendingCount : undefined
    },
    { id: 'insights', label: 'Recovery Insights & Governance', icon: BarChart3 },
  ];

  return (
    <aside className="w-72 bg-slate-50 border-r border-slate-200 flex flex-col h-screen sticky top-0 shrink-0">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-200">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white font-black text-lg shadow-sm shadow-blue-500/20">
            R
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-extrabold text-slate-900 tracking-tight text-lg">RecoverAI</span>
              <span className="bg-blue-100 text-blue-700 text-[10px] font-bold px-1.5 py-0.5 rounded tracking-wide">
                BUILDATHON
              </span>
            </div>
            <p className="text-[11px] text-slate-500 font-medium">Autonomous Revenue Recovery Agent</p>
          </div>
        </div>

        {/* Live Backend Heartbeat */}
        <div className="mt-3.5 flex items-center justify-between bg-white px-3 py-2 rounded-lg border border-slate-200/80 text-xs">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${serverOnline ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="font-semibold text-slate-700">API Backend</span>
          </div>
          <span className={`text-[11px] font-bold ${serverOnline ? 'text-emerald-700' : 'text-red-700'}`}>
            {serverOnline ? 'ONLINE :8000' : 'OFFLINE'}
          </span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="p-3 space-y-1 flex-1 overflow-y-auto">
        <div className="px-3 py-1.5 text-[10px] font-bold tracking-wider text-slate-400 uppercase">
          Workspaces
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-semibold transition-all ${
                isActive
                  ? 'bg-blue-600 text-white shadow-sm shadow-blue-600/20'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80'
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-500'}`} />
                <span className="truncate">{item.label}</span>
              </div>
              {item.badge !== undefined && (
                <span className={`px-1.5 py-0.5 rounded-full text-[10px] font-bold ${
                  isActive ? 'bg-white text-blue-700' : 'bg-amber-100 text-amber-800'
                }`}>
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Governance & Policy Invariants Card */}
      <div className="p-4 border-t border-slate-200 bg-white/70 space-y-3">
        <div>
          <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-slate-500">
            <ShieldCheck className="w-3.5 h-3.5 text-blue-600" />
            <span>Active Parameters</span>
          </div>
          <div className="mt-1.5 grid grid-cols-2 gap-1.5 text-[11px]">
            <div className="bg-slate-50 p-1.5 rounded border border-slate-200/60">
              <span className="text-slate-400 block text-[10px]">Optimal τ</span>
              <span className="font-bold text-slate-800">0.35 (EV)</span>
            </div>
            <div className="bg-slate-50 p-1.5 rounded border border-slate-200/60">
              <span className="text-slate-400 block text-[10px]">Baseline τ</span>
              <span className="font-bold text-slate-800">0.50</span>
            </div>
            <div className="col-span-2 bg-slate-50 p-1.5 rounded border border-slate-200/60">
              <span className="text-slate-400 block text-[10px]">HITL Escalation Threshold</span>
              <span className="font-bold text-amber-700">₹8,500.00 Limit</span>
            </div>
          </div>
        </div>

        <div>
          <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-slate-500">
            <Lock className="w-3.5 h-3.5 text-emerald-600" />
            <span>Safety Invariants</span>
          </div>
          <p className="mt-1 text-[11px] text-slate-500 leading-relaxed">
            Permanent card failures, suspected fraud (Risk &gt; 0.70), and streak limits strictly enforced.
          </p>
        </div>
      </div>
    </aside>
  );
};
