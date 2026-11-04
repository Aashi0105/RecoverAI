import React from 'react';
import { RefreshCw } from 'lucide-react';

interface HeaderProps {
  title: string;
  description: string;
  onRefresh?: () => void;
  isLoading?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  description,
  onRefresh,
  isLoading,
}) => {
  return (
    <header className="bg-white border-b border-slate-200 px-8 py-5 flex items-center justify-between sticky top-0 z-10">
      <div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">{title}</h1>
        <p className="text-xs text-slate-500 mt-0.5">{description}</p>
      </div>
      <div className="flex items-center gap-3">
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 hover:border-slate-300 transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-blue-600' : 'text-slate-500'}`} />
            <span>Refresh</span>
          </button>
        )}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-xs font-semibold">
          <span>Razorpay Test Mode</span>
          <span className="w-1.5 h-1.5 rounded-full bg-blue-600"></span>
        </div>
      </div>
    </header>
  );
};
