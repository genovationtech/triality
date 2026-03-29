import { useState } from 'react';
import { Microscope, FlaskConical, Factory, Zap } from 'lucide-react';
import { QuickstartCard } from './QuickstartCard';
import type { Scenario } from '../../types';

interface Props {
  scenarios: Scenario[];
  onSelectScenario: (scenario: Scenario) => void;
}

const CATEGORY_META: Record<string, { label: string; shortLabel: string; icon: React.ReactNode; color: string }> = {
  fundamentals: {
    label: 'Fundamentals',
    shortLabel: 'Fundamentals',
    icon: <FlaskConical className="w-3 h-3" />,
    color: '#6ee7b7',
  },
  industry: {
    label: 'Industry',
    shortLabel: 'Industry',
    icon: <Factory className="w-3 h-3" />,
    color: '#22d3ee',
  },
  advanced: {
    label: 'Advanced',
    shortLabel: 'Advanced',
    icon: <Zap className="w-3 h-3" />,
    color: '#f59e0b',
  },
};

const CATEGORY_ORDER = ['fundamentals', 'industry', 'advanced'];

export function WelcomeMessage({ scenarios, onSelectScenario }: Props) {
  const [activeTab, setActiveTab] = useState('fundamentals');

  const grouped = scenarios.reduce<Record<string, Scenario[]>>((acc, s) => {
    const cat = s.category || 'fundamentals';
    (acc[cat] ||= []).push(s);
    return acc;
  }, {});

  const activeItems = grouped[activeTab] || [];

  return (
    <div className="animate-fade-in">
      <div className="flex items-start gap-3">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#6ee7b7]/20 to-cyan-400/20 border border-[#6ee7b7]/15 flex items-center justify-center shrink-0 mt-0.5">
          <Microscope className="w-3.5 h-3.5 text-[#6ee7b7]" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[11px] font-mono text-[#6ee7b7]/50 mb-2">Triality Agent</p>
          <div className="text-sm text-white/60 leading-relaxed space-y-2 mb-4">
            <p>
              I'm your physics reasoning agent. I can{' '}
              <strong className="text-white/80">plan</strong>,{' '}
              <strong className="text-white/80">execute</strong>, and{' '}
              <strong className="text-white/80">analyse</strong> physics problems autonomously.
            </p>
            <p className="text-white/40 text-xs">
              Choose a scenario below, or describe what you'd like to analyse.
            </p>
          </div>

          {/* Tab bar */}
          <div className="flex items-center gap-1 mb-4 p-0.5 bg-white/[0.03] rounded-lg border border-white/[0.04] w-fit">
            {CATEGORY_ORDER.map((cat) => {
              const meta = CATEGORY_META[cat];
              const items = grouped[cat] || [];
              if (!items.length) return null;
              const isActive = activeTab === cat;

              return (
                <button
                  key={cat}
                  onClick={() => setActiveTab(cat)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-medium transition-all ${
                    isActive
                      ? 'bg-white/[0.08] text-white/80 shadow-sm'
                      : 'text-white/35 hover:text-white/55 hover:bg-white/[0.03]'
                  }`}
                >
                  <span className={isActive ? '' : 'opacity-50'}
                        style={{ color: isActive ? meta.color : undefined }}>
                    {meta.icon}
                  </span>
                  <span>{meta.shortLabel}</span>
                  <span className={`text-[9px] font-mono px-1 py-0.5 rounded ${
                    isActive
                      ? 'bg-white/[0.08] text-white/50'
                      : 'text-white/20'
                  }`}>
                    {items.length}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Card grid for active tab */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {activeItems.map((s) => (
              <QuickstartCard key={s.id} scenario={s} onClick={onSelectScenario} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
