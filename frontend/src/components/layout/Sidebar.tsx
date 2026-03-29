import { useState } from 'react';
import {
  Microscope,
  Settings,
  FlaskConical,
  Box,
  Wrench,
  Sparkles,
  EyeOff,
  Eye,
  ChevronRight,
} from 'lucide-react';
import * as LucideIcons from 'lucide-react';
import { Pill } from '../ui/Pill';
import { Badge } from '../ui/Badge';
import type { Catalog, Scenario } from '../../types';

interface Props {
  catalog: Catalog | null;
  token: string;
  onTokenChange: (token: string) => void;
  model: string;
  onModelChange: (model: string) => void;
  onSelectScenario: (scenario: Scenario) => void;
  footerStatus: string;
  onClear: () => void;
  onNavigate: (page: string) => void;
}

function getIcon(name: string) {
  const pascal = name
    .split('-')
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join('');
  return (LucideIcons as Record<string, React.ComponentType<{ className?: string }>>)[pascal] || FlaskConical;
}

export function Sidebar({
  catalog,
  token,
  onTokenChange,
  model,
  onModelChange,
  onSelectScenario,
  footerStatus,
  onClear,
  onNavigate,
}: Props) {
  const [showToken, setShowToken] = useState(false);

  return (
    <aside className="w-[300px] min-w-[300px] bg-[#0d0f14]/80 backdrop-blur-xl border-r border-white/[0.04] flex flex-col">
      {/* Logo */}
      <div className="p-5 pb-3">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[#6ee7b7]/20 to-cyan-400/20 border border-[#6ee7b7]/20 flex items-center justify-center">
            <Microscope className="w-4 h-4 text-[#6ee7b7]" />
          </div>
          <div>
            <h2 className="font-display font-bold text-white text-sm tracking-tight">
              Triality Agent
            </h2>
            <span className="text-[9px] font-mono text-white/25 tracking-wider">
              Powered by <span className="text-[#6ee7b7]/50 font-semibold">Mentis OS</span>
            </span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {/* Settings */}
        <div className="mb-5">
          <SectionHeader icon={<Settings className="w-3.5 h-3.5 text-white/30" />} label="Settings" />
          <div className="mb-2.5">
            <label className="text-[10px] font-medium text-white/35 mb-1 block px-1">
              Replicate API Token <span className="text-red-400/60">*</span>
            </label>
            <div className="relative">
              <input
                type={showToken ? 'text' : 'password'}
                value={token}
                onChange={(e) => onTokenChange(e.target.value)}
                placeholder="Required — paste your Replicate token"
                className="token-field w-full rounded-lg px-3 py-2 text-[11px] font-mono text-[#6ee7b7]/80 outline-none pr-8"
              />
              <button
                onClick={() => setShowToken(!showToken)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-white/25 hover:text-[#6ee7b7] transition-colors"
              >
                {showToken ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
              </button>
            </div>
          </div>
          <div>
            <label className="text-[10px] font-medium text-white/35 mb-1 block px-1">Model</label>
            <select
              value={model}
              onChange={(e) => onModelChange(e.target.value)}
              className="custom-select token-field w-full rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 outline-none cursor-pointer"
            >
              <option>meta/llama-4-maverick-instruct</option>
              <option>meta/llama-3.3-70b-instruct</option>
              <option>anthropic/claude-sonnet-4</option>
            </select>
          </div>
        </div>

        <Divider />

        {/* Scenarios */}
        <div className="mb-5">
          <SectionHeader icon={<FlaskConical className="w-3.5 h-3.5 text-white/30" />} label="Scenarios" />
          {(() => {
            const scenarios = catalog?.scenarios || [];
            const grouped = scenarios.reduce<Record<string, typeof scenarios>>((acc, s) => {
              const cat = (s as { category?: string }).category || 'fundamentals';
              (acc[cat] ||= []).push(s);
              return acc;
            }, {});
            const categoryOrder = ['fundamentals', 'industry', 'advanced'];
            const categoryLabels: Record<string, string> = {
              fundamentals: 'Fundamentals',
              industry: 'Industry Problems',
              advanced: 'Advanced Multi-Physics',
            };
            return categoryOrder.map((cat) => {
              const items = grouped[cat];
              if (!items?.length) return null;
              return (
                <div key={cat} className="mb-3">
                  <p className="text-[9px] font-semibold text-white/20 uppercase tracking-[0.15em] px-3 mb-1.5">
                    {categoryLabels[cat] || cat}
                  </p>
                  <div className="space-y-1">
                    {items.map((s) => {
                      const Icon = getIcon(s.icon || 'flask-conical');
                      return (
                        <button
                          key={s.id}
                          className="scenario-card w-full text-left rounded-lg px-3 py-2.5 cursor-pointer group"
                          onClick={() => onSelectScenario(s)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex-1 min-w-0">
                              <p className="text-[12px] font-medium text-white/60 group-hover:text-white/85 transition-colors truncate">
                                {s.business_title || s.title}
                              </p>
                              <p className="text-[10px] font-mono text-[#6ee7b7]/30 mt-0.5">
                                {s.subtitle}
                              </p>
                            </div>
                            <ChevronRight className="w-3 h-3 text-white/15 group-hover:text-[#6ee7b7]/50 transition-colors shrink-0" />
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            });
          })()}
        </div>

        <Divider />

        {/* Modules */}
        <div className="mb-5">
          <SectionHeader icon={<Box className="w-3.5 h-3.5 text-white/30" />} label="Modules" />
          <div className="flex flex-wrap gap-1.5 px-1">
            {Object.keys(catalog?.modules || {}).map((m) => (
              <Pill key={m}>{m}</Pill>
            ))}
          </div>
        </div>

        {/* Tools */}
        <div className="mb-5">
          <SectionHeader icon={<Wrench className="w-3.5 h-3.5 text-white/30" />} label="Tools" />
          <div className="flex flex-wrap gap-1.5 px-1">
            {(catalog?.tools || []).map((t) => (
              <Pill key={t} color="cyan">
                {t}
              </Pill>
            ))}
          </div>
        </div>

        {/* Capabilities */}
        <div>
          <SectionHeader icon={<Sparkles className="w-3.5 h-3.5 text-white/30" />} label="Capabilities" />
          <div className="space-y-1.5 px-1">
            {(catalog?.capabilities || []).map((c) => {
              const CapIcon = getIcon(c.icon);
              return (
                <div key={c.name} className="flex items-center gap-2 py-1">
                  <CapIcon className="w-3 h-3 text-[#6ee7b7]/40" />
                  <span className="text-[10px] text-white/40">{c.name}</span>
                  <span className="ml-auto">
                    <Badge variant={c.status === 'Full' ? 'full' : 'partial'}>
                      {c.status}
                    </Badge>
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-white/[0.04]">
        <div className="px-3 pt-3 pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-[10px] text-white/20">
              <div className="pulse-dot" />
              <span className="font-mono">{footerStatus}</span>
            </div>
            <button
              onClick={onClear}
              className="text-[10px] text-white/20 hover:text-[#6ee7b7] transition-colors font-mono"
            >
              Clear
            </button>
          </div>
        </div>
        <div className="px-3 pb-3 pt-1">
          <div className="text-[9px] text-white/15 leading-relaxed space-y-0.5">
            <p className="font-medium text-white/20">
              Genovation Technological Solutions Pvt. Ltd.
            </p>
            <p>
              <a href="https://genovationsolutions.com" target="_blank" rel="noopener noreferrer" className="hover:text-[#6ee7b7]/50 transition-colors">
                genovationsolutions.com
              </a>
              {' \u00B7 '}
              <a href="mailto:connect@genovationsolutions.com" className="hover:text-[#6ee7b7]/50 transition-colors">
                connect@genovationsolutions.com
              </a>
            </p>
            <div className="flex items-center gap-1.5 pt-1 text-white/12">
              <button onClick={() => onNavigate('terms')} className="hover:text-white/25 transition-colors">Terms</button>
              <span>{'\u00B7'}</span>
              <button onClick={() => onNavigate('privacy')} className="hover:text-white/25 transition-colors">Privacy</button>
              <span>{'\u00B7'}</span>
              <button onClick={() => onNavigate('disclaimer')} className="hover:text-white/25 transition-colors">Disclaimer</button>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

function SectionHeader({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="flex items-center gap-2 mb-2 px-1">
      {icon}
      <span className="text-[10px] font-semibold text-white/30 uppercase tracking-[0.15em]">
        {label}
      </span>
    </div>
  );
}

function Divider() {
  return (
    <div className="h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent mb-5" />
  );
}
