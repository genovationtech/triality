import * as LucideIcons from 'lucide-react';
import type { Scenario } from '../../types';

interface Props {
  scenario: Scenario;
  onClick: (scenario: Scenario) => void;
}

// Map icon name strings to Lucide components
function getIcon(name: string) {
  // Convert kebab-case to PascalCase
  const pascal = name
    .split('-')
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join('');
  const Icon = (LucideIcons as Record<string, React.ComponentType<{ className?: string }>>)[pascal];
  return Icon || LucideIcons.FlaskConical;
}

export function QuickstartCard({ scenario, onClick }: Props) {
  const Icon = getIcon(scenario.icon || 'flask-conical');
  const title = scenario.business_title || scenario.title;
  const problem = scenario.industry_problem || scenario.description;

  return (
    <button
      className="quickstart-card text-left rounded-xl bg-[#11141b]/60 border border-white/[0.06] p-4 hover:border-[#6ee7b7]/30 hover:bg-[#161a24]/60 transition-all group cursor-pointer"
      onClick={() => onClick(scenario)}
    >
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-lg bg-[#6ee7b7]/10 border border-[#6ee7b7]/15 flex items-center justify-center shrink-0 mt-0.5">
          <Icon className="w-4 h-4 text-[#6ee7b7]/60" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[12px] font-semibold text-white/70 group-hover:text-white/90 transition-colors">
            {title}
          </p>
          <p className="text-[11px] text-white/40 mt-1 line-clamp-2">{problem}</p>
          {scenario.decision_focus && (
            <p className="text-[10px] font-mono text-[#6ee7b7]/30 mt-1.5 line-clamp-2">
              {scenario.decision_focus}
            </p>
          )}
        </div>
      </div>
    </button>
  );
}
