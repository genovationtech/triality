import { Brain } from 'lucide-react';
import type { ThinkingEvent } from '../../types';

interface Props {
  data: ThinkingEvent;
}

export function ThinkingBlock({ data }: Props) {
  const source = data.source === 'llm' ? 'LLM' : 'Heuristic';

  return (
    <div className="rounded-xl bg-[#11141b]/60 border border-white/[0.04] p-3 animate-fade-in">
      <div className="flex items-center gap-2 mb-2">
        <Brain className="w-3 h-3 text-[#6ee7b7]/50" />
        <span className="text-[10px] font-mono text-[#6ee7b7]/40 uppercase">
          {source} Planning
        </span>
      </div>
      {data.thinking && (
        <p className="text-xs text-white/40 italic mb-2">{data.thinking}</p>
      )}
      {data.plan?.length > 0 && (
        <div className="space-y-1">
          {data.plan.map((step, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-[#6ee7b7]/30">{i + 1}.</span>
              <span className="text-[11px] text-white/45">{step}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
