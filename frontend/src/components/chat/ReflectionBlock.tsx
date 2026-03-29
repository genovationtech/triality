import { Sparkles } from 'lucide-react';
import { renderMarkdown } from '../../utils/markdown';
import type { ReflectionEvent, ReflectionAddendumEvent } from '../../types';

export function ReflectionNotice({ data }: { data: ReflectionEvent }) {
  return (
    <div className="flex items-center gap-2 mt-2 animate-fade-in">
      <Sparkles className="w-3 h-3 text-cyan-400/50" />
      <span className="text-[10px] font-mono text-cyan-400/40">
        Reflecting: {data.reason}
      </span>
    </div>
  );
}

export function ReflectionAddendum({ data }: { data: ReflectionAddendumEvent }) {
  if (!data.addendum) return null;
  return (
    <div className="animate-fade-in mt-2 pl-5 border-l-2 border-cyan-400/10">
      <div
        className="summary-block text-xs text-white/50 leading-relaxed"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(data.addendum) }}
      />
    </div>
  );
}
