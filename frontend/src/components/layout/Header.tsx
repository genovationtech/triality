import { HelpCircle } from 'lucide-react';

interface Props {
  onHelp: () => void;
}

export function Header({ onHelp }: Props) {
  return (
    <header className="h-12 border-b border-white/[0.04] flex items-center justify-between px-5 bg-[#0d0f14]/40 backdrop-blur-xl">
      <div className="flex items-center gap-2.5">
        <div className="w-1.5 h-1.5 rounded-full bg-[#6ee7b7]/60" />
        <span className="text-[11px] font-mono text-white/30">Triality Agent</span>
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={onHelp}
          className="flex items-center gap-1.5 text-[10px] text-white/25 hover:text-[#6ee7b7]/70 transition-colors"
        >
          <HelpCircle className="w-3.5 h-3.5" />
          <span>Help</span>
        </button>
        <span className="text-[10px] font-mono text-white/15">
          Powered by <span className="text-[#6ee7b7]/30">Mentis OS</span>
        </span>
      </div>
    </header>
  );
}
