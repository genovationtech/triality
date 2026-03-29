interface Props {
  children: React.ReactNode;
  color?: 'accent' | 'cyan';
}

export function Pill({ children, color = 'accent' }: Props) {
  const cls = color === 'cyan' ? 'text-cyan-400/50' : 'text-[#6ee7b7]/50';
  return (
    <span
      className={`border border-white/[0.06] bg-white/[0.02] rounded-full px-2 py-0.5 text-[10px] font-mono ${cls}`}
    >
      {children}
    </span>
  );
}
