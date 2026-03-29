interface Props {
  label: string;
  value: string;
  sub?: string;
}

export function MetricCard({ label, value, sub }: Props) {
  return (
    <div className="bg-white/[0.02] border border-white/[0.04] rounded-lg p-2">
      <p className="text-[9px] font-mono text-white/25 uppercase">{label}</p>
      <p className="text-[11px] text-white/60 mt-0.5">{value}</p>
      {sub && <p className="text-[10px] text-white/35">{sub}</p>}
    </div>
  );
}
