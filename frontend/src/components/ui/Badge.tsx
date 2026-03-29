interface Props {
  variant: 'full' | 'partial' | 'fail';
  children: React.ReactNode;
}

const variants = {
  full: 'bg-[rgba(110,231,183,0.1)] text-[#6ee7b7] border-[rgba(110,231,183,0.2)]',
  partial: 'bg-[rgba(251,191,36,0.1)] text-[#fbbf24] border-[rgba(251,191,36,0.2)]',
  fail: 'bg-[rgba(248,113,113,0.1)] text-[#f87171] border-[rgba(248,113,113,0.2)]',
};

export function Badge({ variant, children }: Props) {
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-px rounded-full text-[0.65rem] font-semibold uppercase tracking-wide font-mono border ${variants[variant]}`}
    >
      {children}
    </span>
  );
}
