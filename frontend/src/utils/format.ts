export function formatNumber(n: number): string {
  if (typeof n !== 'number' || isNaN(n)) return String(n);
  if (Math.abs(n) < 0.001 || Math.abs(n) > 99999) return n.toExponential(3);
  return n.toFixed(4).replace(/\.?0+$/, '');
}

export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
