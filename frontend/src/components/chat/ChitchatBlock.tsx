import { renderMarkdown } from '../../utils/markdown';
import type { ChitchatEvent } from '../../types';

interface Props {
  data: ChitchatEvent;
}

export function ChitchatBlock({ data }: Props) {
  return (
    <div className="animate-fade-in">
      <div
        className="summary-block text-sm text-white/60 leading-relaxed"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(data.response) }}
      />
    </div>
  );
}
