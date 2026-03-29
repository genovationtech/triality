import { renderMarkdown } from '../../utils/markdown';
import type { SummaryEvent } from '../../types';

interface Props {
  data: SummaryEvent;
}

export function SummaryBlock({ data }: Props) {
  return (
    <div className="animate-fade-in mt-2">
      <div
        className="summary-block text-sm text-white/60 leading-relaxed"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(data.summary) }}
      />
    </div>
  );
}
