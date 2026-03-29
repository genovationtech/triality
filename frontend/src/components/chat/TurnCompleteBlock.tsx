import { Badge } from '../ui/Badge';
import type { TurnCompleteEvent } from '../../types';

interface Props {
  data: TurnCompleteEvent;
}

export function TurnCompleteBlock({ data }: Props) {
  return (
    <div className="flex items-center gap-2 mt-3 pt-2 border-t border-white/[0.04] animate-fade-in">
      <Badge variant={data.all_succeeded ? 'full' : 'fail'}>
        {data.all_succeeded ? 'Complete' : 'Partial'}
      </Badge>
      <span className="text-[10px] font-mono text-white/25">
        {data.total_tools} tool(s) executed
      </span>
    </div>
  );
}
