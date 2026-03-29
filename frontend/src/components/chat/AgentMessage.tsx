import { Microscope, Loader2 } from 'lucide-react';
import { ThinkingBlock } from './ThinkingBlock';
import { ToolBlock } from './ToolBlock';
import { SummaryBlock } from './SummaryBlock';
import { ChitchatBlock } from './ChitchatBlock';
import { ReflectionNotice, ReflectionAddendum } from './ReflectionBlock';
import { TurnCompleteBlock } from './TurnCompleteBlock';
import {
  GoalExtractedBlock,
  AnalyticalEstimateBlock,
  ConvergenceStepBlock,
  GoalSatisfiedBlock,
  ConvergenceAdaptingBlock,
  ConvergenceStalledBlock,
  ConvergenceExhaustedBlock,
} from './ConvergenceBlock';
import type { ChatMessage, SSEBlock, ToolStartEvent, ToolResultEvent, ToolProgressEvent } from '../../types';

interface Props {
  message: ChatMessage;
}

function PhaseIndicator({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 py-1.5 animate-fade-in">
      <Loader2 className="w-3 h-3 text-[#6ee7b7]/60 animate-spin" />
      <span className="text-[11px] font-mono text-[#6ee7b7]/50">{message}</span>
    </div>
  );
}

export function AgentMessage({ message }: Props) {
  const blocks = message.blocks;
  const rendered: React.ReactNode[] = [];
  const toolStarts = new Map<number, ToolStartEvent>();

  // Collect the latest progress event per tool index
  const latestProgress = new Map<number, ToolProgressEvent>();
  for (const b of blocks) {
    if (b.type === 'tool_progress') {
      latestProgress.set(b.data.index, b.data);
    }
  }

  // Check if turn is complete to suppress trailing phase indicators
  const isComplete = blocks.some((b) => b.type === 'turn_complete' || b.type === 'error');

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    switch (block.type) {
      case 'phase': {
        // Only show the phase if it's the last phase event and the turn isn't complete
        const isLastPhase = !blocks.slice(i + 1).some((b) => b.type === 'phase');
        if (isLastPhase && !isComplete) {
          rendered.push(<PhaseIndicator key={`phase-${i}`} message={block.data.message || block.data.phase} />);
        }
        break;
      }

      case 'thinking':
        rendered.push(<ThinkingBlock key={`think-${i}`} data={block.data} />);
        break;

      case 'tool_start':
        toolStarts.set(block.data.index, block.data);
        {
          const matchingResult = blocks.find(
            (b): b is { type: 'tool_result'; data: ToolResultEvent } =>
              b.type === 'tool_result' && b.data.index === block.data.index,
          );
          rendered.push(
            <ToolBlock
              key={`tool-${block.data.index}`}
              start={block.data}
              result={matchingResult?.data}
              progress={latestProgress.get(block.data.index)}
            />,
          );
        }
        break;

      case 'tool_result':
        break;

      case 'tool_progress':
        // Handled above via latestProgress map — ToolBlock re-renders as blocks update
        break;

      case 'summary':
        rendered.push(<SummaryBlock key={`summary-${i}`} data={block.data} />);
        break;

      case 'chitchat':
        rendered.push(<ChitchatBlock key={`chitchat-${i}`} data={block.data} />);
        break;

      case 'reflection':
        rendered.push(<ReflectionNotice key={`reflect-${i}`} data={block.data} />);
        break;

      case 'reflection_addendum':
        rendered.push(<ReflectionAddendum key={`addendum-${i}`} data={block.data} />);
        break;

      case 'turn_complete':
        if (block.data.total_tools > 0) {
          rendered.push(<TurnCompleteBlock key={`complete-${i}`} data={block.data} />);
        }
        break;

      case 'error':
        rendered.push(
          <p key={`error-${i}`} className="text-xs text-red-300/70">
            Agent error: {block.data.error}
          </p>,
        );
        break;

      // ---- Goal-driven convergence events ----
      case 'goal_extracted':
        rendered.push(<GoalExtractedBlock key={`goal-${i}`} data={block.data} />);
        break;

      case 'analytical_estimate':
        rendered.push(<AnalyticalEstimateBlock key={`analytical-${i}`} data={block.data} />);
        break;

      case 'convergence_step': {
        // Only show the latest convergence step (not all intermediate ones)
        const isLatestStep = !blocks.slice(i + 1).some((b) => b.type === 'convergence_step');
        const hasAnswer = blocks.some((b) => b.type === 'goal_satisfied');
        if (isLatestStep && !hasAnswer) {
          rendered.push(<ConvergenceStepBlock key={`conv-step-${i}`} data={block.data} />);
        }
        break;
      }

      case 'goal_satisfied':
        rendered.push(<GoalSatisfiedBlock key={`goal-sat-${i}`} data={block.data} />);
        break;

      case 'convergence_adapting': {
        // Only show the latest adapting event
        const isLatestAdapt = !blocks.slice(i + 1).some((b) => b.type === 'convergence_adapting');
        const goalFound = blocks.some((b) => b.type === 'goal_satisfied');
        if (isLatestAdapt && !goalFound) {
          rendered.push(<ConvergenceAdaptingBlock key={`conv-adapt-${i}`} data={block.data} />);
        }
        break;
      }

      case 'convergence_stalled':
        rendered.push(<ConvergenceStalledBlock key={`conv-stall-${i}`} data={block.data} />);
        break;

      case 'convergence_exhausted':
        rendered.push(<ConvergenceExhaustedBlock key={`conv-exhaust-${i}`} data={block.data} />);
        break;
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-start gap-3">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#6ee7b7]/20 to-cyan-400/20 border border-[#6ee7b7]/15 flex items-center justify-center shrink-0 mt-0.5">
          <Microscope className="w-3.5 h-3.5 text-[#6ee7b7]" />
        </div>
        <div className="flex-1 min-w-0 space-y-3">
          <p className="text-[11px] font-mono text-[#6ee7b7]/50">Triality Agent</p>
          {rendered}
        </div>
      </div>
    </div>
  );
}
