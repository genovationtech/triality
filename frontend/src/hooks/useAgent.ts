import { useState, useCallback, useRef } from 'react';
import type { ChatMessage, SSEBlock } from '../types';

let msgCounter = 0;
function nextId() {
  return `msg-${++msgCounter}-${Date.now()}`;
}

/**
 * Extract a text summary from an agent message's blocks for conversation history.
 * Captures the summary text and key observable values so the LLM has context.
 */
function extractAgentSummary(blocks: SSEBlock[]): string {
  const parts: string[] = [];
  for (const b of blocks) {
    if (b.type === 'thinking') {
      parts.push(`Plan: ${b.data.plan?.join('; ') ?? ''}`);
    } else if (b.type === 'summary') {
      // Truncate to keep context window manageable
      parts.push(b.data.summary?.slice(0, 1200) ?? '');
    } else if (b.type === 'tool_result' && b.data.success) {
      const r = b.data.result;
      // Include observables if present
      const obs = r?.observables;
      if (Array.isArray(obs) && obs.length > 0) {
        const topObs = obs.slice(0, 5).map(
          (o: Record<string, unknown>) => `${o.name}=${o.value} ${o.unit ?? ''}`
        ).join(', ');
        parts.push(`[${b.data.tool} observables: ${topObs}]`);
      }
    } else if (b.type === 'chitchat') {
      parts.push(b.data.response?.slice(0, 500) ?? '');
    } else if (b.type === 'goal_satisfied') {
      // Include goal convergence answer in conversation history
      parts.push(
        `[Goal converged: answer=${b.data.answer} ${b.data.answer_unit ?? ''} ` +
        `(±${b.data.accuracy_pct?.toFixed(2) ?? '?'}%, ${b.data.iterations_used} iterations)]`
      );
    } else if (b.type === 'goal_extracted') {
      parts.push(`[Goal: ${b.data.message}]`);
    }
  }
  return parts.join('\n').slice(0, 2000);
}

export function useAgent() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const appendBlock = useCallback((msgId: string, block: SSEBlock) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === msgId ? { ...m, blocks: [...m.blocks, block] } : m)),
    );
  }, []);

  const runAgent = useCallback(
    async (
      prompt: string,
      token: string,
      model: string,
      scenarioId?: string,
    ) => {
      if (isRunning) return;
      setIsRunning(true);

      // Build conversation history from existing messages
      // Use a snapshot of current messages (before adding the new ones)
      const currentMessages = [...messages];
      const conversationHistory: Array<{ role: string; content: string }> = [];
      for (const msg of currentMessages) {
        if (msg.role === 'user' && msg.text) {
          conversationHistory.push({ role: 'user', content: msg.text });
        } else if (msg.role === 'agent' && msg.blocks.length > 0) {
          const summary = extractAgentSummary(msg.blocks);
          if (summary.trim()) {
            conversationHistory.push({ role: 'assistant', content: summary });
          }
        }
      }

      // Add user message
      const userMsg: ChatMessage = { id: nextId(), role: 'user', text: prompt, blocks: [] };
      const agentMsgId = nextId();
      const agentMsg: ChatMessage = { id: agentMsgId, role: 'agent', blocks: [] };
      setMessages((prev) => [...prev, userMsg, agentMsg]);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await fetch('/api/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            scenario_id: scenarioId ?? null,
            replicate_api_token: token,
            llm_model: model,
            conversation_history: conversationHistory.length > 0 ? conversationHistory : undefined,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const errText = await response.text();
          appendBlock(agentMsgId, { type: 'error', data: { error: errText } });
          return;
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let eventType: string | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ') && eventType) {
              try {
                const data = JSON.parse(line.slice(6));
                const block = { type: eventType, data } as SSEBlock;
                appendBlock(agentMsgId, block);
              } catch {
                // ignore malformed JSON
              }
              eventType = null;
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== 'AbortError') {
          appendBlock(agentMsgId, {
            type: 'error',
            data: { error: err.message },
          });
        }
      } finally {
        setIsRunning(false);
        abortRef.current = null;
      }
    },
    [isRunning, appendBlock, messages],
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isRunning, runAgent, clearMessages };
}
