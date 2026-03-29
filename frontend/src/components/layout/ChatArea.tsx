import { useEffect, useRef } from 'react';
import { UserMessage } from '../chat/UserMessage';
import { AgentMessage } from '../chat/AgentMessage';
import { WelcomeMessage } from '../chat/WelcomeMessage';
import type { ChatMessage, Scenario } from '../../types';

interface Props {
  messages: ChatMessage[];
  scenarios: Scenario[];
  onSelectScenario: (scenario: Scenario) => void;
}

export function ChatArea({ messages, scenarios, onSelectScenario }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto" id="chat-container">
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-5">
        <WelcomeMessage scenarios={scenarios} onSelectScenario={onSelectScenario} />

        {messages.map((msg) =>
          msg.role === 'user' ? (
            <UserMessage key={msg.id} text={msg.text!} />
          ) : (
            <AgentMessage key={msg.id} message={msg} />
          ),
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
