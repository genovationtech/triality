import { useState, useRef, useCallback } from 'react';
import { ArrowUp } from 'lucide-react';

interface Props {
  onSend: (text: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: Props) {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = useCallback(() => {
    const trimmed = text.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setText('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  }, [text, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    },
    [submit],
  );

  const handleInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${el.scrollHeight}px`;
    }
  };

  return (
    <div className="p-4 pt-2 bg-gradient-to-t from-[#07080a] via-[#07080a]/95 to-transparent">
      <div className="max-w-4xl mx-auto">
        <div className="input-glow rounded-2xl bg-[#161a24]/60 border border-white/[0.06]">
          <div className="flex items-end gap-3 p-3">
            <div className="flex-1">
              <textarea
                ref={textareaRef}
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={handleKeyDown}
                onInput={handleInput}
                placeholder="Describe what you'd like to analyse..."
                rows={1}
                className="w-full bg-transparent text-sm text-white/80 placeholder-white/20 outline-none resize-none px-2 py-2 font-body max-h-32"
              />
            </div>
            <button
              onClick={submit}
              disabled={disabled || !text.trim()}
              className="send-btn w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
            >
              <ArrowUp className="w-4 h-4 text-[#07080a]" />
            </button>
          </div>
        </div>
        <p className="text-[10px] text-white/[0.12] text-center mt-1.5 font-mono">
          8 tools — run, sweep, optimize, compare, UQ, chain, describe, list
        </p>
      </div>
    </div>
  );
}
