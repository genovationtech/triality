import { useState, useCallback } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { ChatArea } from './components/layout/ChatArea';
import { ChatInput } from './components/layout/ChatInput';
import { HelpGuide } from './components/pages/HelpGuide';
import { LegalModal } from './components/pages/LegalModal';
import { useCatalog } from './hooks/useCatalog';
import { useAgent } from './hooks/useAgent';
import type { Scenario } from './types';

type Page = 'chat' | 'help';
type LegalPage = 'terms' | 'privacy' | 'disclaimer';

export default function App() {
  const { catalog } = useCatalog();
  const { messages, isRunning, runAgent, clearMessages } = useAgent();

  const [token, setToken] = useState('');
  const [model, setModel] = useState('meta/llama-4-maverick-instruct');
  const [page, setPage] = useState<Page>('chat');
  const [legalModal, setLegalModal] = useState<LegalPage | null>(null);

  const footerStatus = catalog
    ? `${Object.keys(catalog.modules).length} modules \u00B7 ${catalog.tools.length} tools`
    : 'Booting...';

  const handleSend = useCallback(
    (text: string) => {
      if (!token.trim()) return;
      runAgent(text, token, model);
    },
    [token, model, runAgent],
  );

  const handleSelectScenario = useCallback(
    (scenario: Scenario) => {
      if (!token.trim()) return;
      setPage('chat');
      runAgent(scenario.prompt, token, model, scenario.id);
    },
    [token, model, runAgent],
  );

  const goBack = useCallback(() => setPage('chat'), []);

  const handleNavigate = useCallback((target: string) => {
    if (target === 'terms' || target === 'privacy' || target === 'disclaimer') {
      setLegalModal(target);
    } else {
      setPage(target as Page);
    }
  }, []);

  return (
    <div className="bg-grid relative">
      <div className="glow-orb glow-orb-1" />
      <div className="glow-orb glow-orb-2" />

      <div className="flex h-screen relative z-10">
        <Sidebar
          catalog={catalog}
          token={token}
          onTokenChange={setToken}
          model={model}
          onModelChange={setModel}
          onSelectScenario={handleSelectScenario}
          footerStatus={footerStatus}
          onClear={clearMessages}
          onNavigate={handleNavigate}
        />

        <main className="flex-1 flex flex-col overflow-hidden">
          {page === 'chat' && (
            <>
              <Header onHelp={() => setPage('help')} />
              <ChatArea
                messages={messages}
                scenarios={catalog?.scenarios || []}
                onSelectScenario={handleSelectScenario}
              />
              <ChatInput onSend={handleSend} disabled={isRunning || !token.trim()} />
            </>
          )}
          {page === 'help' && <HelpGuide onBack={goBack} />}
        </main>
      </div>

      {/* Legal modal overlay */}
      {legalModal && (
        <LegalModal page={legalModal} onClose={() => setLegalModal(null)} />
      )}
    </div>
  );
}
