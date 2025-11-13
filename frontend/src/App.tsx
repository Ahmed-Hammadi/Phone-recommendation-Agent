import { useEffect } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { ChatPane } from "@/components/chat/ChatPane";
import { useAgentChat } from "@/hooks/useAgentChat";

interface SidebarProps {
  onNewSession: () => void;
}

function Sidebar({ onNewSession }: SidebarProps) {
  return (
    <div className="flex h-full flex-col gap-6 text-slate-200">
      <div className="flex items-center gap-3">
        <img src="/assets/robot-sticker.svg" alt="Robot companion" className="h-12 w-12 drop-shadow-lg" />
        <div>
          <p className="text-xs uppercase tracking-[0.6em] text-slate-400">Companion</p>
          <h2 className="text-lg font-semibold text-white">Copilot Node</h2>
        </div>
      </div>
      <div>
        <p className="text-sm uppercase tracking-[0.4em] text-slate-400">Navigator</p>
        <h1 className="mt-2 text-2xl font-semibold">Phone Intelligence Deck</h1>
        <p className="mt-3 text-sm leading-relaxed text-slate-400">
          Discover live pricing, compare specs, and explore community insights in a single experience.
        </p>
      </div>
      <div className="mt-auto">
        <button
          onClick={onNewSession}
          className="group flex items-center justify-between rounded-full border border-white/10 px-4 py-3 text-sm font-medium text-slate-200 transition hover:border-white/20 hover:bg-white/5"
        >
          Start new session <span className="text-xs text-slate-400">⌘ + N</span>
        </button>
      </div>
    </div>
  );
}

function App() {
  useEffect(() => {
    document.body.classList.add("overflow-hidden");
    return () => document.body.classList.remove("overflow-hidden");
  }, []);

  const { messages, reasoning, isLoading, error, sendMessage, resetConversation } = useAgentChat();

  return (
    <AppShell sidebar={<Sidebar onNewSession={resetConversation} />}>
      <ChatPane
        messages={messages}
        reasoning={reasoning}
        isSending={isLoading}
        onSend={message => sendMessage(message, { includeReasoning: true })}
        onReset={resetConversation}
        hasError={Boolean(error)}
      />
    </AppShell>
  );
}

export default App;
