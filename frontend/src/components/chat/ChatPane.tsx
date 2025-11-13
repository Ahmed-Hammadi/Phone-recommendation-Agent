import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { ChatMessage, AgentReasoning } from "@/types/chat";
import { ChatFeed } from "./ChatFeed";
import { ChatComposer } from "./ChatComposer";
import { ReasoningPanel } from "./ReasoningPanel";
import clsx from "clsx";

interface ChatPaneProps {
  messages: ChatMessage[];
  reasoning?: AgentReasoning;
  isSending: boolean;
  onSend: (message: string) => Promise<unknown> | void;
  onReset: () => void;
  hasError: boolean;
}

export function ChatPane({ messages, reasoning, isSending, onSend, onReset, hasError }: ChatPaneProps) {
  const [reasoningVisible, setReasoningVisible] = useState(false);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-white/10 px-10 py-6">
        <div>
          <h2 className="text-lg font-semibold text-white">Conversation Studio</h2>
          <p className="text-sm text-slate-400">Ask about phones, follow up with specs, reviews, or alternatives.</p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setReasoningVisible(prev => !prev)}
            className={clsx(
              "rounded-full border border-white/20 px-4 py-2 text-sm text-slate-200 transition",
              reasoningVisible && "bg-white/10 text-white"
            )}
          >
            {reasoningVisible ? "Hide reasoning" : "Show reasoning"}
          </button>
          <button
            type="button"
            onClick={onReset}
            className="rounded-full border border-white/20 px-4 py-2 text-sm text-slate-200 transition hover:border-white/40"
          >
            Reset
          </button>
        </div>
      </header>

      <AnimatePresence initial={false}>
        {hasError && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="bg-red-500/20 text-red-100 px-10 py-3 text-sm"
          >
            Something went wrong while contacting the agent. Please try again.
          </motion.div>
        )}
      </AnimatePresence>

      <ChatFeed messages={messages} />

      <ReasoningPanel reasoning={reasoning} isOpen={reasoningVisible} />

      <ChatComposer onSend={onSend} isSending={isSending} />
    </div>
  );
}
