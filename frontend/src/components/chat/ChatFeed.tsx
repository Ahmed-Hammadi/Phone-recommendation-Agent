import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { ChatMessage } from "@/types/chat";
import { MessageBubble } from "./MessageBubble";

interface ChatFeedProps {
  messages: ChatMessage[];
}

export function ChatFeed({ messages }: ChatFeedProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [messages]);

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto px-10 py-8 space-y-4">
      <AnimatePresence initial={false}>
        {messages.map(message => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
          >
            <MessageBubble message={message} />
          </motion.div>
        ))}
      </AnimatePresence>
      {!messages.length && (
        <div className="mt-16 text-center text-sm text-slate-400">
          Ask about price, specs, or community sentiment to begin.
        </div>
      )}
    </div>
  );
}
