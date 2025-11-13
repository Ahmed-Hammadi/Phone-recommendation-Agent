import { FormEvent, useState } from "react";
import clsx from "clsx";

interface ChatComposerProps {
  onSend: (message: string) => Promise<unknown> | void;
  isSending?: boolean;
}

export function ChatComposer({ onSend, isSending }: ChatComposerProps) {
  const [draft, setDraft] = useState("");

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed) return;
    try {
      await onSend(trimmed);
      setDraft("");
    } catch (error) {
      console.error("Failed to send message", error);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t border-white/10 px-10 py-6">
      <div className="flex items-center gap-4 rounded-full border border-white/10 bg-white/5 p-3 shadow-inner shadow-white/5">
        <input
          className="flex-1 border-none bg-transparent text-sm text-white placeholder:text-slate-400 focus:outline-none"
          placeholder="Ask something like “Compare the camera performance of Pixel 8 Pro and iPhone 15 Pro Max”"
          value={draft}
          onChange={event => setDraft(event.target.value)}
          disabled={isSending}
        />
        <button
          type="submit"
          className={clsx(
            "rounded-full bg-midnight-500 px-5 py-2 text-sm font-medium text-white shadow-glow transition",
            isSending && "opacity-60"
          )}
          disabled={isSending}
        >
          {isSending ? "Sending…" : "Send"}
        </button>
      </div>
    </form>
  );
}
