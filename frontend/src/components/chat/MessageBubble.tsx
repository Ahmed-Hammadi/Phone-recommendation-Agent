import type { ChatMessage } from "@/types/chat";
import clsx from "clsx";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  const components: Components = {
    p: ({ children }) => (
      <p className="mb-3 last:mb-0 text-base leading-relaxed">{children}</p>
    ),
    strong: ({ children }) => (
      <strong
        className={clsx(
          "font-semibold",
          isUser ? "text-white" : "text-slate-50"
        )}
      >
        {children}
      </strong>
    ),
    em: ({ children }) => (
      <em className="italic text-slate-200">{children}</em>
    ),
    ul: ({ children }) => (
      <ul className="mb-3 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className="mb-3 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>
    ),
    li: ({ children }) => <li className="leading-relaxed">{children}</li>,
    a: ({ children, href }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="font-medium text-sky-300 underline decoration-sky-200/70 underline-offset-4 hover:text-sky-100"
      >
        {children}
      </a>
    ),
    code: ({ children }) => (
      <code
        className={clsx(
          "rounded-md px-1.5 py-0.5 text-[0.9rem]",
          isUser ? "bg-black/30 text-white" : "bg-white/10 text-slate-100"
        )}
      >
        {children}
      </code>
    ),
    pre: ({ children }) => (
      <pre
        className={clsx(
          "mb-3 overflow-x-auto rounded-2xl p-4 font-mono text-sm",
          isUser ? "bg-black/40 text-white" : "bg-white/5 text-slate-100"
        )}
      >
        {children}
      </pre>
    ),
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-white/20 pl-4 italic text-slate-200">
        {children}
      </blockquote>
    ),
    table: ({ children }) => (
      <div className="mb-3 overflow-hidden rounded-xl border border-white/10">
        <table className="w-full text-left text-sm text-slate-100">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className="bg-white/10 text-xs uppercase tracking-wide text-slate-200">
        {children}
      </thead>
    ),
    tbody: ({ children }) => <tbody className="divide-y divide-white/10">{children}</tbody>,
    tr: ({ children }) => <tr>{children}</tr>,
    th: ({ children }) => (
      <th className="px-4 py-3 font-medium">{children}</th>
    ),
    td: ({ children }) => <td className="px-4 py-3 align-top">{children}</td>,
  };

  return (
    <div className={clsx("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={clsx(
          "max-w-xl rounded-3xl px-5 py-4 text-sm leading-relaxed shadow-lg backdrop-blur",
          isUser
            ? "bg-gradient-to-r from-midnight-500 to-midnight-300 text-white"
            : "bg-white/8 text-slate-100 border border-white/10"
        )}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={components}
        >
          {message.content.trim()}
        </ReactMarkdown>
      </div>
    </div>
  );
}
