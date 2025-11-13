import type { AgentReasoning } from "@/types/chat";
import clsx from "clsx";

interface ReasoningPanelProps {
  reasoning?: AgentReasoning;
  isOpen: boolean;
}

export function ReasoningPanel({ reasoning, isOpen }: ReasoningPanelProps) {
  if (!isOpen) {
    return null;
  }

  if (!reasoning) {
    return (
      <div className="border-t border-white/10 bg-white/5 px-10 py-6 text-sm text-slate-300">
        Reasoning data is unavailable for this turn.
      </div>
    );
  }

  const badge = (label: string, tone: "base" | "accent" = "base") => (
    <span
      key={label}
      className={clsx(
        "mr-2 inline-flex items-center rounded-full px-3 py-1 text-xs",
        tone === "accent" ? "bg-midnight-500/40 text-sky-200" : "bg-white/10 text-slate-200"
      )}
    >
      {label}
    </span>
  );

  return (
    <div className="border-t border-white/10 bg-white/5 px-10 py-6 text-sm text-slate-200">
      <div className="mb-3 flex flex-wrap gap-2">
        {reasoning.intent && badge(`Intent: ${reasoning.intent}`, "accent")}
        {reasoning.context_used && badge("Used context", "accent")}
        {Array.isArray(reasoning.tools_selected) && reasoning.tools_selected.map(tool => badge(tool))}
      </div>
      {Array.isArray(reasoning.requirements_detected) && reasoning.requirements_detected.length > 0 && (
        <p className="mb-2 text-xs text-slate-400">
          Requirements: {reasoning.requirements_detected.join(", ")}
        </p>
      )}
      {reasoning.matched_phone_name && (
        <p className="text-xs text-slate-400">Matched phone: {reasoning.matched_phone_name}</p>
      )}
    </div>
  );
}
