import { ReactNode } from "react";
import { twMerge } from "tailwind-merge";

interface AppShellProps {
  sidebar: ReactNode;
  children: ReactNode;
  className?: string;
}

export function AppShell({ sidebar, children, className }: AppShellProps) {
  return (
    <div className={twMerge("relative min-h-screen overflow-hidden", className)}>
      <div className="pointer-events-none fixed inset-0 z-0">
        <div className="absolute inset-0 bg-neon-grid opacity-60" />
        <div className="absolute inset-0" style={{ background: "radial-gradient(circle at top, rgba(94, 234, 212, 0.2), transparent 55%)" }} />
      </div>

      <div className="relative z-10 flex h-screen flex-col sm:flex-row gap-6 p-6">
        <aside className="glass-panel w-full sm:w-80 rounded-3xl p-6 backdrop-blur-2xl">
          {sidebar}
        </aside>
        <main className="glass-panel flex-1 rounded-3xl backdrop-blur-2xl border border-white/10 overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}
