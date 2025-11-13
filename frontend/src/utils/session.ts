import { createId } from "@/utils/id";

const STORAGE_KEY = "phone-assistant-session";

export function getPersistedSessionId(): string {
  if (typeof window === "undefined") {
    return createId();
  }

  const existing = window.localStorage.getItem(STORAGE_KEY);
  if (existing) {
    return existing;
  }

  const fresh = createId();
  window.localStorage.setItem(STORAGE_KEY, fresh);
  return fresh;
}

export function resetSessionId(): string {
  const fresh = createId();
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, fresh);
  }
  return fresh;
}
