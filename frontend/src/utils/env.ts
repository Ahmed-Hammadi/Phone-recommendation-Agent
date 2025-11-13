const DEFAULT_API_URL = "http://localhost:8000";

export function getApiBaseUrl(): string {
  return import.meta.env.VITE_API_URL ?? DEFAULT_API_URL;
}
