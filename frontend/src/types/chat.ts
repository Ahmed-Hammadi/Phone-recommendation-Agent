export type ChatRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
}

export interface AgentReasoning {
  intent?: string;
  phone_detected?: string | null;
  requirements_detected?: string[];
  tools_selected?: string[];
  tools_executed?: number;
  tool_details?: Array<Record<string, unknown>>;
  matched_phone_name?: string | null;
  context_used?: boolean;
}

export interface AgentResponse {
  query: string;
  response: string;
  total_time: number;
  session_id?: string;
  reasoning?: AgentReasoning;
}
