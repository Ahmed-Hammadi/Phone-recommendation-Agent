import { useState, useMemo, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { getApiBaseUrl } from "@/utils/env";
import { getPersistedSessionId, resetSessionId } from "@/utils/session";
import { createId } from "@/utils/id";
import type { AgentResponse, ChatMessage } from "@/types/chat";

interface UseAgentChatOptions {
  onMessage?: (message: ChatMessage) => void;
}

async function postAgentChat(payload: { query: string; sessionId: string; includeReasoning: boolean; signal?: AbortSignal }): Promise<AgentResponse> {
  const response = await fetch(`${getApiBaseUrl()}/agent/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: payload.query,
      session_id: payload.sessionId,
      include_reasoning: payload.includeReasoning,
    }),
    signal: payload.signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Request failed");
  }

  return (await response.json()) as AgentResponse;
}

export function useAgentChat({ onMessage }: UseAgentChatOptions = {}) {
  const [sessionId, setSessionId] = useState<string>(() => getPersistedSessionId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [reasoning, setReasoning] = useState<AgentResponse["reasoning"]>(undefined);

  const mutation = useMutation({
    mutationFn: postAgentChat,
    onSuccess: (data, variables) => {
      if (data.session_id) {
        setSessionId(data.session_id);
      }
      const assistantMessage: ChatMessage = {
        id: createId(),
        role: "assistant",
        content: data.response,
        createdAt: Date.now(),
      };
      setMessages(prev => [...prev, assistantMessage]);
      setReasoning(data.reasoning);
      onMessage?.(assistantMessage);
    },
  });

  const sendMessage = useCallback(
    (content: string, options: { includeReasoning?: boolean } = {}) => {
      const userMessage: ChatMessage = {
        id: createId(),
        role: "user",
        content,
        createdAt: Date.now(),
      };
      setMessages(prev => [...prev, userMessage]);
      onMessage?.(userMessage);

      mutation.reset();
      return mutation.mutateAsync({
        query: content,
        sessionId,
        includeReasoning: options.includeReasoning ?? false,
      });
    },
    [mutation, onMessage, sessionId]
  );

  const resetConversation = useCallback(() => {
    const freshSession = resetSessionId();
    setSessionId(freshSession);
    setMessages([]);
    setReasoning(undefined);
    mutation.reset();
  }, [mutation]);

  const state = useMemo(
    () => ({
      sessionId,
      messages,
      reasoning,
      isLoading: mutation.isPending,
      error: mutation.error instanceof Error ? mutation.error : null,
    }),
    [messages, mutation.error, mutation.isPending, reasoning, sessionId]
  );

  return {
    ...state,
    sendMessage,
    resetConversation,
  };
}
