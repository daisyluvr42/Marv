export type TaskStatus = "active" | "paused" | "completed" | "archived";

export type TaskContextRole = "user" | "assistant" | "system" | "tool";

export type TaskContextEntry = {
  id: string;
  taskId: string;
  sequence: number;
  role: TaskContextRole;
  content: string;
  contentHash: string;
  summary?: string;
  tokenCount: number;
  createdAt: number;
  metadata?: string;
  summarized: boolean;
};

export type TaskContext = {
  taskId: string;
  agentId: string;
  title: string;
  status: TaskStatus;
  parentTaskId?: string;
  scopeId: string;
  createdAt: number;
  updatedAt: number;
  completedAt?: number;
  totalEntries: number;
  totalTokens: number;
};
