// A list of possible icons for the frontend to use
export type ActivityIcon =
  | "search"
  | "read"
  | "synthesize"
  | "web"
  | "error"
  | "done";

// New Activity interface following the enhanced plan
export interface Activity {
  id: string;
  phase: string;
  title: string;
  details?: string;
  status: "in_progress" | "completed" | "failed" | "skipped";
  timestamp: Date;
  progress?: { current: number; total: number };
  estimatedDuration?: number;
  icon?: string;
  retryable?: boolean;
  importance?: "critical" | "normal" | "optional";
}
