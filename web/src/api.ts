// Typed client for the pokerbot API (see api/main.py).

export const API_URL: string = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export type GameName = "leduc" | "pushfold";

export interface LogEntry {
  actor: "you" | "bot" | "dealer";
  action?: string;
  label?: string;
  round: number;
  nash?: Record<string, number>;
  board?: string;
}

export interface HandResult {
  payoff: number;
  outcome: "fold" | "showdown";
  winner: "you" | "bot" | "split";
  your_hand_name?: string;
  bot_hand_name?: string;
}

export interface HandView {
  hand_id: string;
  game: GameName;
  human_seat: 0 | 1;
  stack?: number;
  your_cards: string[];
  your_hand_class?: string;
  bot_cards: string[] | null;
  board: string[];
  pot: number;
  your_chips_in: number;
  bot_chips_in: number;
  log: LogEntry[];
  legal_actions: { action: string; label: string }[];
  nash: Record<string, number> | null;
  info_set: string | null;
  result: HandResult | null;
}

export interface SolveResult {
  stack: number;
  source: "precomputed" | "cached" | "solved";
  iterations: number;
  exploitability_mbb: number;
  seconds: number;
  push: number[];
  call: number[];
  push_pct: number;
  call_pct: number;
  hand_classes: string[];
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") message = data.detail;
      else if (Array.isArray(data.detail) && data.detail[0]?.msg) message = data.detail[0].msg;
    } catch {
      // keep the generic message
    }
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  newHand: (game: GameName, seat: 0 | 1, stack?: number) =>
    request<HandView>("/hands", { game, seat, ...(stack === undefined ? {} : { stack }) }),
  act: (handId: string, action: string) =>
    request<HandView>(`/hands/${handId}/actions`, { action }),
  solvePushFold: (stack: number) => request<SolveResult>("/solve/pushfold", { stack }),
};
