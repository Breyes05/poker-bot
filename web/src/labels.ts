import type { GameName } from "./api";

/** Human-readable names for a spot's action codes. */
export function actionLabels(game: GameName, codes: string[]): Record<string, string> {
  if (game === "pushfold") {
    return codes.includes("p")
      ? { f: "Fold", p: "Push" }
      : { f: "Fold", c: "Call" };
  }
  const facing = codes.includes("f");
  return { f: "Fold", c: facing ? "Call" : "Check", r: facing ? "Raise" : "Bet" };
}
