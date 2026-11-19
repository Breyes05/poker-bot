import type { GameName, LogEntry } from "../api";
import { actionLabels } from "../labels";

export function NashBars({
  game,
  probs,
  chosen,
}: {
  game: GameName;
  probs: Record<string, number>;
  chosen?: string;
}) {
  const codes = Object.keys(probs);
  const labels = actionLabels(game, codes);
  return (
    <div className="nash">
      {codes.map((code) => (
        <div key={code} className={`nash-row ${code === chosen ? "nash-chosen" : ""}`}>
          <span className="nash-label">{labels[code]}</span>
          <span className="nash-track">
            <span className="nash-fill" style={{ width: `${probs[code] * 100}%` }} />
          </span>
          <span className="nash-pct">{(probs[code] * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

export function HandLog({
  game,
  log,
  showNash,
}: {
  game: GameName;
  log: LogEntry[];
  showNash: boolean;
}) {
  if (log.length === 0) return <p className="muted small">No actions yet. You act first.</p>;
  return (
    <ol className="log">
      {log.map((entry, i) =>
        entry.actor === "dealer" ? (
          <li key={i} className="log-dealer">
            Board: <strong>{entry.board?.[0]}</strong>
          </li>
        ) : (
          <li key={i} className={`log-${entry.actor}`}>
            <div className="log-line">
              <span className="log-actor">{entry.actor === "you" ? "You" : "Bot"}</span>
              <span>{entry.label}</span>
              {game === "leduc" && <span className="muted small">round {entry.round}</span>}
            </div>
            {showNash && entry.nash && (
              <NashBars game={game} probs={entry.nash} chosen={entry.action} />
            )}
          </li>
        ),
      )}
    </ol>
  );
}
