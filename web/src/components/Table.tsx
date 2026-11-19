import type { ReactNode } from "react";
import type { HandView } from "../api";
import type { Score } from "../useHandSession";
import { CardRow } from "./Card";
import { HandLog, NashBars } from "./Strategy";

interface TableProps {
  hand: HandView | null;
  holeCards: number;
  unit: string; // "chips" or "BB"
  score: Score;
  busy: boolean;
  error: string | null;
  showNash: boolean;
  seatNames: [string, string];
  onAct: (action: string) => void;
  onDeal: () => void;
  onResetScore: () => void;
  controls?: ReactNode; // extra controls above the table (e.g. stack depth)
  aside?: ReactNode; // extra panel under the log (e.g. a range chart)
}

const fmt = (n: number) => (Number.isInteger(n) ? `${n}` : n.toFixed(1));
const signed = (n: number) => `${n > 0 ? "+" : n < 0 ? "−" : ""}${fmt(Math.abs(n))}`;

export function Table(props: TableProps) {
  const { hand, holeCards, unit, score, busy, error, showNash, seatNames } = props;
  const result = hand?.result;

  return (
    <div className="play">
      <section className="table-area">
        {props.controls}
        <div className="felt">
          <div className="seat">
            <div className="seat-head">
              <span className="seat-name">Bot</span>
              {hand && <span className="seat-pos">{seatNames[1 - hand.human_seat]}</span>}
              {hand && <span className="chips">{fmt(hand.bot_chips_in)} in</span>}
            </div>
            <CardRow cards={hand?.bot_cards ?? null} count={holeCards} />
          </div>

          <div className="center">
            <div className="board">
              {hand && hand.board.length > 0 ? (
                <CardRow cards={hand.board} count={hand.board.length} />
              ) : (
                <span className="muted small">{hand ? "No board yet" : ""}</span>
              )}
            </div>
            <div className="pot">
              Pot <strong>{hand ? fmt(hand.pot) : "–"}</strong> {unit}
            </div>
          </div>

          <div className="seat">
            <div className="seat-head">
              <span className="seat-name">You</span>
              {hand && <span className="seat-pos">{seatNames[hand.human_seat]}</span>}
              {hand && <span className="chips">{fmt(hand.your_chips_in)} in</span>}
              {hand?.your_hand_class && <span className="class-tag">{hand.your_hand_class}</span>}
            </div>
            <CardRow cards={hand?.your_cards ?? null} count={holeCards} />
          </div>
        </div>

        <div className="actions">
          {result ? (
            <div className={`result result-${result.winner}`} role="status">
              <strong>
                {result.winner === "you" ? "You win" : result.winner === "bot" ? "Bot wins" : "Split pot"}{" "}
                {signed(result.payoff)} {unit}
              </strong>
              <span>
                {result.outcome === "fold"
                  ? "by fold"
                  : result.your_hand_name
                    ? `${result.your_hand_name} vs ${result.bot_hand_name}`
                    : "at showdown"}
              </span>
            </div>
          ) : (
            hand?.legal_actions.map((a) => (
              <button
                key={a.action}
                className={`btn ${a.action === "f" ? "btn-quiet" : "btn-primary"}`}
                disabled={busy}
                onClick={() => props.onAct(a.action)}
              >
                {a.label}
              </button>
            ))
          )}
          {(result || !hand) && (
            <button className="btn btn-primary" disabled={busy} onClick={props.onDeal}>
              {hand ? "Next hand" : "Deal"}
            </button>
          )}
        </div>
        {error && <p className="error" role="alert">{error}</p>}

        {showNash && hand?.nash && !result && (
          <div className="hint">
            <div className="hint-title">
              Nash strategy for your spot <code>{hand.info_set}</code>
            </div>
            <NashBars game={hand.game} probs={hand.nash} />
          </div>
        )}
      </section>

      <aside className="side">
        <div className="score">
          <div>
            <div className="score-value">{signed(score.net)}</div>
            <div className="muted small">
              {unit} over {score.hands} hand{score.hands === 1 ? "" : "s"}
            </div>
          </div>
          <button className="btn-link" onClick={props.onResetScore} disabled={score.hands === 0}>
            Reset
          </button>
        </div>
        <div className="panel">
          <h3>This hand</h3>
          {hand ? <HandLog game={hand.game} log={hand.log} showNash={showNash} /> : null}
        </div>
        {props.aside}
      </aside>
    </div>
  );
}
