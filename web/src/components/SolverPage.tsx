import { useEffect, useState, type FormEvent } from "react";
import { api, type SolveResult } from "../api";
import { RangeGrid } from "./RangeGrid";

const PRESETS = [3, 7.5, 10, 12.5, 15, 20];

export function SolverPage() {
  const [input, setInput] = useState("10");
  const [result, setResult] = useState<SolveResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function solve(stack: number) {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.solvePushFold(stack));
    } catch (e) {
      setError(e instanceof Error ? e.message : "The solve failed.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    solve(10);
  }, []);

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    const stack = Number(input);
    if (!Number.isFinite(stack) || stack < 1.5 || stack > 30) {
      setError("Enter a stack depth between 1.5 and 30 big blinds.");
      return;
    }
    solve(stack);
  }

  return (
    <>
      <p className="intro">
        Pick any stack depth and the server computes the heads-up push/fold Nash equilibrium with
        CFR+. Whole-number depths are precomputed; anything else is solved live.
      </p>
      <form className="solve-form" onSubmit={onSubmit}>
        <label htmlFor="stack-input">Stack depth (BB)</label>
        <input
          id="stack-input"
          type="number"
          inputMode="decimal"
          min={1.5}
          max={30}
          step={0.1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button className="btn btn-primary" type="submit" disabled={busy}>
          {busy ? "Solving…" : "Solve"}
        </button>
        <div className="presets">
          {PRESETS.map((s) => (
            <button
              key={s}
              type="button"
              className="chip-btn"
              disabled={busy}
              onClick={() => {
                setInput(String(s));
                solve(s);
              }}
            >
              {s}
            </button>
          ))}
        </div>
      </form>
      {error && <p className="error" role="alert">{error}</p>}

      {result && (
        <div className={busy ? "solve-result stale" : "solve-result"}>
          <dl className="stats">
            <div>
              <dt>Small blind pushes</dt>
              <dd>{(result.push_pct * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>Big blind calls</dt>
              <dd>{(result.call_pct * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>Exploitability</dt>
              <dd>{result.exploitability_mbb.toFixed(3)} mbb/hand</dd>
            </div>
            <div>
              <dt>Source</dt>
              <dd>
                {result.source === "solved"
                  ? `solved live in ${result.seconds.toFixed(2)} s`
                  : result.source}{" "}
                <span className="muted">({result.iterations} CFR+ iterations)</span>
              </dd>
            </div>
          </dl>
          <div className="charts">
            <RangeGrid
              classes={result.hand_classes}
              values={result.push}
              title={`Small blind: push at ${result.stack} BB`}
            />
            <RangeGrid
              classes={result.hand_classes}
              values={result.call}
              title={`Big blind: call a push at ${result.stack} BB`}
            />
          </div>
          <p className="muted small">
            Pairs on the diagonal, suited hands above it, offsuit below. Numbers in a cell mean the
            hand is mixed: played that percent of the time.
          </p>
        </div>
      )}
    </>
  );
}
