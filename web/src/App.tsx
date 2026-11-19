import { useEffect, useState } from "react";
import { api } from "./api";
import { LeducPage } from "./components/LeducPage";
import { PushFoldPage } from "./components/PushFoldPage";
import { SolverPage } from "./components/SolverPage";

type Tab = "leduc" | "pushfold" | "solver";
const TABS: { id: Tab; label: string }[] = [
  { id: "leduc", label: "Leduc Hold'em" },
  { id: "pushfold", label: "Push/fold Hold'em" },
  { id: "solver", label: "Solver" },
];
const REPO_URL = "https://github.com/Breyes05/poker-bot";

function tabFromHash(): Tab {
  const id = window.location.hash.slice(1);
  return TABS.some((t) => t.id === id) ? (id as Tab) : "leduc";
}

type ServerState = "checking" | "waking" | "ready" | "down";

/** Ping /health until the API answers. Free hosting sleeps when idle, so the first
 * request after a quiet spell can take up to a minute. */
function useServer(): ServerState {
  const [state, setState] = useState<ServerState>("checking");
  useEffect(() => {
    let cancelled = false;
    const started = Date.now();
    const slow = setTimeout(() => !cancelled && setState((s) => (s === "checking" ? "waking" : s)), 1500);
    async function poll() {
      while (!cancelled) {
        try {
          await api.health();
          if (!cancelled) setState("ready");
          return;
        } catch {
          if (Date.now() - started > 90_000) {
            if (!cancelled) setState("down");
            return;
          }
          await new Promise((r) => setTimeout(r, 3000));
        }
      }
    }
    poll();
    return () => {
      cancelled = true;
      clearTimeout(slow);
    };
  }, []);
  return state;
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [showNash, setShowNash] = useState(false);
  const server = useServer();

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>pokerbot</h1>
          <p>Play heads-up poker against strategies solved with counterfactual regret minimization.</p>
        </div>
        <nav className="tabs" aria-label="Games">
          {TABS.map((t) => (
            <a
              key={t.id}
              href={`#${t.id}`}
              className={`tab ${tab === t.id ? "tab-active" : ""}`}
              aria-current={tab === t.id ? "page" : undefined}
            >
              {t.label}
            </a>
          ))}
          {tab !== "solver" && (
            <label className="toggle">
              <input
                id="show-nash"
                type="checkbox"
                checked={showNash}
                onChange={(e) => setShowNash(e.target.checked)}
              />
              <span>Show Nash strategy</span>
            </label>
          )}
        </nav>
      </header>

      <main className="main">
        {server === "ready" ? (
          <>
            {tab === "leduc" && <LeducPage showNash={showNash} />}
            {tab === "pushfold" && <PushFoldPage showNash={showNash} />}
            {tab === "solver" && <SolverPage />}
          </>
        ) : (
          <div className="waking" role="status">
            {server === "down" ? (
              <>
                <h2>The server isn't responding</h2>
                <p>Try reloading in a minute.</p>
              </>
            ) : (
              <>
                <div className="spinner" aria-hidden="true" />
                <h2>{server === "waking" ? "Waking up the server…" : "Connecting…"}</h2>
                {server === "waking" && (
                  <p>It runs on free hosting that sleeps when idle, so this can take up to a minute.</p>
                )}
              </>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          The bot plays a Nash equilibrium: no strategy can beat it in the long run, though it
          won't exploit your mistakes either. Strategies come from CFR+ and are within 0.1
          milli-big-blinds per hand of exact equilibrium.
        </p>
        <a href={REPO_URL} target="_blank" rel="noreferrer">
          Source on GitHub
        </a>
      </footer>
    </div>
  );
}
