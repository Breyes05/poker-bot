# pokerbot

Game-theoretic poker agents built from scratch.

## Roadmap

Goal: a game-agnostic CFR engine for three poker variants, checked against known
results and playable in the browser. Each milestone ends with a "done when" check.

### M0 — Warm-up: regret matching (an evening)
- [x] Rock-paper-scissors self-play with regret matching (`notebooks/` is fine)
- [x] Understand why the *average* strategy converges and the current one doesn't
- **Done when:** the average strategy converges to (1/3, 1/3, 1/3)
- Reading: Neller & Lanctot, *An Introduction to CFR*, sections 1–2

### M1 — Kuhn poker + vanilla CFR (a weekend)
- [x] Finalize the `Game` interface in `games/base.py` (state, legal actions, current player,
      chance outcomes, info-set key, terminal utility)
- [x] `games/kuhn.py`
- [x] `solvers/cfr.py`: recursive CFR with regret + average-strategy tables
- [x] `scripts/train.py --game kuhn` saves the average strategy to `results/`
- [x] Tests: rule tests (legal actions, payoffs for every terminal history)
- **Done when:** the game value is within 1e-3 of −1/18 and the strategy lies in Kuhn's
  one-parameter equilibrium family (test it)
- Reading: Neller & Lanctot section 3; Kuhn (1950)

### M2 — Exploitability (about a week)
- [x] `eval/exploitability.py`: best response against a fixed strategy, for each player
- [x] Report exploitability in mbb/hand
- [x] Log exploitability during training and plot it (`viz` extra)
- **Done when:** Kuhn exploitability falls toward 0 as iterations go up; a uniform random
  strategy gives a clearly positive value
- Reading: Lanctot thesis (best-response chapter); Johanson et al. 2011

### M3 — Leduc Hold'em + MCCFR (1–2 weeks)
- [ ] `games/leduc.py`: 6-card deck, 2 betting rounds, public card, raise cap
- [ ] Vanilla CFR runs on Leduc unchanged (proves the interface is game-agnostic)
- [ ] `solvers/mccfr.py`: external-sampling MCCFR (outcome sampling optional)
- [ ] Plot exploitability vs. wall-clock time: CFR vs. MCCFR
- **Done when:** your numbers line up with OpenSpiel's Leduc exploitability for similar
  iteration counts
- Reading: Southey et al. 2005 (rules); Lanctot et al. 2009

### M4 — Push/fold No-Limit Hold'em (1–2 weeks)
- [ ] Fast 7-card hand evaluator (write your own; benchmark it)
- [ ] Preflop equity table for the 169 hand classes (compute once, cache to disk)
- [ ] `games/pushfold.py`: heads-up, configurable stack depth in big blinds
- [ ] Solve for several stack depths (e.g. 5, 10, 15, 20 BB)
- [ ] Render the strategy as a 13×13 hand grid
- **Done when:** your charts match published heads-up push/fold Nash charts closely
- Reading: Cactus Kev evaluator write-up; *The Mathematics of Poker* (jam-or-fold chapters)

### M5 — Performance (a few days)
- [ ] Profile training (`cProfile` / `py-spy`); write down the hot spots
- [ ] Vectorize the hot paths with NumPy (e.g. array-backed regret tables)
- [ ] Benchmark script with before/after numbers recorded in this README
- **Done when:** a measurable speedup, and all correctness tests still pass

### M6 — Playable demo (1–2 weeks)
- [ ] `api/`: FastAPI service loading saved strategies (no solving at request time)
      - endpoints: new hand, take action, bot action, show strategy for current info set
- [ ] `web/`: small React/TypeScript front end to play Leduc and push/fold vs. the bot
- [ ] "Show bot's strategy" toggle (action probabilities for the current spot)
- [ ] Deploy (Render / Fly.io / Vercel); add the link here
- **Done when:** a stranger can open the link and play a full hand

### M7 — Polish (a few days)
- [ ] GitHub Actions: `ruff`, `mypy`, `pytest` on every push
- [ ] Convergence regression test in CI (exploitability must stay below a threshold)
- [ ] Dockerfile for the API
- [ ] Type hints throughout
- [ ] README: demo GIF, architecture diagram, convergence plots, "how it works",
      benchmark table, results
- [ ] `docs/devlog.md`: bugs you hit and how you found them (interview material)

### Out of scope (on purpose)
Full no-limit Hold'em with abstraction, Deep CFR, 3+ players. Revisit only after M7.

## Setup
```bash
uv sync
uv run pytest
uv run python scripts/train.py --game kuhn --iterations 10000
```

## Layout
```
src/pokerbot/
  games/     game definitions (state, legal actions, payoffs, info sets)
  solvers/   CFR variants
  eval/      exploitability, head-to-head matches
api/         FastAPI service serving trained strategies (M6)
web/         browser front end (M6)
scripts/     training and evaluation entry points
tests/       unit tests
results/     saved strategies and plots (gitignored)
```

## Results
_TBD_
