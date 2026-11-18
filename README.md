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
- [x] `games/leduc.py`: 6-card deck, 2 betting rounds, public card, raise cap
- [x] Vanilla CFR runs on Leduc unchanged (proves the interface is game-agnostic)
- [x] `solvers/mccfr.py`: external-sampling MCCFR (outcome sampling optional)
- [x] Plot exploitability vs. wall-clock time: CFR vs. MCCFR
- **Done when:** your numbers line up with OpenSpiel's Leduc exploitability for similar
  iteration counts
- Reading: Southey et al. 2005 (rules); Lanctot et al. 2009

### M4 — Push/fold No-Limit Hold'em (1–2 weeks)
- [x] Fast 7-card hand evaluator (write your own; benchmark it)
- [x] Preflop equity table for the 169 hand classes (compute once, cache to disk)
- [x] `games/pushfold.py`: heads-up, configurable stack depth in big blinds
- [x] Solve for several stack depths (e.g. 5, 10, 15, 20 BB)
- [x] Render the strategy as a 13×13 hand grid
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
uv run --extra viz python scripts/pushfold_charts.py --stacks 5 10 15 20 --plot
```

## Layout
```
src/pokerbot/
  games/     game definitions (state, legal actions, payoffs, info sets)
  poker/     hand evaluator, 169 hand classes, preflop equity table (cached in data/)
  solvers/   CFR variants
  eval/      exploitability, head-to-head matches
  viz/       13x13 range charts
api/         FastAPI service serving trained strategies (M6)
web/         browser front end (M6)
scripts/     training and evaluation entry points
tests/       unit tests
results/     saved strategies and plots (gitignored)
```

## Results

### Push/fold Hold'em (M4)
Heads-up, no ante, vanilla CFR for 1,000 iterations per stack depth (about 3 minutes each).
"Published" is the heads-up Nash table in PokerStrategy's
[Nash ranges for push-or-fold play](https://www.pokerstrategy.com/strategy/sit-and-go/1779/).

| Stack | Exploitability | SB push (ours / published) | BB call (ours / published) |
|---|---|---|---|
| 5 BB  | 0.47 mbb/hand | 71.4% / 74.1% | 61.9% / 68.6% |
| 10 BB | 0.67 mbb/hand | 58.6% / 56.6% | 37.3% / 38.5% |
| 15 BB | 0.92 mbb/hand | 45.7% / 45.7% | 28.6% / 28.4% |
| 20 BB | 1.20 mbb/hand | 40.0% / 40.9% | 21.8% / 21.7% |

At 15 and 20 BB the charts agree to within a few marginal hands. The published 5 BB row
isn't self-consistent at 5 BB: its push and call ranges are only best responses to each
other at about 4.3-4.5 BB.

The equity table samples 200k boards. Spot checks against exact enumeration are within
0.3%, for example AA vs KK 0.8182 against an exact 0.8195. The evaluator scores about
1.2M seven-card hands per second (numpy, Apple M2 Pro).
