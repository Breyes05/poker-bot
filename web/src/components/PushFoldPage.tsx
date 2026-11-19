import { useEffect, useState } from "react";
import { api, type SolveResult } from "../api";
import { useHandSession } from "../useHandSession";
import { RangeGrid } from "./RangeGrid";
import { Table } from "./Table";

const STACKS = [5, 10, 15, 20];

export function PushFoldPage({ showNash }: { showNash: boolean }) {
  const session = useHandSession("pushfold");
  const { deal, hand } = session;
  const [stack, setStack] = useState(10);
  const [charts, setCharts] = useState<SolveResult | null>(null);

  useEffect(() => {
    deal(stack);
    // Deal once on mount; later hands use the stack chosen at that time.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deal]);

  // Fetch the chart for the hand's stack depth when the strategy is visible.
  const handStack = hand?.stack;
  useEffect(() => {
    if (!showNash || handStack === undefined || charts?.stack === handStack) return;
    api.solvePushFold(handStack).then(setCharts).catch(() => setCharts(null));
  }, [showNash, handStack, charts?.stack]);

  const youAreSB = hand?.human_seat === 0;
  const controls = (
    <div className="controls">
      <span className="muted small">Stack depth</span>
      {STACKS.map((s) => (
        <button
          key={s}
          className={`chip-btn ${s === stack ? "chip-active" : ""}`}
          onClick={() => setStack(s)}
        >
          {s} BB
        </button>
      ))}
      {hand && hand.stack !== stack && !hand.result && (
        <span className="muted small">applies from the next hand</span>
      )}
    </div>
  );

  return (
    <>
      <p className="intro">
        Heads-up No-Limit Hold'em where the only options are all-in or fold. The small blind (0.5
        BB) acts first: push or fold. Facing a push, the big blind (1 BB) calls or folds. A call
        deals all five board cards.
      </p>
      <Table
        {...session}
        holeCards={2}
        unit="BB"
        showNash={showNash}
        seatNames={["small blind", "big blind"]}
        onAct={session.act}
        onDeal={() => deal(stack)}
        onResetScore={session.resetScore}
        controls={controls}
        aside={
          showNash && charts && hand ? (
            <div className="panel">
              <RangeGrid
                classes={charts.hand_classes}
                values={youAreSB ? charts.push : charts.call}
                title={`${youAreSB ? "Small blind push" : "Big blind call"} range at ${charts.stack} BB`}
                highlight={hand.your_hand_class}
              />
            </div>
          ) : null
        }
      />
    </>
  );
}
