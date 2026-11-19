import { useEffect } from "react";
import { useHandSession } from "../useHandSession";
import { Table } from "./Table";

export function LeducPage({ showNash }: { showNash: boolean }) {
  const session = useHandSession("leduc");
  const { deal } = session;

  useEffect(() => {
    deal();
  }, [deal]);

  return (
    <>
      <p className="intro">
        Six cards (two each of J, Q, K), one private card each and one shared board card. Pair the
        board to win; otherwise the higher card wins. Bets are 2 chips before the board and 4
        after, with at most two raises per round.
      </p>
      <Table
        {...session}
        holeCards={1}
        unit="chips"
        showNash={showNash}
        seatNames={["acts first", "acts second"]}
        onAct={session.act}
        onDeal={() => deal()}
        onResetScore={session.resetScore}
      />
    </>
  );
}
