import { useCallback, useRef, useState } from "react";
import { api, ApiError, type GameName, type HandView } from "./api";

export interface Score {
  hands: number;
  net: number; // chips (Leduc) or big blinds (push/fold), from your side
}

/** Deal hands, send actions, and keep a running score. Seats alternate each hand. */
export function useHandSession(game: GameName) {
  const [hand, setHand] = useState<HandView | null>(null);
  const [score, setScore] = useState<Score>({ hands: 0, net: 0 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextSeat = useRef<0 | 1>(0);
  const counted = useRef(new Set<string>());

  const receive = useCallback((view: HandView) => {
    setHand(view);
    if (view.result && !counted.current.has(view.hand_id)) {
      counted.current.add(view.hand_id);
      const payoff = view.result.payoff;
      setScore((s) => ({ hands: s.hands + 1, net: s.net + payoff }));
    }
  }, []);

  const deal = useCallback(
    async (stack?: number) => {
      setBusy(true);
      setError(null);
      try {
        const seat = nextSeat.current;
        receive(await api.newHand(game, seat, stack));
        nextSeat.current = seat === 0 ? 1 : 0;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Couldn't deal a hand.");
      } finally {
        setBusy(false);
      }
    },
    [game, receive],
  );

  const act = useCallback(
    async (action: string) => {
      if (!hand) return;
      setBusy(true);
      setError(null);
      try {
        receive(await api.act(hand.hand_id, action));
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) {
          // The server restarted (free hosting sleeps); start over with a fresh hand.
          setError("That hand expired when the server restarted. Deal a new one.");
          setHand(null);
        } else {
          setError(e instanceof Error ? e.message : "Something went wrong.");
        }
      } finally {
        setBusy(false);
      }
    },
    [hand, receive],
  );

  const resetScore = useCallback(() => setScore({ hands: 0, net: 0 }), []);

  return { hand, score, busy, error, deal, act, resetScore };
}
