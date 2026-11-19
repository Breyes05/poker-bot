"""Tests for the FastAPI demo service."""

import random

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.hands import LeducHand, PushFoldHand
from api.main import LEDUC, SOLVER, app
from pokerbot.games.pushfold import PushFoldHoldem
from pokerbot.solvers.tree import GameTree

client = TestClient(app)


def deal(game: str, **kwargs) -> dict:
    response = client.post("/hands", json={"game": game, **kwargs})
    assert response.status_code == 200, response.text
    return response.json()


def act(hand: dict, action: str) -> dict:
    response = client.post(f"/hands/{hand['hand_id']}/actions", json={"action": action})
    assert response.status_code == 200, response.text
    return response.json()


def play_out(hand: dict, rng: random.Random) -> dict:
    while hand["result"] is None:
        hand = act(hand, rng.choice(hand["legal_actions"])["action"])
    return hand


# --- Basics ------------------------------------------------------------------------------


def test_health_and_meta():
    assert client.get("/health").json() == {"status": "ok"}
    meta = client.get("/meta").json()
    assert len(meta["hand_classes"]) == 169
    assert meta["precomputed_stacks"] == [float(s) for s in range(2, 31)]
    assert meta["leduc"]["exploitability_mbb"] < 1


def test_cors_allows_the_local_front_end():
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


# --- Leduc ----------------------------------------------------------------------------------


def test_leduc_first_to_act_sees_check_or_bet():
    hand = deal("leduc", seat=0)
    assert [a["label"] for a in hand["legal_actions"]] == ["Check", "Bet"]
    assert hand["log"] == []  # the human acts first, so the bot hasn't moved
    assert sum(hand["nash"].values()) == pytest.approx(1)
    assert hand["info_set"] == f"{hand['your_cards'][0][0]}|"
    assert hand["bot_cards"] is None and hand["board"] == []


def test_leduc_second_seat_sees_the_bots_first_action_with_its_probabilities():
    hand = deal("leduc", seat=1)
    first = hand["log"][0]
    assert first["actor"] == "bot"
    assert first["action"] in ("c", "r")
    assert sum(first["nash"].values()) == pytest.approx(1)


def test_leduc_hands_play_to_a_consistent_finish():
    rng = random.Random(0)
    for i in range(40):
        hand = play_out(deal("leduc", seat=i % 2), rng)
        result = hand["result"]
        # The winner takes the loser's chips: at most 13 (1 ante + 4 + 8).
        assert abs(result["payoff"]) <= 13
        assert hand["pot"] == hand["your_chips_in"] + hand["bot_chips_in"]
        if result["outcome"] == "showdown":
            assert len(hand["bot_cards"]) == 1 and len(hand["board"]) == 1
        else:
            assert hand["bot_cards"] is None  # a folded hand isn't shown
        assert hand["legal_actions"] == []


def test_bot_follows_the_solved_strategy():
    # At "K|" (holding a king, first to act) count how often the bot bets.
    probs = dict(zip(["c", "r"], LEDUC["strategy"]["K|"]))
    rng = random.Random(1)
    bets = total = 0
    while total < 400:
        hand = LeducHand(LEDUC["strategy"], human_seat=1, rng=rng)
        if hand.state.cards[0] // 2 == 2:  # bot holds a king
            bets += hand.log[0]["action"] == "r"
            total += 1
    assert bets / total == pytest.approx(probs["r"], abs=0.07)


# --- Push/fold -------------------------------------------------------------------------------


def test_pushfold_small_blind_fold_loses_half_a_blind():
    hand = deal("pushfold", seat=0, stack=10)
    assert [a["action"] for a in hand["legal_actions"]] == ["f", "p"]
    assert len(hand["your_cards"]) == 2
    hand = act(hand, "f")
    assert hand["result"]["payoff"] == -0.5
    assert hand["result"]["outcome"] == "fold"


def test_pushfold_hands_play_to_a_consistent_finish():
    rng = random.Random(2)
    for i in range(60):
        hand = play_out(deal("pushfold", seat=i % 2, stack=8), rng)
        result = hand["result"]
        assert result["payoff"] in (-8, -1, -0.5, 0, 0.5, 1, 8)
        if result["outcome"] == "showdown":
            assert len(hand["board"]) == 5 and len(hand["bot_cards"]) == 2
            assert result["your_hand_name"] and result["bot_hand_name"]
            cards = hand["your_cards"] + hand["bot_cards"] + hand["board"]
            assert len(set(cards)) == 9  # no card dealt twice


def test_pushfold_bot_never_folds_aces():
    strategy = SOLVER.strategy(10)
    rng = random.Random(3)
    for _ in range(300):
        hand = PushFoldHand(strategy, 10, human_seat=1, rng=rng)
        if hand.view()["log"] and hand.key(0).startswith("AA|"):
            assert hand.log[0]["action"] == "p"


# --- Errors ---------------------------------------------------------------------------------


def test_illegal_actions_are_rejected():
    hand = deal("leduc", seat=0)
    url = f"/hands/{hand['hand_id']}/actions"
    assert client.post(url, json={"action": "f"}).status_code == 409  # nothing to fold to
    assert client.post(url, json={"action": "x"}).status_code == 409
    finished = play_out(hand, random.Random(4))
    assert client.post(url, json={"action": "c"}).status_code == 409  # hand is over
    assert client.get(f"/hands/{finished['hand_id']}").json()["result"] is not None


def test_unknown_hand_and_bad_requests():
    assert client.get("/hands/nope").status_code == 404
    assert client.post("/hands", json={"game": "chess"}).status_code == 422
    assert client.post("/hands", json={"game": "pushfold", "stack": 99}).status_code == 422


# --- On-demand solving ------------------------------------------------------------------------


def test_precomputed_stack_matches_published_chart():
    body = client.post("/solve/pushfold", json={"stack": 15}).json()
    assert body["source"] == "precomputed"
    assert body["push_pct"] == pytest.approx(0.457, abs=0.003)
    assert len(body["push"]) == len(body["call"]) == len(body["hand_classes"]) == 169


def test_custom_stack_is_solved_then_cached():
    first = client.post("/solve/pushfold", json={"stack": 7.5}).json()
    assert first["source"] in ("solved", "cached")
    assert first["exploitability_mbb"] < 1
    second = client.post("/solve/pushfold", json={"stack": 7.5}).json()
    assert second["source"] == "cached"
    assert second["push"] == first["push"]
    # Ranges sit between the neighbouring precomputed depths.
    seven = client.post("/solve/pushfold", json={"stack": 7}).json()
    eight = client.post("/solve/pushfold", json={"stack": 8}).json()
    assert eight["push_pct"] - 0.02 <= first["push_pct"] <= seven["push_pct"] + 0.02


def test_solve_rejects_out_of_range_stacks():
    assert client.post("/solve/pushfold", json={"stack": 1}).status_code == 422
    assert client.post("/solve/pushfold", json={"stack": 31}).status_code == 422


def test_interpolated_tree_payoffs_are_exact():
    fresh = GameTree(PushFoldHoldem(7.5))
    assert np.allclose(SOLVER.tree_for(7.5).payoff, fresh.payoff, atol=1e-12)
