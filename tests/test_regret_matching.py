# mixed signs, all negative, all zero, sum to zero but not all zero, different array lengths
from pokerbot.solvers.regret_matching import regret_matching

def test_mixed_signs():
    assert regret_matching([2, 1, -1]) == [2/3, 1/3, 0]

def test_all_negative_signs():
    assert regret_matching([-1, -1, -1, -1]) == [0.25, 0.25, 0.25, 0.25]

def test_all_zero():
    assert regret_matching([0, 0, 0, 0, 0]) == [0.2, 0.2, 0.2, 0.2, 0.2]

def test_sum_to_zero():
    assert regret_matching([1, 0, -1]) == [1, 0, 0]