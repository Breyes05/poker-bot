def regret_matching(regrets: list):
    # get rid of negative values, calc prob w/ positive regrets
    pos_regrets_arr = [x if x > 0 else 0 for x in regrets]
    sum_of_pos = sum(pos_regrets_arr)
    if sum_of_pos == 0:
        return [1 / len(regrets) for _ in pos_regrets_arr]
    else:
        return [x / sum_of_pos for x in pos_regrets_arr]

def average_strategy_helper(strategies: list):
    # take the summed strategies and returns normalized
    sum_of_strats = sum(strategies)
    if sum_of_strats == 0:
        return [1 / len(strategies) for _ in strategies]
    else:
        return [x / sum_of_strats for x in strategies]