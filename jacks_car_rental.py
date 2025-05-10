"""
This is an implementation of the 'Jacks Car Rental' excercise from the
'Reinforcement Learning: An Introduction' Book.
"""
import numpy as np
import matplotlib.pyplot as plt
import ctypes
from scipy.stats import skellam, poisson

# three argument p gets evaluated a bunch so we put in c code
_p3 = ctypes.CDLL('./libp3.so')
_p3.c_prime_given_c.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
_p3.c_prime_given_c.restype = ctypes.c_double

def _acc_c_prime_given_c(c_prime, c, mean_requests, mean_returns, MAX):
    global _p3
    result = _p3.c_prime_given_c(ctypes.c_int(c_prime), ctypes.c_int(c), ctypes.c_int(mean_requests), ctypes.c_int(mean_returns), ctypes.c_int(MAX))
    return result

# some parameters to control the problem setting
MAX_CARS = 20
MAX_MOVE = 5
RENTAL_REQUEST_MEANS = (3, 4)
RENTAL_RETURN_MEANS = (3, 2)
COST_OF_MOVING = -2
PROFIT_FROM_RENTING = 10

# some parameters to control policy iteration
EVALUATION_TOLERANCE = 1e-5
ITERATION_STEPS = 20
DISCOUNT = 0.9


"""
Helper function for the 3-argument p; Computes the state transition
probability at just one of the locations
parameters:
  request_mean: how many requests made on average at this location
  return_mean: how many returns made on average at this location
  new_state: state (no. of cars) for which you want the probability
  old_state_after-action: how many cars are at the location in the
                          morning (after cars were moved but none
                          were rented)
returns:
  probability of this occuring
"""
def partial_p(request_mean: int, return_mean: int, new_state: int, old_state_after_action: int):
    return _acc_c_prime_given_c(new_state, old_state_after_action, request_mean, return_mean, MAX_CARS)

"""
three-argument p
parameters:
   new_state: tuple of car counts at each location
   old_state: car counts at end of previous day
   a: how many cars to move from 1 to 2
returns:
   probability of getting to new state if choosing a in old_state
"""
def p (new_state: (int, int), old_state: (int, int), a: int):
    partial_1 = partial_p(RENTAL_REQUEST_MEANS[0], RENTAL_RETURN_MEANS[0], new_state[0], old_state[0] - a)
    partial_2 = partial_p(RENTAL_REQUEST_MEANS[1], RENTAL_RETURN_MEANS[1], new_state[1], old_state[1] + a)
    # joint probability of the partials is combined probability
    # probability of actual car count change through business is independent
    return partial_1 * partial_2

"""
calculates the probability of renting out a specific number of cars
parameters:
  request_mean: how many cars are requested on average
  max_cars: how many cars can be rented out
  rent_num: number of cars for which you want to know the probability
"""
def prob_rental(request_mean: int, max_cars: int, rent_num: int):
    if rent_num == max_cars:
        # if the maximum number of cars is to be rented out
        # then either that amount or more must be requested
        # Pr(requests >= max_cars)
        # 1 - Pr(requests < max_cars)
        # 1 - Pr(requests <= max_cars - 1)
        if max_cars == 0:
            return 1
        return 1 - poisson._cdf(max_cars-1, request_mean)
    else:
        return poisson._pmf(rent_num, request_mean)

"""
expected reward/cost function of the problem
parameters:
  old_state: state at end of last business day
  a: action taken after business day
return:
  expected profit next day (cost of moving taken into account)
"""
def r(old_state: (int, int), a: int):
    rewards = COST_OF_MOVING * abs(a)
    max_cars = (old_state[0] - a, old_state[1] + a)
    for i in range(0, max_cars[0]+1):
        rewards += prob_rental(RENTAL_REQUEST_MEANS[0], max_cars[0], i) * i * PROFIT_FROM_RENTING
    for i in range(0, max_cars[1]+1):
        rewards += prob_rental(RENTAL_REQUEST_MEANS[1], max_cars[1], i) * i * PROFIT_FROM_RENTING
    return rewards


def q(state: (int, int), action: int, S: list[(int, int)], value: dict[(int, int), int], policy: dict[(int, int), int]):
    v = r(state, action) # immedate reward
    for s_prime in S:
        v += p(s_prime, state, action) * value[s_prime] * DISCOUNT
    return v

def A(state: (int, int)):
    s = []
    for i in range(-MAX_MOVE, MAX_MOVE+1):
        if i <= state[0] and i >= -state[1]:
            s.append(i)
    return s

# Policy Iteration Algorithm (identical to the book, directly transfered to python)

# initial policy and values: no action taken, all 0 values
S: list[(int, int)] = []
for i in range(0, MAX_CARS+1):
    for k in range(0, MAX_CARS+1):
        S.append((k, i))

policy: dict[(int, int), int] = {}
value: dict[(int, int), int] = {}
for s in S:
    policy[s] = 0
    value[s] = 0

policy_history: list[dict[(int, int), int]] = [policy.copy()]
value_history:  list[dict[(int, int), int]] = [value.copy()]

for on_iteration in range(ITERATION_STEPS):
    print(f"Starting Policy Iteration Step {on_iteration}")
    # POLICY EVALUATION
    delta = 0
    _steps = 0
    while True:
        _steps += 1
        delta = 0
        print(f"Start Eval Step {_steps}")
        for (idx, s) in enumerate(S):
            print(f"{int(idx*100/len(S))}%", end="\r")
            v = value[s]
            new_v_temp = r(s, policy[s])
            for s_prime in S:
                new_v_temp += p(s_prime, s, policy[s]) * value[s_prime] * DISCOUNT
            value[s] = new_v_temp
            delta = max(delta, abs(v - value[s]))
        print(f"Evaluation Step: delta = {delta}")
        if delta < EVALUATION_TOLERANCE:
            break
    # POLICY IMPROVEMENT
    print("Policy Improvement")
    policy_stable = True
    for (idx,s) in enumerate(S):
        print(f"{int(idx*100/len(S))}%", end="\r")
        old_action = policy[s]
        # find max action
        best_action = old_action
        best_value = value[s]
        for candidate_action in A(s):
            candidate_value = q(s, candidate_action, S, value, policy)
            if candidate_value > best_value:
                best_value = candidate_value
                best_action = candidate_action
        policy[s] = best_action
        if old_action != best_action:
            policy_stable = False

    policy_history.append(policy.copy)
    value_history.append(value_history)

    arr = np.zeros((21,21))
    arr2 = np.zeros((21,21))
    for s in S:
        arr[s[0], s[1]] = policy[s]
        arr2[s[0], s[1]] = value[s]
    plt.matshow(arr[::-1,:], cmap="hsv")
    plt.colorbar()
    plt.savefig(f"Policy_Figure_{on_iteration}")
    plt.matshow(arr2[::-1,:], cmap="viridis")
    plt.colorbar()
    plt.savefig(f"Value_Figure_{on_iteration}")

    
    if policy_stable:
        print(f"Stopped in Iteration {on_iteration} because of stable policy")
        break

