import numpy as np
from numpy import linalg as LA

def value_iteration(env, theta, discount_factor):
    cc, P = 0, env.P
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + discount_factor * V[next_state] * (not done))
        cc += 1
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi = np.argmax(Q, axis=1)
    return pi, V, Q, cc

def policy_evaluation(pi, env, theta, discount_factor):
    P = env.P
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi[s]]:
                V[s] += prob * (reward + discount_factor * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V

def policy_improvement(V, env, discount_factor):
    P = env.P
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + discount_factor * V[next_state] * (not done))
    new_pi = np.argmax(Q, axis=1)
    return new_pi, Q

def policy_iteration(env, theta, discount_factor):
    cc, P = 0, env.P
    pi = np.random.choice(tuple(P[0].keys()), len(P))
    while True:
        old_pi = pi.copy()
        V = policy_evaluation(pi, env, theta, discount_factor)
        pi, Q = policy_improvement(V, env, discount_factor)
        cc += 1
        if np.all(old_pi == pi):
            break
    return pi, V, Q, cc

def policy_eval(env, policy, theta, discount_factor, k=10000000, init=None):
    v = np.zeros(env.nS) if init is None else init
    cc = 0
    for i in range(k):
        value_fc = np.zeros(env.nS)
        for s in range(env.nS):
            r_pi = np.dot(policy[s, :], env.Rmat[s, :])
            pv = np.dot(env.Pmat[s, :, :].T, v)
            p_pi = np.dot(pv, policy[s, :])
            value_fc[s] = r_pi + discount_factor * p_pi
        delta = LA.norm(value_fc - v, np.inf)
        v[:] = value_fc
        cc += 1
        if delta < theta:
            break
    return v, cc


def modified_policy_iteration(env, k, theta, discount_factor):
    v = np.zeros(env.nS)
    threshold = (theta * (1 - discount_factor))/(2 * discount_factor)
    counter = 0
    while True:
        q = np.zeros([env.nS, env.nA])
        for a in range(env.nA):
            q[:, a] = env.Rmat[:, a] + discount_factor * np.dot(env.Pmat[:, :, a], v)
        greedy_v = np.max(q, -1)
        best_action = np.argmax(q, -1)
        policy = np.eye(env.nA)[best_action]
        if LA.norm(v - greedy_v, np.inf) <= threshold:
            return policy.argmax(axis=1), greedy_v, q, counter
        else:
            v, cc = policy_eval(env, policy, theta=theta, discount_factor=discount_factor, k=k,  init=greedy_v)
            counter += 1