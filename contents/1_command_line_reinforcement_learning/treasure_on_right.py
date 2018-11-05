"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

A better explanation is available at
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
"""

"""
Notes:
Unlike policy gradient methods, which attempt to learn functions which directly map an observation to an action, Q-Learning attempts to learn the value of being in a given state, and taking a specific action there.

Q learning helps learn long term expected rewards

In it’s simplest implementation, Q-Learning is a table of values for every state (row) and action (column) possible in the environment. Within each cell of the table, we learn a value for how good it is to take a given action within a given state. In the case of the FrozenLake environment (OpenAI), we have 16 possible states (one for each block), and 4 possible actions (the four directions of movement), giving us a 16x4 table of Q-values. We start by initializing the table to be uniform (all zeros), and then as we observe the rewards we obtain for various actions, we update the table accordingly.
For making updates to Q-table values, we use Bellman equation:
    Q(s,a) = r + γ(max(Q(s’,a’))
This says that the Q-value for a given state (s) and action (a) should represent the current reward (r) plus the maximum discounted (γ) future reward expected according to our own table for the next state (s’) we would end up in. The discount variable allows us to decide how important the possible future rewards are compared to the present reward. By updating in this way, the table slowly begins to obtain accurate measures of the expected future reward for a given action in a given state.

"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible

N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police. There is 0.1 probability of randomness so that agent may be able to explore the world and find robust solutions
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor. discount variable allows us to decide how important the possible future rewards are compared to the present reward.
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.2    # fresh time for one move


def build_q_table(n_states, actions):
    """
    Initialize a zero-valued q-table of states and actions
    """
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    """
    Decide on the next move.
    Act non-greedily every now and then,
    or explore arena if unexplored,
    else choose the state with maximum reward
    """
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value (unexplored arena)
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    # New state and reward obtained
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment is updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def update_q_table(q_table, S, A, S_, R):
    """
    Bellman equation
    """
    is_terminated = False
    q_predict = q_table.loc[S, A]
    if S_ != 'terminal':
        q_target = R + GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
    else:
        q_target = R  # next state is terminal
        is_terminated = True  # terminate this episode

    q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
    return q_table, S_, is_terminated


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_table, S, is_terminated = update_q_table(q_table, S, A, S_, R) # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
