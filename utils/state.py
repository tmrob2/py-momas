from itertools import combinations, permutations, product
import itertools
from typing import List
from utils.numeric import mult

def get_free_agents(state, n):
    """
    :param state: A state should be a tuple 
    S1 x ... x Sn x Q1 x ... x Qm x W1 x ... x Wn
    :param n: is the number of agents

    We are only concerned with the last W1 x ... Wn

    :returns tuple (num agents free, list of which agents are free)
    """

    working = state[-n:]

    # get the locations of 0s in working
    agents_free = [i for (i, x) in enumerate(working) if x == 0]
    return agents_free, len(agents_free)


def construct_product_state_space(*args):
    """
    Each of the input arguments should be a list
    """

    # check that each of the arguments is a list
    for i in args:
        isinstance(i, list)

    return list(product(*args))


def make_allocations(T1, T2):
    return list(zip(list(T1), list(T2)))


# for efficiency this should be returning the transitions
def get_neighbour_vertex(state, action, dfas, n, m):
    """
    Params:
    state is a tuple representing the MAMDP state
    action is a list of integers of the action that the robot is taking, agents act asynchronously
    dfas is the set of DFAs representing the tasks
    n is the the number of agents
    m is the number of tasks

    Description:
    Given some state, determine its neighbouring states in the MAMDP

    To get the neighbour vertices, we first need to determine how many agents are not
    working and how many tasks are remaining. 

    The remaining tasks can be determined from how many tasks are not finished i.e.
    if the state is 
    (s1, ..., sn, q1, ..., qm, w1, ..., wn) then locations q1, ..., qm that are in
    q0 \in Qj are those tasks which have not yet begun
    """
    neigbours = []
    # reset the DFAs according to the state
    for (j, dfa) in enumerate(dfas):
        dfa.current_state = state[n + j]

    # for the tasks not yet started determine all the combinations of those tasks to the 
    # agents which are currently not working
    agents_free, num_agents_free = get_free_agents(state, m)
    # get the combinations of the tasks which could be allocated to the free agents in the initial state
    task_permutations = list(permutations(get_tasks_remaining(state, dfas, n, m), num_agents_free))
    possible_agents = list(combinations(range(0, n), num_agents_free))


    


def construct_state(current_state, neighbours, dfas, actions, allocation):
    """
    neigbouring states are the mdp states which 

    If the allocation is not empty then work on an allocation
    otherwise 
    """


def get_tasks_remaining(state, dfas, n, m) -> List[int]:
    """
    For each of the q1, ..., qm \in Q1 x ... x Qm making up the state
    determine if qj has not yet begun. This information is recorded in the DFA for task j

    The return of this function is a list of the tasks indices not yet started
    """
    tasks_not_yet_started = []
    for (k, j) in enumerate(range(n, n+m)):
        if dfas[k].start_state == state[j]:
            tasks_not_yet_started.append(k)
    return tasks_not_yet_started


def get_current_tasks(state, dfas, n, m):
    current_tasks = [] # current tasks being worked on
    for (j, k) in enumerate(range(n, n+m)):
        # If q_k is not in initial, accepting, or rejecting then continue working on the task
        if (dfas[j].start_state != state[k]) and (state[k] not in dfas[j].rej) and (state[k] not in dfas[j].done):
            current_tasks.append(j)
    return current_tasks
            


def construct_initial_state(agents, dfas):
    state = []
    for agent in agents:
        state.append(agent.init_state)
    for dfa in dfas:
        state.append(dfa.initial_state)
    for _ in range(0, len(agents)):
        state.append(0)
    return state

