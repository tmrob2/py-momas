"""
This script will consist of constructing a MAMDP model
"""
from codecs import replace_errors
import rust_mamdp
from itertools import combinations, permutations, product

NUM_TASKS = 3
NUM_AGENTS = 2

# define a set of global action

agent = rust_mamdp.Agent(0, list(range(5)), [1, 2])
# for this experiment we just want some identical copies of the agents
# define agent transitions
agent.add_transition(0, 0, [(0, 0.01, ""), (1, 0.99, "init")])
agent.add_transition(1, 0, [(2, 1., "ready")])
agent.add_transition(2, 0, [(3, 0.99, "send"), (4, 0.01, "exit")])
agent.add_transition(2, 1, [(4, 1.0, "exit")])
agent.add_transition(3, 0, [(2, 1.0, "ready")])
agent.add_transition(4, 0, [(0, 1.0, "")])
# define rewards
agent.add_reward(0, 0, -1)
agent.add_reward(1, 0, -1)
agent.add_reward(2, 0, -1)
agent.add_reward(2, 1, -1)
agent.add_reward(3, 0, -1)
agent.add_reward(4, 0, -1)

# Specify the agents
team = rust_mamdp.Team()
for i in range(0, NUM_AGENTS):
    team.add_agent(agent.clone())


def construct_message_sending_task(r):
    task = rust_mamdp.DFA(list(range(0, 6)), 0, [2 + r + 1], [2 + r + 3], [2 + r + 2])
    task.add_transition(0, "", 1)
    task.add_transition(0, "init", 2)
    task.add_transition(1, "init", 2)
    task.add_transition(1, "", 1)
    for repeat in range(0, r + 1):
        task.add_transition(2 + repeat, "", 2 + repeat)
        task.add_transition(2 + repeat, "init", 2 + repeat)
        task.add_transition(2 + repeat, "ready", 2 + repeat)
        task.add_transition(2 + repeat, "send", 2 + repeat + 1)
        task.add_transition(2 + repeat, "exit", 2 + r + 3)
    for w in ["", "send", "ready", "exit", "init"]:
        task.add_transition(2 + r + 1, w, 2 + r + 2)
        task.add_transition(2 + r + 2, w, 2 + r + 2)
        task.add_transition(2 + r + 3, w, 2 + r + 3)
    return task

# mission is a python owned object which is a collection of tasks, tasks are DFAs
mission = rust_mamdp.Mission()
for k in range(0, NUM_TASKS):
    mission.add_task(construct_message_sending_task(k))

# do some operation to check that this mission is being constructed correctly. 
mission.print_task_transitions()
team.print_initial_states()


if __name__ == "__main__":
    # start by determining the initial transitions of the state
    # define agent 1
    initial_state = rust_mamdp.construct_initial_state(mission, team)
    print("Initial state: ", initial_state)
    #rust_mamdp.find_neighbour_states(
    #    initial_state, [0] * NUM_AGENTS, team, mission, NUM_AGENTS, NUM_TASKS
    #)
    # action_products = rust_mamdp.compute_input_actions(initial_state, team, mission, NUM_AGENTS, NUM_TASKS)
    rust_mamdp.test_get_all_transitions(initial_state, team, mission, NUM_AGENTS, NUM_TASKS);
    # Test a new state
    # let the state be one of the transition states from the initial state 
    # following a task allocation
    test_state = [0, 1, 1, 2, -1, 0, 1]
    rust_mamdp.test_get_all_transitions(test_state, team, mission, NUM_AGENTS, NUM_TASKS)
    mamdp = rust_mamdp.build_model(initial_state, team, mission, NUM_AGENTS, NUM_TASKS);
