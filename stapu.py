"""
This script will import a simultaneouse task allocation and planning framework
and solve the problem with a Team MDP model
"""

import rust_motap
from rust_motap import stapu as stapulib


NUM_TASKS = 9
NUM_AGENTS = 2


agent = rust_motap.Agent(0, list(range(5)), [1, 2])
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
team = rust_motap.Team()
for i in range(0, NUM_AGENTS):
    team.add_agent(agent.clone())


def construct_message_sending_task(r):
    task = rust_motap.DFA(list(range(0, 6)), 0, [2 + r + 1], [2 + r + 3], [2 + r + 2])
    for w in ["", "send", "ready", "exit"]:
        task.add_transition(0, w, 1)
    task.add_transition(0, "init", 2)
    task.add_transition(1, "init", 2)
    for w in ["", "send", "ready", "exit"]:
        task.add_transition(1, w, 1)
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
mission = rust_motap.Mission()
for k in range(0, NUM_TASKS):
    mission.add_task(construct_message_sending_task(k))

# do some operation to check that this mission is being constructed correctly. 
mission.print_task_transitions()
team.print_initial_states()


if __name__ == "__main__":

    # construct the initial state of the Team MDP model
    initial_state = stapulib.construct_initial_state(mission, team, NUM_TASKS, 0) 
    #initial_state.print_state()

    # Test get the available actions for the initial state
    actions = stapulib.get_available_actions(initial_state, team, mission, NUM_AGENTS, NUM_TASKS)
    #new_states = []
    #for action in actions:
    #    action.print_action()
    #    transitions = stapulib.transitions(initial_state, action, team, mission)
    #    for (s, p) in transitions:
    #        print(s.state_repr(), p)
    #        new_states.append(s)
    #
    #print("finished initial test")
    #for state in new_states:
    #    state.print_state()
    #    actions = stapulib.get_available_actions(state, team, mission, NUM_AGENTS, NUM_TASKS)
    #    for action in actions:
    #        action.print_action()

    team_mdp = stapulib.build_model(initial_state, team, mission, NUM_AGENTS, NUM_TASKS);
    #team_mdp.print_transitions()
    #team_mdp.print_rewards()
    momdp = stapulib.convert_to_multobj_mdp(team_mdp, team, mission, NUM_AGENTS, NUM_TASKS);
    #momdp.print_transitions()
    #momdp.print_rewards()
    target = [-150.] * NUM_AGENTS + [0.90] * NUM_TASKS
    eps = 1e-5
    (mu, r) = rust_motap.multiobjective_scheduler_synthesis(
        eps, target, momdp, NUM_AGENTS, NUM_TASKS
    )
    momdp.print_model_size()