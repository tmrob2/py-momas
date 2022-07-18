import copy
from utils.state import construct_initial_state, construct_product_state_space, get_current_tasks
from utils.agent import Agent, Actions
from utils.dfa import DFA
import unittest

NUM_TASKS = 2
NUM_AGENTS = 2

# define a set of global actions
actions = Actions([1, 2])

agent = Agent(0, list(range(5)), actions)
# for this experiment we just want some identical copies of the agents
# define agent transitions
agent.add_transition(0, 1, [(0, 0.01), (1, 0.99)])
agent.add_transition(1, 1, [(2, 1.)])
agent.add_transition(2, 1, [(3, 0.99), (4, 0.01)])
agent.add_transition(2, 2, [(4, 1.0)])
agent.add_transition(3, 1, [(2, 1.0)])
agent.add_transition(4, 1, [(0, 1.0)])
# define rewards
agent.add_reward(0, 1, -1)
agent.add_reward(1, 1, -1)
agent.add_reward(2, 1, -1)
agent.add_reward(2, 2, -1)
agent.add_reward(3, 1, -1)
agent.add_reward(4, 1, -1)

# Specify the agents
agents = [copy.deepcopy(agent) for _ in range(0, NUM_AGENTS)]

# define the DFA for a message sending task
def start(**kwargs) -> int:
    if kwargs["word"] == "init":
        return 2
    else:
        return 1

def init(**kwargs) -> int:
    if kwargs["word"] == "init":
        return 2
    else:
        return 1

def send_one_msg(**kwargs):
    if kwargs["word"] not in ["exit", "send"]:
        return kwargs["state"]
    elif kwargs["word"] == "exit":
        return kwargs["fail"]
    else:
        return kwargs["state"] + 1

def finish(**kwargs):
    return kwargs["state"] + 1

def done(**kwargs):
    return kwargs["state"]

def fail(**kwargs):
    return kwargs["state"]


def construct_message_sending_task(r):
    # calculate the rejected state and accepting state from the number of repeats
    R = 2 + r + 3
    F = 2 + r + 2
    accepting = 2 + r + 1
    task = DFA(0, [accepting], [F], [R])
    task.add_state(0, start)
    task.add_state(1, init)
    for r in range(0, r):
        task.add_state(2 + r, send_one_msg)
    task.add_state(2 + r + 1, finish)
    task.add_state(2 + r + 2, done)
    task.add_state(2 + r + 3, fail)
    return task

# Construct the tasks according to the message sending task protocol for iteration in the MDP
# we will have to manually set the current state a lot of the time but at least the DFA contains
# the set of instructions necessary to compute the task transition function
tasks = [construct_message_sending_task(k) for k in range(0, NUM_TASKS)]

class TestStateMethods(unittest.TestCase):
    def test_init_state(self):

        init_state = construct_initial_state(agents, tasks)
        self.assertEqual(init_state, [0] * (2 * len(agents) + len(tasks)))

    def test_current_tasks(self):
        init_state = construct_initial_state(agents, tasks)
        current_tasks = get_current_tasks(init_state, tasks, NUM_AGENTS, NUM_TASKS)
        self.assertEqual(current_tasks, [])


if __name__ == "__main__":
    unittest.main()