use pyo3::prelude::*;
use hashbrown::HashMap;
use crate::dfa::dfa::*;
use crate::agent::agent::*;
use super::action_utils::*;

fn tasks_remaining(state: &[i32], n: usize, m: usize, tasks: &Mission) -> Vec<i32> {
    let mut tasks_not_yet_started = vec![];
    for (k, j) in (n..n + m).enumerate() {
        if tasks.tasks[k].initial_state == state[j] {
            tasks_not_yet_started.push(k as i32);
        }
    }
    tasks_not_yet_started
}

/// Function to construct an initial state for the MAMDP
/// Before using this function the tasks, and agents already need to be described. See Mission
/// and Team structs. 
#[pyfunction]
pub fn construct_initial_state(tasks: &Mission, agents: &Team) -> Vec<i32> {
    let mut state = vec![];
    for i in 0..agents.size {
        state.push(agents.agents[i].init_state);
    }
    for j in 0..tasks.size {
        state.push(tasks.tasks[j].initial_state);
    }
    for _ in 0..agents.size {
        state.push(-1); // represents which task the agent is working on, i.e. the n + m + ith position
        // represents which task the ith agent is working on. If the variable is -1 then the agent
        // is not working on anything
    }
    state
}

fn add_to_product(
    agent_idx: usize,
    acc: Vec<(Vec<(usize, i32, i32, String)>, f64, Vec<String>)>, 
    workingon: &[i32],
    words: &Vec<Vec<String>>,
    probabilities: &Vec<Vec<f64>>,
    b: &[i32]
) -> Vec<(Vec<(usize, i32, i32, String)>, f64, Vec<String>)> {
    let mut acc_new: Vec<(Vec<(usize, i32, i32, String)>, f64, Vec<String>)> = Vec::new();
    if !b.is_empty() {
        // then there is no sprime to add
        // acc is consumed on input into this function
        if acc.is_empty() {
            for (idx, val) in b.iter().enumerate() {
                // for the chosen b value get the corresponding word
                // then compute the corresponding q value
                acc_new.push((
                    vec![(agent_idx, *val, workingon[agent_idx], words[agent_idx][idx].to_string())],
                    probabilities[agent_idx][idx],
                    vec![words[agent_idx][idx].to_string()]
                ));
            }
        } else {
            for (v, p, w) in acc.iter() {
                for (idx, val) in b.iter().enumerate() {
                    let mut new_prod: Vec<(usize, i32, i32, String)> = v.clone();
                    new_prod.push((agent_idx, *val, workingon[agent_idx], words[agent_idx][idx].to_string()));
                    let mut new_words: Vec<String> = w.to_vec();
                    new_words.push(words[agent_idx][idx].to_string());
                    let new_tuple = (
                        new_prod, 
                        probabilities[agent_idx][idx] * p,
                        new_words
                    );
                    acc_new.push(new_tuple);
                }
            }
        }
    } else {
        acc_new = acc;
    }
    
    acc_new
}

pub fn get_available_actions(
    state: &[i32], 
    agents: &Team, 
    mission: &Mission, 
    n: usize, 
    m: usize
) -> Vec<Vec<MultiAgentAction>> {
    // If the agent is not working, then determine which tasks are still remaining
    let mut available_actions: Vec<Vec<MultiAgentAction>> = Vec::new();
    for agent in 0..n {
        let mut agent_actions: Vec<MultiAgentAction> = Vec::new();
        // If an agent is working then no task allocation can occur, otherwise a task allocation     
        if state[n + m + agent] == -1 {
            // get the available tasks
            let tasks = tasks_remaining(&state[..], n, m, mission);
            // what about the case where there are no tasks reamining?
            // then we can do a task allocation=
            if tasks.is_empty() {
                for base_action in agents.agents[agent].available_actions
                    .get(&state[agent]).unwrap().iter() {
                        agent_actions.push(MultiAgentAction {
                            base_action: *base_action,
                            working: false,
                            task_job: state[n + m + agent]
                        });
                    }
            } else {
                for task in tasks.iter() {
                    for base_action in agents.agents[agent].available_actions
                        .get(&state[agent]).unwrap().iter() {
                        agent_actions.push(MultiAgentAction {
                            base_action: *base_action,
                            working: false,
                            task_job: *task
                        });  
                    }
                }
            }
        } else {
            for base_action in agents.agents[agent].available_actions
                .get(&state[agent]).unwrap().iter() {
                    agent_actions.push(MultiAgentAction {
                        base_action: *base_action,
                        working: true,
                        task_job: state[n + m + agent]
                    })
                }
        }
        available_actions.push(agent_actions);
    }
    available_actions
}

fn accumulate_actions(acc: Vec<Vec<MultiAgentAction>>, new_agent_actions: &[MultiAgentAction]) 
    -> Vec<Vec<MultiAgentAction>> {
    let mut new_acc: Vec<Vec<MultiAgentAction>> = Vec::new();
    if acc.is_empty() {
        for action in new_agent_actions.iter() {
            new_acc.push(vec![*action])
        }  
    } else {
        for action_combo in acc.iter() {
            for action in new_agent_actions.iter() {
                // The agents must be working on different tasks, this is one of the 
                // conditions of the emulating the SCPM
                if !action_combo.iter().any(|a| a.task_job == action.task_job) {
                    let mut new_combo = action_combo.to_vec();
                    new_combo.push(*action);
                    new_acc.push(new_combo);
                }
            }
        }
    }
    new_acc
}

fn compute_action_product(actions_per_agent: Vec<Vec<MultiAgentAction>>, n: usize) 
    -> Vec<Vec<MultiAgentAction>>  {
    let mut acc: Vec<Vec<MultiAgentAction>> = Vec::new();
    for agent_idx in 0..n {
        acc = accumulate_actions(acc, &actions_per_agent[agent_idx][..]);
    }
    acc
}

/// Computes the set of input actions for a particular state. The set of input actions
/// which can be taken at a particular state is dependent on the base agent actions
/// available to the agents, and the possible task allocation to agent combinations. 
/// To compute the set of actions available, we construct the cross product of the 
/// set of available actions to each agent, with the restriction that each agent must
/// be working on a different task. 
/// 
/// In the true MAMDP, multiple agents could be working on the same task. 
pub fn compute_input_actions(
    state: &[i32],
    team: &Team, 
    tasks: &Mission,
    n: usize, 
    m: usize
) -> Vec<Vec<MultiAgentAction>> {
    let action_sets = get_available_actions(state, team, tasks, n, m);
    let action_products = compute_action_product(action_sets, n);
    action_products
}

/// Computes the mamdp product transitions for a team of agents and a set of asynchronous actions given some input state
fn product_transition(
    sprimes: Vec<Vec<i32>>,
    workingon: &[i32], 
    words: Vec<Vec<String>>,
    probabilities: Vec<Vec<f64>>
) -> Vec<(Vec<(usize, i32, i32, String)>, f64, Vec<String>)> {
    // if an agent is not working the transition will always be to the same state, 
    // regardless of the action
    let mut acc: Vec<(Vec<(usize, i32, i32, String)>, f64, Vec<String>)> = Vec::new();
    for (idx, k) in sprimes.iter().enumerate() {
        acc = add_to_product(idx, acc, workingon, &words, &probabilities, &k[..]);
    }
    acc 
}

/// A function which computes all the available transitions at a given state for the
/// set of available actions to each agent. The transitions are accumulated in a 
/// a mutable reference `hashbrown::HashMap`.
pub fn get_all_transitions_for_state(
    state: &[i32], 
    actions: Vec<Vec<MultiAgentAction>>, 
    team: &Team, 
    tasks: &Mission,
    n: usize,
    m: usize,
    transitions: &mut HashMap<(Vec<i32>, Vec<MultiAgentAction>), Vec<(Vec<i32>, f64)>>
) {
    let state_ref = &state[..];
    for comb in actions.iter() {
        let transitions_to = transition_to_per_action_comb(state_ref, &comb[..], team, tasks, n, m);
        transitions.insert((state_ref.to_vec(), comb.clone()), transitions_to);
    }
}

#[pyfunction]
pub fn test_get_all_transitions(state: Vec<i32>, team: &Team, tasks: &Mission, n: usize, m: usize) {
    let mut transitions: HashMap<(Vec<i32>, Vec<MultiAgentAction>), Vec<(Vec<i32>, f64)>> = HashMap::new();
    let actions = compute_input_actions(&state[..], team, tasks, n, m);
    get_all_transitions_for_state(&state[..], actions, team, tasks, n, m, &mut transitions);
    for transition in transitions.iter() {
        println!("{:?}", transition);
    }
}

/// For a given multiagent action i.e. `[A1, A2, ... An]` for a set of agents in a team, 
/// compute all the transitions P: (S x Q) x A x (S x Q) -> [0, 1] such that 
/// P((s, q), a, (s', q')) > 0
pub fn transition_to_per_action_comb(
    state: &[i32],
    action: &[MultiAgentAction],
    team: &Team, 
    tasks: &Mission,
    n: usize, 
    m: usize
) -> Vec<(Vec<i32>, f64)> {

    // for each agent follow the instructions of the action,, which may include a task
    // allocation
    let mut sprimes: Vec<Vec<i32>> = vec![Vec::new(); n];
    let mut probabilities: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut words: Vec<Vec<String>> = vec![Vec::new(); n];
    let mut working_on: Vec<i32> = vec![-1; n];
    for agent_idx in 0..n {
        let agent_action: &MultiAgentAction = &action[agent_idx];
        // what is the agent working on
        working_on[agent_idx] = agent_action.task_job;
        let agent_transitions = team.agents[agent_idx]
            .transitions
            .get(&(state[agent_idx], agent_action.base_action))
            .unwrap();
        for (s, p, w) in agent_transitions.iter() {
            sprimes[agent_idx].push(*s);
            probabilities[agent_idx].push(*p);
            words[agent_idx].push(w.to_string());
        }
    }

    // once the agent information per action has been filled in, the next step is to 
    // determine the cross product of these outcomes. 

    let product_transitions = product_transition(
        sprimes, &working_on[..], words, probabilities
    );
    // construct a new state from the transition
    let mut transition_to_pairs: Vec<(Vec<i32>, f64)> = Vec::new();
    for (sprimes, p, _words) in product_transitions.iter() {
        let mut new_state: Vec<i32> = state.to_vec();
        for (agent, s, task, word) in sprimes.iter() {
            new_state[*agent] = *s;
            if *task > -1 {
                let qprime = match tasks.tasks[*task as usize]
                    .transitions
                    .get(&(state[n + *task as usize], word.to_string()))
                    {
                        Some(z) => { z }
                        None => { 
                            panic!("state: {:?}, word: {:?} combination for DFA not found!", 
                                state[n + *task as usize], word)
                            }
                    };
                new_state[n + *task as usize] = *qprime;
                // check if qprime is a member of the tasks done or failed states,
                // if so then the agent will no longer work on this task
                if tasks.tasks[*task as usize].done.contains(qprime) || 
                    tasks.tasks[*task as usize].rejecting.contains(qprime) {
                        new_state[n + m + *agent] = -1;
                    }
                else {
                    new_state[n + m + *agent] = *task
                }
            }
        }
        transition_to_pairs.push((new_state, *p));
    }
    transition_to_pairs
}