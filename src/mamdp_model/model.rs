//! Used to construct the MAMDP model under the following 
//! restrictions:
//! 
//! 1. Each agent must be working on a different task
//! 
//! 2. Each agent works on a task until completion.

use pyo3::prelude::*;
use hashbrown::{HashMap, HashSet};
use std::collections::VecDeque;
use crate::agent::agent::Team;
use crate::dfa::dfa::Mission;
use super::action_utils::MultiAgentAction;
use super::state_utils::*;
use crate::mdp::model::MultiObjectiveMDP;
use crate::reverse_key_value_pairs;

#[pyclass]
pub struct MAMDP {
    pub initial_state: Vec<i32>,
    pub states: Vec<Vec<i32>>,
    // Available actions for a particular state is much more useful then all of the action as a set
    pub actions: HashMap<Vec<i32>, Vec<Vec<MultiAgentAction>>>,
    pub transitions: HashMap<(Vec<i32>, Vec<MultiAgentAction>), Vec<(Vec<i32>, f64)>>,
    pub rewards: HashMap<(Vec<i32>, Vec<MultiAgentAction>), Vec<f64>>,
    action_space: HashSet<Vec<MultiAgentAction>>,
    state_mapping: HashMap<Vec<i32>, i32>,
    action_map: HashMap<i32, Vec<MultiAgentAction>>,
    reverse_action_map: HashMap<Vec<MultiAgentAction>, i32>,
}

// TODO: rewards structure for the MAMDP
// TODO: arbitrary model conversion -> We essentially want some generic multi-agent
//  MDP which we can conduct model checking on with our current algorithms.

impl MAMDP {
    fn new() -> Self {
        MAMDP {
            initial_state: Vec::new(),
            states: Vec::new(),
            actions: HashMap::new(),
            transitions: HashMap::new(),
            rewards: HashMap::new(),
            action_space: HashSet::new(),
            state_mapping: HashMap::new(),
            action_map: HashMap::new(),
            reverse_action_map: HashMap::new(),
        }
    }

    fn insert_init_state(&mut self, init_state: &[i32]) {
        self.initial_state = init_state.to_vec();
    }

    fn insert_transition(
        &mut self,
        sprime: (Vec<i32>, f64), 
        state: &[i32], 
        action: &[MultiAgentAction]) {
        match self.transitions.get_mut(&(state.to_vec(), action.to_vec())) {
            Some(t) => { 
                // the transition already exists and we are just adding to it
                t.push(sprime);
            }
            None => {
                // insert a function
                self.transitions
                    .insert((state.to_vec(), action.to_vec()), vec![sprime]);
            }        
        }
    }

    fn insert_rewards(
        &mut self,
        state: &[i32],
        action: &[MultiAgentAction],
        tasks: &Mission,
        team: &Team,
        n: usize, 
        m: usize
    ) {
        let mut new_reward: Vec<f64> = vec![0.; n + m];
        // for each agent in the team, determine the reward for taking the subaction
        for i in 0..n {
            let agent_reward = team.agents[i].rewards
                .get(&(state[i], action[i].base_action))
                .unwrap();
            new_reward[i] = *agent_reward
        }

        for j in 0..m {
            if tasks.tasks[j].accepting.contains(&state[n + j]) {
                new_reward[n + j] = 1.
            }
        }
        self.rewards.insert((state.to_vec(), action.to_vec()), new_reward);
    }

    fn insert_state(&mut self, state: &[i32]) {
        let state_idx = self.states.len();
        self.states.push(state.to_vec());
        self.state_mapping.insert(state.to_vec(), state_idx as i32);
    }

    fn insert_action(
        &mut self, 
        state: &[i32], 
        action: &[MultiAgentAction]
    ) {
        self.action_space.insert(action.to_vec());
        
        match self.actions.get_mut(state) {
            Some(a) => { 
                // the state-action key already exists, push to available action set
                a.push(action.to_vec());
            }
            None => { 
                // the action currently does not exist
                self.actions.insert(state.to_vec(), vec![action.to_vec()]);
            }
        }
    }

    fn action_space_mapping(&mut self) {
        for (i, a) in self.action_space.iter().enumerate() {
            self.action_map.insert(i as i32, a.to_vec());
        }
    }
}

// Learned something new, we can have multiple implementations with different decorations
#[pymethods]
impl MAMDP {
    pub fn get_number_states(&self) -> usize {
        self.states.len()
    } 

    pub fn get_number_transitions(&self) -> usize {
        self.transitions.len()
    }

    pub fn print_model_attr(&self) {
        println!("|S|: {}, |P|: {}", self.get_number_states(), self.get_number_transitions());
    }

    pub fn print_transitions(&self) {
        for t in self.transitions.iter() {
            println!("{:?}", t);
        }
    }

    pub fn print_rewards(&self) {
        for r in self.rewards.iter() {
            println!("{:?}", r)
        }
    }

    pub fn find_initial_transitions(&self) {
        for t in self.transitions.iter().filter(|(k, _v)| k.0 == self.initial_state) {
            println!("{:?}", t);
        }
    }

    pub fn print_action_mapping(&self) {
        for k in 0..self.action_map.len() {
            println!("{:?}", self.action_map.get(&(k as i32)).unwrap());
        }
    }
}

#[pyfunction]
pub fn build_model(
    initial_state: Vec<i32>,
    team: &Team, 
    tasks: &Mission,
    n: usize, 
    m: usize
) -> MAMDP {
    // starting from some initial state, use a BFS routine to
    // determine all of the following states
    // When the stack is empty then all of the states have been 
    // discovered. 
    let mamdp = mamdp_bfs(initial_state, team, tasks, n, m);
    mamdp
}

fn mamdp_bfs(
    initial_state: Vec<i32>, 
    team: &Team, 
    tasks: &Mission, 
    n: usize, 
    m: usize,
) -> MAMDP {

    let mut visited: HashSet<Vec<i32>> = HashSet::new();
    let mut stack: VecDeque<Vec<i32>> = VecDeque::new();
    let mut mamdp = MAMDP::new();

    // input the initial state into the back of the stack
    stack.push_back(initial_state.to_vec());
    visited.insert(initial_state.to_vec());
    mamdp.insert_state(&initial_state[..]);
    mamdp.insert_init_state(&initial_state[..]);
    while !stack.is_empty() { 
        // pop from the front of the stack
        let newstate = stack.pop_front().unwrap().to_vec();
        // for the new state
        // 1. has the new state already been visited?
        // 2. If the new state has not been visited then:
        //      a. determine its action set
        //      b. determine all of the transitions for the action set
        //      c. add the transitions to the back of the stack
        let actions = compute_input_actions(&newstate[..], team, tasks, n, m);
        for comb in actions.iter() {
            let transitions_to = transition_to_per_action_comb(
                &newstate[..], &comb[..], team, tasks, n, m
            );
            // for each of the transitions to
            // add the new state to the 
            mamdp.insert_action(&newstate[..], comb);
            mamdp.insert_rewards(&newstate[..], comb, tasks, team, n, m);
            for (s, p) in transitions_to.iter() {
                if !visited.contains(s) {
                    visited.insert(s.to_vec());
                    stack.push_back(s.to_vec());
                    mamdp.insert_state(&s[..]);
                    mamdp.insert_transition((s.to_vec(), *p), &newstate[..], comb);
                }
            }
        }
    }
    mamdp.action_space_mapping();
    mamdp.reverse_action_map = reverse_key_value_pairs(&mamdp.action_map);
    mamdp
}

#[pyfunction]
pub fn convert_to_multobj_mdp(
    model: &MAMDP, 
    team: &Team, 
    tasks: &Mission, 
    n: usize, 
    m: usize
) -> MultiObjectiveMDP {
    let mut condensed_model = MultiObjectiveMDP::new();
    for state in model.states.iter() {
        condensed_model.states.push(*model.state_mapping.get(&state[..]).unwrap());
    }
    let mut action_space: HashSet<i32> = HashSet::new();

    for state in model.states.iter() {
        let state_actions = compute_input_actions(state, team, tasks, n, m);
        for comb in state_actions.iter() {
            let vec_act_idx = *model.reverse_action_map
                .get(&comb.to_vec())
                .unwrap();
            action_space.insert(vec_act_idx);
            let state_idx = *model.state_mapping.get(&state.to_vec()).unwrap();
            match model.transitions.get(&(state.to_vec(), comb.to_vec())) {
                Some(x) => {
                    let sprime_idx_vec = x.iter()
                        .map(|(sprime, p)| (*model.state_mapping.get(sprime).unwrap(), *p))
                        .collect::<Vec<(i32, f64)>>();
                    condensed_model.transitions.insert((state_idx, vec_act_idx), sprime_idx_vec);
                }
                None => { }
            }
            match model.rewards.get(&(state.to_vec(), comb.to_vec())) {
                Some(x) => {
                    condensed_model.rewards.insert((state_idx, vec_act_idx), x.to_vec());
                }
                None => { 

                }
            }
        }
    }
    condensed_model.actions = action_space.into_iter().collect();
    condensed_model.construct_sparse_transition_matrix();
    condensed_model.construct_rewards_matrix(n, m);
    condensed_model
}
