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

#[pyclass]
pub struct MAMDP {
    pub states: Vec<Vec<i32>>,
    // Available actions for a particular state is much more useful then all of the action as a set
    pub actions: HashMap<Vec<i32>, Vec<Vec<MultiAgentAction>>>,
    pub transitions: HashMap<(Vec<i32>, Vec<MultiAgentAction>), Vec<(Vec<i32>, f64)>>,
    pub rewards: HashMap<(Vec<i32>, Vec<MultiAgentAction>), f64>
}

impl MAMDP {
    fn new() -> Self {
        MAMDP {
            states: Vec::new(),
            actions: HashMap::new(),
            transitions: HashMap::new(),
            rewards: HashMap::new()
        }
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
                self.transitions.insert((state.to_vec(), action.to_vec()), vec![sprime]);
            }        
        }
    }

    fn insert_state(&mut self, state: &[i32]) {
        self.states.push(state.to_vec());
    }

    fn insert_action(
        &mut self, 
        state: &[i32], 
        action: &[MultiAgentAction]
    ) {
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
            let transitions_to = 
                transition_to_per_action_comb(&newstate[..], &comb[..], team, tasks, n, m);
            // for each of the transitions to
            // add the new state to the 
            mamdp.insert_action(&newstate[..], comb);
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
    mamdp
}