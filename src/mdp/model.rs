//! The purpose of the model is to construct a generic model which is accepted by
//! the algorithms for multiobjective model checking
use hashbrown::HashMap;
use pyo3::prelude::*;
use crate::{COO, MatrixAttr};

/// A generic MDP used for implementing and running algorithms. The type i32 is 
/// used to describe states but it is really a reference to the position in the vector
/// of the origin model state space. 
#[pyclass]
pub struct MultiObjectiveMDP {
    pub states: Vec<i32>,
    pub initial_state: i32,
    pub transitions: HashMap<(i32, i32), Vec<(i32, f64)>>,
    pub actions: Vec<i32>,
    pub rewards: HashMap<(i32, i32), Vec<f64>>, // A multi-objective mdp will always have a vector of rewards
    pub available_actions: HashMap<i32, Vec<i32>>, // the available actions in each state
    transition_matrices: HashMap<i32, COO>,
    reward_matrices: HashMap<i32, MatrixAttr>
}

impl MultiObjectiveMDP {
    pub fn new() -> Self {
        MultiObjectiveMDP {
            states: Vec::new(),
            initial_state: 0,
            transitions: HashMap::new(),
            actions: Vec::new(),
            rewards: HashMap::new(),
            available_actions: HashMap::new(),
            transition_matrices: HashMap::new(),
            reward_matrices: HashMap::new()
        }
    }

    #[allow(non_snake_case)]
    pub fn construct_sparse_transition_matrix(&mut self) {
        let mut sparse_matrices: HashMap<i32, COO> = HashMap::new();
        let size = self.states.len();
        for action in self.actions.iter() {
            let mut ii: Vec<i32> = Vec::new();
            let mut jj: Vec<i32> = Vec::new();
            let mut vals: Vec<f64> = Vec::new();
            for state in self.states.iter() {
                match self.transitions.get(&(*state, *action)) {
                    Some(v) => { // v is a vector of transition tuples (s, p)
                        for (s, p) in v.iter() {
                            ii.push(*state);
                            jj.push(*s);
                            vals.push(*p);
                        }
                     }
                    None => { }
                }
            }
            let nnz = vals.len() as i32;
            let S = COO {
                nzmax: nnz,
                nr: size as i32,
                nc: size as i32,
                i: ii,
                j: jj,
                x: vals,
                nz: nnz
            };
            sparse_matrices.insert(*action, S);
        }
        self.transition_matrices = sparse_matrices;
    }

    pub fn construct_rewards_matrix(&mut self, n: usize, m: usize) {
        let mut rewards_matrices: HashMap<i32, MatrixAttr> = HashMap::new();
        let size: usize = self.states.len();
        for action in self.actions.iter() {
            let mut r: Vec<f64> = vec![-f32::MAX as f64; size * (n + m)];
            for state in self.states.iter() {
                match self.rewards.get(&(*state, *action)) {
                    Some(reward) => { 
                        for (k, val) in reward.iter().enumerate() {
                            r[k * size + *state as usize] = *val;
                        }
                    }
                    None => { }
                }
            }
            rewards_matrices.insert(*action, MatrixAttr {
                m: r,
                nr: size,
                nc: n + m
            });
        }
        self.reward_matrices = rewards_matrices;
    }

    pub fn get_reward_matricies(&self) -> &HashMap<i32, MatrixAttr> {
        &self.reward_matrices
    }

    pub fn get_transition_matrices(&self) -> &HashMap<i32, COO> {
        &self.transition_matrices
    }

    pub fn insert_available_action(&mut self, state: i32, action: i32) {
        match self.available_actions.get_mut(&state) {
            Some(x) => { x.push(action); }
            None => { self.available_actions.insert(state, vec![action]); }
        }
    }
}

#[pymethods]
impl MultiObjectiveMDP {

    pub fn print_model_size(&self) {
        println!("|S|: {}, |P|: {}", self.states.len(), self.transitions.len())
    }

    pub fn print_states(&self) {
        println!("{:?}", self.states);
    }

    pub fn print_transitions(&self) {
        for state in self.states.iter() {
            for action in self.actions.iter() {
                match self.transitions.get(&(*state, *action)) {
                    Some(t) => { 
                        println!("{}, {} => {:?}", state, action , t);
                    }
                    None => { }
                }
            }
        }
    }

    pub fn print_rewards(&self) {
        for state in self.states.iter() {
            for action in self.actions.iter() {
                match self.rewards.get(&(*state, *action)) {
                    Some(r) => {
                        println!("{}, {} =>  {:?}", state, action, r);
                    }
                    None => {}
                }
            }
        }
    }

    pub fn print_available_actions(&self) {
        for state in self.states.iter() {
            println!("state: {:?} => {:?}", state, self.available_actions.get(state).unwrap());
        }
    }
}
