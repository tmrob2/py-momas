use pyo3::prelude::*;
use hashbrown::{HashMap, HashSet};
use std::collections::VecDeque;
use crate::agent::agent::Team;
use crate::dfa::dfa::Mission;
use super::state_utils::*;
use super::action_utils::*;
use crate::reverse_key_value_pairs;
use crate::mdp::model::MultiObjectiveMDP;

#[pyclass]
pub struct TeamMDP {
    pub initial_state: TeamState,
    pub states: Vec<TeamState>,
    pub actions: HashMap<TeamState, Vec<TeamMDPAction>>,
    pub transitions: HashMap<(TeamState, TeamMDPAction), Vec<(TeamState, f64)>>,
    pub rewards: HashMap<(TeamState, TeamMDPAction), Vec<f64>>,
    actions_space: HashSet<TeamMDPAction>,
    action_map: HashMap<i32, TeamMDPAction>,
    reverse_action_map: HashMap<TeamMDPAction, i32>,
    state_mapping: HashMap<TeamState, i32>,
}

#[pymethods]
impl TeamMDP {
    pub fn print_transitions(&self) {
        for state in self.states.iter() {
            for action in self.actions.get(&state).unwrap().iter() {
                let transitions = self.transitions
                    .get(&(state.clone(), action.clone()))
                    .unwrap();
                println!("{:?}, {:?} => {:?}", state, action, transitions);
            }
        }
    }

    pub fn print_rewards(&self) {
        for state in self.states.iter() {
            // get the state transition
            let state_idx = self.state_mapping.get(&state).unwrap();
            for action in self.actions.get(&state).unwrap().iter() {
                let rbar = self.rewards.get(&(state.clone(), action.clone())).unwrap();
                println!("[{}]:{:?}, {:?} => {:?}", state_idx, state, action, rbar);
            }
        }
    }
}

impl TeamMDP {
    fn new() -> Self {
        TeamMDP {
            initial_state: TeamState::default(),
            states: Vec::new(),
            actions: HashMap::new(),
            transitions: HashMap::new(),
            rewards: HashMap::new(),
            actions_space: HashSet::new(),
            action_map: HashMap::new(),
            state_mapping: HashMap::new(),
            reverse_action_map: HashMap::new()
        }
    }

    pub fn get_action_space(&self) -> &HashSet<TeamMDPAction> {
        &self.actions_space
    }

    pub fn get_state_mapping(&self) -> &HashMap<TeamState, i32> {
        &self.state_mapping
    }

    fn insert_init_state(&mut self, init_state: &TeamState) {
        self.initial_state = init_state.clone();
    }

    fn insert_transition(
        &mut self,
        sprime: (TeamState, f64),
        state: &TeamState,
        action: TeamMDPAction
    ) {
        match self.transitions.get_mut(&(state.clone(), action)) {
            Some(t) => {
                t.push(sprime);
            }
            None => {
                self.transitions.insert((state.clone(), action), vec![sprime]);
            }
        }
    }

    fn insert_rewards(
        &mut self,
        agent: usize,
        state: &TeamState,
        action: TeamMDPAction,
        tasks: &Mission,
        team: &Team,
        n: usize,
        m: usize
    ) {
        // 1 + m because we have the agent cost, and all of the task rewards to compute
        let mut rewards = vec![0.; n + m]; 

        if action.switch {
            self.rewards.insert((state.clone(), action), rewards);
        } else {
            let agent_reward = match team.agents[agent].rewards
            .get(&(state.s, action.base_action)) {
                Some(r) => { 
                    *r
                }
                None => { panic!("could not find reward for state: {:?}, action: {:?}", state.s, action.base_action)}
            };
            if state.q.iter().enumerate().all(|(j, x)| 
                tasks.tasks[j].accepting.contains(x)
                || tasks.tasks[j].done.contains(x)
                || tasks.tasks[j].rejecting.contains(x)
            ) {
                rewards[agent] = 0.;
            } else {
                rewards[agent] = agent_reward;
            }
            for j in 0..m {
                if tasks.tasks[j].accepting.contains(&state.q[j]) {
                    rewards[n + j] = 1.;
                }
            }
            self.rewards.insert((state.clone(), action), rewards);
        }
    }

    fn insert_state(&mut self, state: &TeamState) {
        let state_idx = self.states.len();
        self.states.push(state.clone());
        self.state_mapping.insert(state.clone(), state_idx as i32);
    }

    fn insert_action(
        &mut self,
        state: &TeamState,
        action: TeamMDPAction
    ) {
        self.actions_space.insert(action);

        match self.actions.get_mut(state) {
            Some(available_actions) => { 
                available_actions.push(action);
            }
            None => { 
                self.actions.insert(state.clone(), vec![action]);
            }
        }
    }

    fn action_space_mapping(&mut self) {
        for (i, a) in self.actions_space.iter().enumerate() {
            self.action_map.insert(i as i32, a.clone());
        }
    }
}

#[pyfunction]
pub fn build_model(
    initial_state: &TeamState,
    team: &Team,
    tasks: &Mission,
    n: usize, 
    m: usize
) -> TeamMDP {
    let teammdp = teammdp_bfs(initial_state, team, tasks, n, m);
    teammdp
}

fn teammdp_bfs(
    initial_state: &TeamState,
    team: &Team, 
    tasks: &Mission,
    n: usize, 
    m: usize
) -> TeamMDP {

    let mut visited: HashSet<TeamState> = HashSet::new();
    let mut stack: VecDeque<TeamState> = VecDeque::new();
    let mut teammdp = TeamMDP::new();

    // input the inital state into the back of the stack
    stack.push_back(initial_state.clone());
    visited.insert(initial_state.clone());
    teammdp.insert_state(initial_state);
    teammdp.insert_init_state(initial_state);
    while !stack.is_empty() {
        // pop the fron of the stack
        let newstate = stack.pop_front().unwrap();
        // for the new state
        // 1. has the new state already been visited
        // 2. If the new state has not been visited then: 
        //      a. determine its action set
        //      b. determine all of the transitions of the action set
        //      c. add the transitions to the back of the stack
        //println!("new state: {:?}", newstate);
        let actions = get_available_actions(&newstate, team, tasks, n, m);
        //println!("actions: {:?}", actions);
        for action in actions.iter() {
            /*println!("state: {:?}, action: {:?}, working: {:?}", 
                newstate, 
                action, 
                agent_working(&newstate, tasks)
            );*/
            // insert the action into the team MDP
            teammdp.insert_action(&newstate, action.clone());
            teammdp.insert_rewards(
                newstate.agent as usize, 
                &newstate, 
                action.clone(), 
                tasks, 
                team, 
                n,
                m
            );
            for (s, p) in transitions(&newstate, action, team, tasks).iter() {
                if !visited.contains(s) {
                    visited.insert(s.clone());
                    stack.push_back(s.clone());
                    teammdp.insert_state(s);
                }
                teammdp.insert_transition((s.clone(), *p), &newstate, action.clone());
            }
        }
    }
    teammdp.action_space_mapping();
    teammdp.reverse_action_map = reverse_key_value_pairs(&teammdp.action_map);
    teammdp
}

#[pyfunction]
pub fn convert_to_multobj_mdp(
    model: &TeamMDP,
    team: &Team,
    tasks: &Mission,
    n: usize, 
    m: usize
) -> MultiObjectiveMDP {
    let mut condensed_model = MultiObjectiveMDP::new();
    for state in model.states.iter() {
        condensed_model.states.push(*model.state_mapping.get(&state).unwrap());
    }
    let mut action_space: HashSet<i32> = HashSet::new();
    for state in model.states.iter() {
        let actions = get_available_actions(state, team, tasks, n, m);
        for action in actions.iter() {
            let state_idx = *model.state_mapping.get(state).unwrap();
            let act_idx = *model.reverse_action_map.get(action).unwrap();
            action_space.insert(act_idx);
            match model.transitions.get(&(state.clone(), action.clone())) {
                Some(t) => {
                    condensed_model.insert_available_action(
                        *model.state_mapping.get(state).unwrap(), 
                        act_idx
                    );
                    let sprime_idx_vec = t.iter()
                        .map(|(sprime, p)| (*model.state_mapping.get(sprime).unwrap(), *p))
                        .collect::<Vec<(i32, f64)>>();
                    condensed_model.transitions.insert((state_idx, act_idx), sprime_idx_vec);
                }
                None => { }
            }
            match model.rewards.get(&(state.clone(), action.clone())) {
                Some(r) =>  {
                    condensed_model.rewards.insert((state_idx, act_idx), r.to_vec());
                }
                None => { }
            }
        }
    }
    condensed_model.actions = action_space.into_iter().collect();
    condensed_model.construct_sparse_transition_matrix();
    condensed_model.construct_rewards_matrix(n, m);
    condensed_model
}

