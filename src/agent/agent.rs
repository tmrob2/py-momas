use pyo3::prelude::*;
use hashbrown::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct Agent {
    #[pyo3(get)]
    pub states: Vec<i32>,
    #[pyo3(get)]
    pub init_state: i32,
    #[pyo3(get)]
    pub transitions: HashMap<(i32, i32), Vec<(i32, f64, String)>>,
    #[pyo3(get)]
    pub rewards: HashMap<(i32, i32), f64>,
    pub actions: Vec<i32>,
    pub available_actions: HashMap<i32, Vec<i32>>
}

#[pymethods]
impl Agent {
    #[new]
    fn new(init_state: i32, states: Vec<i32>, actions: Vec<i32>) -> PyResult<Self> {
        Ok(Agent { 
            states, 
            init_state, 
            transitions: HashMap::new(), 
            rewards: HashMap::new(), 
            actions,
            available_actions: HashMap::new()
        })
    }

    fn add_transition(&mut self, state: i32, action: i32, sprimes: Vec<(i32, f64, String)>) {
        match self.available_actions.get_mut(&state) {
            Some(x) => { x.push(action); }
            None => { self.available_actions.insert(state, vec![action]); }
        }
        self.transitions.insert((state, action), sprimes);
    }

    fn add_reward(&mut self, state: i32, action: i32, reward: f64) {
        self.rewards.insert((state, action), reward);
    }

    fn clone(&self) -> Self {
        Agent { 
            states: self.states.to_vec(),
            init_state: self.init_state,
            transitions: self.transitions.clone(),
            rewards: self.rewards.clone(),
            actions: self.actions.to_vec(),
            available_actions: self.available_actions.clone()
        }
    }
}

#[pyclass]
pub struct Team {
    pub agents: Vec<Agent>,
    pub size: usize
}

#[pymethods]
impl Team {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Team { agents: Vec::new(), size: 0})
    }

    fn add_agent(&mut self, agent: Agent) {
        self.agents.push(agent);
        self.size += 1;
    }

    fn print_initial_states(&self){
        for agent in self.agents.iter() {
            println!("init state: {:?}", agent.init_state);
        }
    }

    pub fn print_transitions(&self, agent: usize) {
        for transition in self.agents[agent].transitions.iter() {
            println!("{:?}", transition)
        }
    }
}

