use pyo3::prelude::*;
use crate::dfa::dfa::*;
use crate::agent::agent::*;
use super::action_utils::TeamMDPAction;

#[pyclass]
#[derive(Hash, PartialEq, Eq, Default, Clone, Debug)]
pub struct TeamState {
    pub s: i32,
    pub agent: i32,
    pub q: Vec<i32>
}

#[pymethods]
impl TeamState {
    pub fn print_state(&self) {
        println!("{:?}", &self)
    }

    pub fn state_repr(&self) -> String {
        format!("{:?}", &self)
    }
}

pub fn agent_working(state: &TeamState, tasks: &Mission) -> bool {
    if state.q.iter().enumerate().all(|(i, x)| 
        tasks.tasks[i].done.contains(x) 
        || tasks.tasks[i].rejecting.contains(x) 
        || tasks.tasks[i].initial_state == *x
    ) {
        // the agent is not working
        false
    } else {
        true
    }
}

pub fn current_task(state: &TeamState, tasks: &Mission) -> i32 {
    let mut task_working_on: Vec<i32> = Vec::new();
    for (j, q) in state.q.iter().enumerate() {
        if !(
            tasks.tasks[j].initial_state == *q 
            || tasks.tasks[j].rejecting.contains(q) 
            || tasks.tasks[j].done.contains(q)
        ) {
            //println!("Task: {} q: {} => means working", j, q);
            task_working_on.push(j as i32);
        }
    }
    match task_working_on.len() {
        1 => { }
        _ => { panic!("state: {:?}, task being worked on: {:?}", state, task_working_on); }
    }
    task_working_on[0]
}


fn tasks_remaining(state: &TeamState, m: usize, tasks: &Mission) -> Vec<i32> {
    let mut tasks_not_yet_started = vec![];
    for task in 0..m {
        if tasks.tasks[task].initial_state == state.q[task] {
            tasks_not_yet_started.push(task as i32);
        }
    }
    tasks_not_yet_started
}

#[pyfunction]
#[pyo3(name="construct_initial_state")]
pub fn construct_initial_state(
    tasks: &Mission, 
    team: &Team, 
    m: usize, 
    agent: usize
) -> TeamState {
    let mut q: Vec<i32> = Vec::new();
    for j in 0..m {
        q.push(tasks.tasks[j].initial_state)
    }  
    TeamState {
        s: team.agents[agent].init_state,
        agent: agent as i32,
        q 
    }
}

#[pyfunction]
pub fn get_available_actions(
    state: &TeamState,
    agents: &Team, 
    tasks: &Mission,
    n: usize,
    m: usize
) -> Vec<TeamMDPAction> {
    let mut actions: Vec<TeamMDPAction> = Vec::new();
    // check if the agent is working or if a task allocation is required
    if agent_working(state, tasks) {
        // continue working on the current task
        for base_action in agents.agents[state.agent as usize].available_actions
            .get(&state.s).unwrap().iter() {
            actions.push(TeamMDPAction {
                base_action: *base_action,
                task: current_task(state, tasks),
                switch: false,
                working: agent_working(state, tasks)
            });
        }
    } else {
        let tremaining = tasks_remaining(state, m, tasks);
        if tremaining.is_empty() {
            for base_action in agents.agents[state.agent as usize].available_actions
                .get(&state.s).unwrap().iter() {
                actions.push(TeamMDPAction {
                    base_action: *base_action,
                    task: -1,
                    switch: false,
                    working: agent_working(state, tasks)
                });
            }
        } else {
            //println!("Switch: {:?}", check_for_switch(state, tasks, n, m));
            if check_for_switch(state, tasks, n, m) {
                // If it is possible to generate a switch transition, switch to the same
                // state configuration in the next agent
                actions.push(TeamMDPAction {
                    base_action: -1,
                    task: -1,
                    switch: true,
                    working: agent_working(state, tasks)
                });
            }
            for task in tremaining.iter() {
                for base_action in agents.agents[state.agent as usize].available_actions
                    .get(&state.s).unwrap().iter() {
                    actions.push(TeamMDPAction {
                        base_action: *base_action,
                        task: *task,
                        switch: false,
                        working: agent_working(state, tasks)
                    });
                }
            }
        }
    }
    actions
}

fn check_for_switch(state: &TeamState, tasks: &Mission, n: usize, m: usize) -> bool {
    if state.agent < (n - 1) as i32 {
        // if is possible that there can be a switch transition
        let mut init_or_final = vec![false; m];
        for j in 0..m {
            if tasks.tasks[j].done.contains(&state.q[j]) 
                || tasks.tasks[j].rejecting.contains(&state.q[j]) 
                || tasks.tasks[j].initial_state == state.q[j] {
                init_or_final[j] = true;
            } else {
                init_or_final[j] = false;
            }
        }
        if init_or_final.iter().all(|x| *x) {
            true
        } else {
            false
        }
    } else {
        false
    }
}

#[pyfunction]
pub fn transitions(
    state: &TeamState, 
    action: &TeamMDPAction, 
    team: &Team, 
    tasks: &Mission, 
) -> Vec<(TeamState, f64)> {
    let mut new_transitions: Vec<(TeamState, f64)> = Vec::new();
    // two branches at this point
    // 1. is the action a switch or not?
    if action.switch {
        // pass control of the mission over to the next agent in the initial position
        new_transitions.push((
            TeamState {
                agent: state.agent + 1,
                s: team.agents[(state.agent + 1) as usize].init_state,
                q: state.q.to_vec()
            },
            1.0
        ));
    } else {    
        // the agent will take a base action
        // the task which moves forward is the that according to the action
        // this process is the same for both a task allocation and when the 
        // agent is already working on a task
        if action.task > -1 {
            for (s, p, w) in team.agents[state.agent as usize].transitions
                .get(&(state.s, action.base_action)).unwrap().iter() {
                // compute the DFA transition based on the word w
                let qprime_elem: i32 = tasks.tasks[action.task as usize]
                    .get_transitions(state.q[action.task as usize], w.to_string());
                let mut q_prime = state.q.to_vec();
                q_prime[action.task as usize] = qprime_elem;
                new_transitions.push((TeamState {
                        agent: state.agent,
                        s: *s,
                        q: q_prime
                    }, 
                    *p));
            }
        } else {
            // the agent is not working on anything and there is no task allocation
            // therefore the agent should stay in the same state
            new_transitions.push((TeamState {
                agent: state.agent,
                s: state.s,
                q: state.q.to_vec()
            }, 1.0));
        }
    }
    new_transitions
}