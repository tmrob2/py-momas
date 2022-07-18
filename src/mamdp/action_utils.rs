use pyo3::prelude::*;
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[pyclass]
pub struct MultiAgentAction {
    pub working: bool, // is the agent already working on something
    pub task_job: i32, // The task number j to be assigned to the agent
    pub base_action: i32 // the available base action in the sub-mdp that this 
    // actino is replacing
}

#[pymethods]
impl MultiAgentAction {
    pub fn print_action(&self) -> String {
        let ret_str = format!("[ Working on: {}, job: {}, base action: {} ], ",
            self.working, self.task_job, self.base_action);
        ret_str
    }
}