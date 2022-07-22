use pyo3::prelude::*;

#[pyclass]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TeamMDPAction {
    pub task: i32,
    pub base_action: i32,
    pub switch: bool,
    pub working: bool,
}

#[pymethods]
impl TeamMDPAction {
    pub fn print_action(&self) {
        println!("{:?}", &self);
    }
}