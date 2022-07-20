use super::model::MAMDP;
use crate::dfa::dfa::Mission;
use pyo3::prelude::*;

#[pyfunction]
pub fn assert_done(
    mamdp: &MAMDP, 
    tasks: &Mission,
    n: usize,
    m: usize
) -> bool {
    let done_states = mamdp.states
        .iter()
        .filter(|s| s[n..n+m]
            .iter()
            .enumerate()
            .all(|(k, x)| 
                tasks.tasks[k].done.contains(x) || tasks.tasks[k].rejecting.contains(x))
        )
        .count();
    match done_states {
        x if x > 0 => { true }
        _ => { false }
    }
}