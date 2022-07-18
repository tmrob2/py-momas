use pyo3::prelude::*;
pub mod dfa;
pub mod agent;
pub mod mamdp;
use dfa::dfa::*;
use agent::agent::*;
use mamdp::state_utils::*;
use mamdp::model::*;


/// Function to wrap and export the module to Python
#[pymodule]
pub fn rust_mamdp(_py: Python, module: &PyModule) -> PyResult<()> {
    //module.add_function(wrap_pyfunction!(do_something_with_dfa, module)?)?;
    //module.add_function(wrap_pyfunction!(find_neighbour_states, module)?)?;
    module.add_function(wrap_pyfunction!(construct_initial_state, module)?)?;
    module.add_function(wrap_pyfunction!(test_get_all_transitions, module)?)?;
    module.add_function(wrap_pyfunction!(build_model, module)?)?;
    module.add_class::<Agent>()?;
    module.add_class::<DFA>()?;
    module.add_class::<Mission>()?;
    module.add_class::<Team>()?;
    module.add_class::<MAMDP>()?;
    Ok(())
}