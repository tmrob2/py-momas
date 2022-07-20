use gurobi::{attr, Continuous, LinExpr, Model, param, Status};
use hashbrown::HashMap;

/// The linear programming solver specific to a MOTAP problem
///
/// We are interested in generating weight vectors wbar representing separating hyperplanes.
/// To do this we implement a linear programming heuristic. Given some point r (derived as
/// the y0 values for each objective k) and a set of availability points X, choose wbar to
/// maximise min_{x in X}(wq - wx).
///
/// In practice this can be done by constructing the linear programming problem:
///
/// constraint 1: Sum w_i = 1
/// constraint 2..{X}+1 = w_i . (q_i - x_i ) >= d
///
/// Even for many tasks and many agents, these problems are generally very small and
/// will be quickly solved
///
/// Note on the ordering of variables:
///
pub fn gurobi_solver(h: &HashMap<usize, Vec<f64>>, t: &[f64], dim: &usize) -> Option<Vec<f64>> {
    let mut env = gurobi::Env::new("").unwrap();
    env.set(param::OutputFlag, 0).ok();
    env.set(param::LogToConsole, 0).ok();
    env.set(param::InfUnbdInfo, 1).ok();
    //env.set(param::FeasibilityTol,10e-9).unwrap();
    env.set(param::NumericFocus,2).ok();
    let mut model = Model::new("model1", &env).unwrap();

    // add variables
    let mut vars: HashMap<String, gurobi::Var> = HashMap::new();
    for i in 0..*dim {
        vars.insert(format!("w{}", i), model.add_var(
            &*format!("w{}", i),
            Continuous,
            0.0,
            0.0,
            1.0,
            &[],
            &[]).unwrap()
        );
    }
    let d = model.add_var(
        "d", Continuous, 0.0, -gurobi::INFINITY, gurobi::INFINITY, &[], &[]
    ).unwrap();

    model.update().unwrap();
    let mut w_vars = Vec::new();
    for i in 0..*dim {
        let w = vars.get(&format!("w{}", i)).unwrap();
        w_vars.push(w.clone());
    }
    let t_expr = LinExpr::new();
    let t_expr1 = t_expr.add_terms(&t[..], &w_vars[..]);
    let t_expr2 = t_expr1.add_term(1.0, d.clone());
    let t_expr3 = t_expr2.add_constant(-1.0);
    model.add_constr("t0", t_expr3, gurobi::Greater, 0.0).ok();

    for ii in 0..h.len() {
        let expr = LinExpr::new();
        let expr1 = expr.add_terms(&h.get(&ii).unwrap()[..], &w_vars[..]);
        let expr2 = expr1.add_term(1.0, d.clone());
        let expr3 = expr2.add_constant(-1.0);
        model.add_constr(&*format!("c{}", ii), expr3, gurobi::Less, 0.0).ok();
    }
    let w_expr = LinExpr::new();
    let coeffs: Vec<f64> = vec![1.0; *dim];
    let final_expr = w_expr.add_terms(&coeffs[..], &w_vars[..]);
    model.add_constr("w_final", final_expr, gurobi::Equal, 1.0).ok();

    model.update().unwrap();
    model.set_objective(&d, gurobi::Maximize).unwrap();
    model.optimize().unwrap();
    let mut varsnew = Vec::new();
    for i in 0..*dim {
        let var = vars.get(&format!("w{}", i)).unwrap();
        varsnew.push(var.clone());
    }
    let val = model.get_values(attr::X, &varsnew[..]).unwrap();
    //println!("model: {:?}", model.status());
    if model.status().unwrap() == Status::Infeasible {
        None
    } else {
        Some(val)
    }
}