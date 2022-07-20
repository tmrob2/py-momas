//! Using a generic multiobjective MDP
//! 
use super::model::MultiObjectiveMDP;
use rand::{thread_rng, prelude::IteratorRandom};
use crate::*;
use hashbrown::{HashMap, HashSet};
use float_eq::float_eq;
use std::time::Instant;
use crate::utils::lpsolver::gurobi_solver;

const UNSTABLE_POLICY: i32 = 5;

#[pyfunction]
pub fn multiobjective_scheduler_synthesis(
    eps: f64, 
    t: Vec<f64>, 
    model: &MultiObjectiveMDP,
    n: usize, 
    m: usize
) -> (HashMap<usize, Vec<f64>>, HashMap<usize, Vec<f64>>) {

    let t1 = Instant::now();

    let mut hullset: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut weights: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut X: HashSet<Vec<Mantissa>> = HashSet::new();
    let mut W: HashSet<Vec<Mantissa>> = HashSet::new();
    let mut schedulers: HashMap<usize, Vec<f64>> = HashMap::new();
    let tot_objs = n + m;

    let mut w = vec![0.; n];
    let mut w2 = vec![1. / m as f64; m];
    w.append(&mut w2);
    let (mu, r) = value_iteration(model, &w[..], eps, n, m);
    schedulers.insert(0, mu);
    X.insert(
        r
            .iter()
            .cloned()
            .map(|f| Mantissa::new(f))
            .collect::<Vec<Mantissa>>()
    );
    W.insert(
        w
            .iter()
            .cloned()
            .map(|f| Mantissa::new(f))
            .collect::<Vec<Mantissa>>()
    );

    let wrl = blas_dot_product(&r[..], &w[..]);
    let wt = blas_dot_product(&t[..], &w[..]);
    println!("r: {:.3?}", r);
    println!("w.r_l = {}", wrl);
    println!("w.t = {}", wt);
    println!("wrl < wt: {}", wrl < wt);
    if wrl < wt {
        println!("Ran in t(s): {:?}", t1.elapsed().as_secs_f64());
        return (schedulers, hullset)
    }
    hullset.insert(0, r);
    weights.insert(0, w);

    let mut lpvalid = true;
    // Once the extreme points are calculated then we can calculate the first separating
    // hyperplane
    let mut w: Vec<f64> = vec![0.; tot_objs];
    let mut count: usize = 1;
    while lpvalid {
        let gurobi_result = gurobi_solver(&hullset, &t[..], &tot_objs);
        match gurobi_result {
            //Ok(sol) => {
            Some(sol) => {
                // construct the new w based on the values from lp solution (if it exists)
                //for (ix, (var, val)) in sol.iter().enumerate() {
                for (ix, val) in sol.iter().enumerate() {
                    if ix < tot_objs {
                        //println!(" w[{:?}] = {}", ix, val);
                        //println!(" w[{:?}] = {:.3}", ix, val);
                        w[ix] = val_or_zero_one(val);
                    }
                }

                let new_w = w
                    .iter()
                    .clone()
                    .map(|f| Mantissa::new(*f))
                    .collect::<Vec<Mantissa>>();

                match W.contains(&new_w) {
                    true => {
                        println!("All points discovered");
                        lpvalid = false;
                    }
                    false => {
                        // calculate the new expected weighted cost based on w
                        let (mu, r) = value_iteration(model, &w[..], eps, n, m);
                        println!("r: {:.3?}", r);

                        let wrl = blas_dot_product(&r[..], &w[..]);
                        let wt = blas_dot_product(&t[..], &w[..]);
                        println!("w: {:3.3?}, r: {:.3?}", w, r);
                        println!("w.r_l = {}", wrl);
                        println!("w.t = {}", wt);
                        println!("wrl < wt: {}", wrl < wt);
                        if wrl < wt {
                            println!("Ran in t(s): {:?}", t1.elapsed().as_secs_f64());
                            return (schedulers, hullset)
                        }
                        // Insert the new solution
                        schedulers.insert(count, mu);
                        hullset.insert(count, r);
                        // create a copy of the weight vector and insert it into the set of values
                        W.insert(new_w);
                        weights.insert(count, w.to_vec());
                        count += 1;
                    }
                }
            }
            //Err(err) => {
            None => {
                //println!("{:?}", err);
                println!("infeasible");
                // the LP has finished and there are no more points which can be added to the
                // the polytope
                lpvalid = false;
            }
        }
    }
    (schedulers, hullset)
}

/// A hybrid policy-value iteration routine which computes a Pareto point
fn value_iteration(
    model: &MultiObjectiveMDP, 
    w: &[f64], 
    eps: f64, 
    n: usize, 
    m: usize
) -> (Vec<f64>, Vec<f64>) {
    // number of states
    let ns: usize = model.states.len();
    //let mut cs_matrices: Vec<_> = Vec::new();
    //let mut rewards_map: HashMap<i32, Vec<f64>> = HashMap::new();

    // Policy variables
    let mut pi: Vec<f64> = vec![0.; ns];
    let mut pi_new: Vec<f64> = vec![0.; ns];
    let mut x = vec![0.; ns];
    let mut xnew = vec![0.; ns];
    let mut xtemp = vec![0.; ns];

    // Multi-objective value variables
    let mut X: Vec<f64> = vec![0f64; ns * (n+m)];
    let mut Xnew: Vec<f64> = vec![0f64; ns * (n+m)];
    let mut Xtemp: Vec<f64> = vec![0f64; ns * (n+m)];
    let mut epsold: Vec<f64> = vec![0f64; ns * (n+m)];
    let mut inf_indicies: Vec<f64>;
    let mut inf_indicies_old: Vec<f64> = Vec::new();
    let mut unstable_count: i32 = 0;
    // Q matrix for storing state-action values
    let mut q = vec![0f64; ns * model.actions.len()];
    // tolerance update
    let mut epsilon: f64;
    let mut policy_stable = false;

    rand_proper_policy(&mut pi[..], model);
    let argmaxPinit = construct_argmaxPmatrix(&pi[..], model);
    let Rinit = construct_argmaxRagents(&pi[..], model, ns);
    // multiply the agent components of w with Rinit (dot)
    let mut rmv = vec![0f64; ns];
    blas_matrix_vector_mulf64(
        &Rinit.m, 
        &w[..n], 
        Rinit.nr as i32, 
        Rinit.nc as i32, 
        &mut rmv[..]
    );
    // compute the value for the init policy, lower bound on random action resolutions
    value_for_init_policy(
        &mut rmv[..], 
        &mut x[..], 
        &eps, 
        &argmaxPinit
    );

    while !policy_stable {
        policy_stable = true;
        for (ii, a) in model.actions.iter().enumerate() {
            // Instantiate a new matrix
            let S = model.get_transition_matrices().get(a).unwrap();
            let P = sparse_to_cs(S);
            // Compute the operation P.v
            let mut vmv = vec![0f64; ns];
            sp_mv_multiply_f64(P, &x[..], &mut vmv[..]);
            // Compute the matrix operation R.W
            let mut rmv = vec![0f64; ns];
            let R = model.get_reward_matricies().get(a).unwrap();
            blas_matrix_vector_mulf64(
                &R.m, 
                &w[..],
                R.nr as i32,
                R.nc as i32,
                &mut rmv[..]
            );
            assert_eq!(vmv.len(), rmv.len());
            // Perform the operation R.w + P.v
            add_vecs(&rmv[..], &mut vmv[..], ns as i32, 1.0);
            update_qmat(&mut q[..], &vmv[..], ii, model.actions.len()).unwrap();
        }
        max_values(
            &mut xnew[..], 
            &q[..],
            &mut pi_new[..],
            ns,
            model.actions.len()
        );
        // copy over the new value vector to calculate epsilon
        copy(&xnew[..], &mut xtemp[..], ns as i32);
        add_vecs(&x[..], &mut xnew[..], ns as i32, -1.0);
        // update the policy based on instability of action selection in the Q table
        update_policy(&xnew[..], &eps, &mut pi[..], &pi_new[..], ns, &mut policy_stable);
        // update the value vector
        copy(&xtemp[..], &mut x[..], ns as i32);
    }
    // Now that the policy has been computed, we need to determine the multi-objective
    // outcome of these policy choices
    let argmaxP = construct_argmaxPmatrix(&pi[..], model);
    let argmaxR = construct_argmaxRmatrix(&pi[..], model, n, m);
    // reset the epsilon value
    epsilon = 1.0;
    let obj_len = (ns * (n + m)) as i32;
    while epsilon > eps && unstable_count < UNSTABLE_POLICY {
        // For each of the objectives compute the value of the policy
        for k in 0..n + m {
            let mut vobjvec = vec![0f64; ns];
            // compute the operation argmax P.v_k
            sp_mv_multiply_f64(argmaxP.m, &X[k*ns..(k+1)*ns], &mut vobjvec[..]);
            // Perform the operation R[k] + P.v_k
            add_vecs(&argmaxR.m[k*ns..(k+1)*ns], &mut vobjvec[..], ns as i32, 1.0);
            //println!("vobj({}):\n{:.3?}", k, vobjvec);
            copy(&vobjvec[..], &mut Xnew[k*ns..(k+1)*ns], ns as i32);
        }

        // determine the diffences between X, Xnew
        copy(&Xnew[..], &mut Xtemp[..], obj_len);
        add_vecs(&Xnew[..], &mut X[..], obj_len, -1.0);
        epsilon = max_eps(&X[..]);
        // Compute if there are any infinite values we need to deal with
        inf_indicies = X.iter()
            .zip(epsold.iter())
            .enumerate()
            .filter(|(_, (x, z))| float_eq!(**x - **z, 0., abs <= eps) && **x != 0.)
            .map(|(ix, _)| ix as f64)
            .collect::<Vec<f64>>();
        if inf_indicies.len() == inf_indicies_old.len() {
            if inf_indicies.iter().zip(inf_indicies_old.iter()).all(|(val1, val2)| val1 == val2) {
                unstable_count += 1;
            } else {
                unstable_count = 0;
            }
        } else {
            unstable_count = 0;
        }
        copy(&X[..], &mut epsold[..], obj_len);
        // Copy X <- Xnew
        copy(&Xtemp[..], &mut X[..], obj_len);
        // copy the unstable indices
        inf_indicies_old = inf_indicies;
    }
    let mut r: Vec<f64> = Vec::with_capacity(n + m);
    for k in 0..n + m {
        r.push(X[k * ns + model.initial_state as usize]);
    }
    (pi, r)
}

fn rand_proper_policy(
    pi: &mut [f64], 
    model: &MultiObjectiveMDP,
) {
    // TODO: problem to solve, because some states are unreachable sometimes there
    // will not be an action to select
    for s in model.states.iter() {
        // get the random states
        match model.available_actions.get(s) {
            Some(actions) => {
                // choose a random action from the available states
                let act = actions.iter().choose(&mut thread_rng()).unwrap();
                pi[*s as usize] = *act as f64;
            }
            None => { 
                pi[*s as usize] = -1.;
            }   
        };
    }
}

/// Function computing the 
fn construct_argmaxPmatrix(
    pi: &[f64], 
    model: &MultiObjectiveMDP
) -> SparseMatrixAttr {

    let size = model.states.len();
    let mut ii: Vec<i32> = Vec::new();
    let mut jj: Vec<i32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    for state in model.states.iter() {
        let v = model.transitions
            .get(&(*state, pi[*state as usize] as i32))
            .unwrap();
        for (s, p) in v.iter() {
            ii.push(*state);
            jj.push(*s);
            vals.push(*p);
        }
    }

    let T = create_sparse_matrix(
        size as i32, 
        size as i32, 
        &ii[..], 
        &jj[..], 
        &vals[..]
    );

    let A = convert_to_compressed(T);


    SparseMatrixAttr {
        m: A,
        nr: size,
        nc: size,
        nnz: vals.len()
    }
}

/// Function computing the multi-objective rewards matrix for a set of agents completing
/// a set of tasks for some policy. 
/// 
/// Policy: pi
/// 
/// n: number of agens
/// 
/// m: number of tasks
fn construct_argmaxRmatrix(
    pi: &[f64],
    model: &MultiObjectiveMDP,
    n: usize,
    m: usize
) -> MatrixAttr {
    let size: usize = model.states.len();
    let mut R: Vec<f64> = vec![0.; size * n];
    for state in model.states.iter() {
        let r = model.get_reward_matricies()
            .get(&(pi[*state as usize] as i32)).unwrap();
        for c in 0..n + m {
            R[c * size + *state as usize] = r.m[c * size + *state as usize];
        }
    }
    MatrixAttr {
        m: R,
        nr: size,
        nc: n
    }
}

/// A rewards matrix which only considers the agents, This is useful in the computation
/// of the initial policy value for some random vector, which needs to be the lower bound
/// on a proper policy. 
fn construct_argmaxRagents(
    pi: &[f64],
    model: &MultiObjectiveMDP,
    n: usize
) -> MatrixAttr {
    let size: usize = model.states.len();
    let mut R: Vec<f64> = vec![0.; size * n];
    for state in model.states.iter() {
        let r = model.get_reward_matricies()
            .get(&(pi[*state as usize] as i32)).unwrap();
        for i in 0..n {
            R[i * size + *state as usize] = r.m[i * size + *state as usize];
        }
    }
    MatrixAttr {
        m: R,
        nr: size,
        nc: n
    }
}

fn value_for_init_policy(
    b: &mut [f64], // R.w rewards dot weight of the agent component of the multi-objective problem
    x: &mut [f64], // the initialised value vector
    eps: &f64, // tolerance
    argmaxP: &SparseMatrixAttr, // Transition matrix for a given initial policy
) {
    let mut epsilon: f64 = 1.0;
    let mut xnew: Vec<f64> = vec![0f64; argmaxP.nc];
    let mut epsold: Vec<f64> = vec![0f64; argmaxP.nc];
    let mut unstable_count: i32 = 0;
    let mut inf_indicies: Vec<f64>;
    let mut inf_indicies_old: Vec<f64> = Vec::new();
    while (epsilon > *eps) && (unstable_count < UNSTABLE_POLICY) {
        // Compute the matrix operations
        // vmv is the mutable value vector for accumulating products
        let mut vmv = vec![0f64; argmaxP.nr];
        // P.v operation
        sp_mv_multiply_f64(argmaxP.m, x, &mut vmv[..]);
        // R.w + P.v operation
        add_vecs(&mut b[..], &mut vmv[..], argmaxP.nr as i32, 1.0);
        copy(&vmv, &mut xnew[..argmaxP.nr], argmaxP.nr as i32);
        // xnew - x operation to compute epsilon
        add_vecs(&xnew[..], &mut x[..], argmaxP.nc as i32, -1.0);
        // get the largest difference
        epsilon = max_eps(&x[..]); 

        inf_indicies = x.iter()
            .zip(epsold.iter())
            .enumerate()
            .filter(|(_, (x, z))| float_eq!(**x - **z, 0., abs <= eps) && **x != 0.)
            .map(|(ix, _)| ix as f64)
            .collect::<Vec<f64>>();

        if inf_indicies.len() == inf_indicies_old.len() {
            if inf_indicies.iter().zip(inf_indicies_old.iter()).all(|(val1, val2)| val1 == val2) {
                unstable_count += 1;
            } else {
                unstable_count = 0;
            }
        } else {
            unstable_count = 0;
        }

        copy(&x[..], &mut epsold[..], argmaxP.nc as i32);
        // replace all of the values where x and epsold are equal with NEG_INFINITY or INFINITY
        // depending on sign
        copy(&vmv[..], &mut x[..argmaxP.nr], argmaxP.nr as i32);

        inf_indicies_old = inf_indicies;
    }
    if unstable_count >= UNSTABLE_POLICY {
        //println!("inf indices: {:?}", inf_indices_old);
        for ix in inf_indicies_old.iter() {
            if x[*ix as usize] < 0. {
                x[*ix as usize] = -f32::MAX as f64;
            }
        }
    }
}