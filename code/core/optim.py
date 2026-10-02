import math
import numpy as np
import casadi as ca
from core.emu import EMUSubNetwork
from scipy.stats import chi2
from scipy.linalg import qr
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize, LinearConstraint, linprog
import os
import contextlib
import signal



@contextlib.contextmanager
def silence(timeout=120):
    with open(os.devnull, "w") as f:
        old1, old2 = os.dup(1), os.dup(2)
        try:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            yield
        finally:
            os.dup2(old1, 1)
            os.dup2(old2, 2)
            os.close(old1)
            os.close(old2)


def linalg_lstsq_bounded(model, v):
    
    N= model.N
    v_lb= model.lb
    v_ub= model.ub

    u0, _, _, _ = np.linalg.lstsq(N, v, rcond=None)

    def obj(u):
        r = model.N @ u - v
        return 0.5 * np.dot(r, r)

    def grad(u):
        return model.N.T @ (model.N @ u - v)

    constraint = LinearConstraint(N, v_lb, v_ub)

    res = minimize(
        obj,
        u0,
        jac=grad,
        constraints=[constraint],
        method="trust-constr",
    )

    return res.x


def make_v_free_feasible(v_init, K, lb_v_net, ub_v_net, lb_v_free, ub_v_free):

    def objective(v):
        d = v - v_init
        return np.dot(d, d)

    def gradient(v):
        return 2.0 * (v - v_init)

    linear_constraint = LinearConstraint(K, lb_v_net, ub_v_net)
    bounds = Bounds(lb_v_free, ub_v_free)

    x0 = np.clip(v_init, lb_v_free, ub_v_free)

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="trust-constr",
        constraints=[linear_constraint],
        bounds=bounds,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x


def make_u_feasible(u_init,N, T, v_lb, v_ub, v_net_lb, v_net_ub):

    A = np.vstack([N, T @ N])
    lb = np.concatenate([v_lb, v_net_lb])
    ub = np.concatenate([v_ub, v_net_ub])
    con = LinearConstraint(A, lb, ub)

    obj = lambda u: np.sum((u - u_init) ** 2)
    grad = lambda u: 2 * (u - u_init)

    res = minimize(obj, u_init, jac=grad, constraints=[con], method='SLSQP')
    return res.x



def nullspace_parameterization(S, tol=1e-12):
    m, n = S.shape

    Q, R, P = qr(S, pivoting=True)

    r = np.sum(np.abs(np.diag(R)) > tol)

    pivot = P[:r]
    free = P[r:]

    R11 = R[:r, :r]
    R12 = R[:r, r:]

    X = -np.linalg.solve(R11, R12)

    # parameterization in permuted coordinates
    Kperm = np.vstack((X, np.eye(n-r)))

    # undo permutation
    K = np.zeros((n, n-r))
    K[P, :] = Kperm

    return K, free, pivot


def inst_estimate(model, **kwargs):

    ### using null space setting
    null_space = ca.DM(model.N)
    u_size = null_space.shape[1]
    u= ca.MX.sym('u', u_size)
    
    v= ca.mtimes(null_space, u)
    v_size= v.numel()

    for i, rxn in enumerate(model.rxns):
        rxn.v= v[i]

    for emu in model.emu_net.V:
        if emu.met in model.tracer:
            emu.x0= ca.DM(emu.MID).T
            emu.x= ca.DM(emu.MID).T
        else:
            emu.x0= ca.DM(emu.MID).T
            emu.x= ca.MX.sym(str(emu), 1, emu.size + 1)

    met_pools= []
    met_pools_str= []

    comps= []
    comps_str= []


    target_emus_x_index = dict()
    x_index_counter = 0

    subnet_size = 1
    subnet_number = 1

    max_subnet_size = max([emu.size for emu in model.target_emus])

    X_subnets= []
    dXdt_subnets= []
    X0_subnets = []

    while subnet_size <= max_subnet_size:
        subnet = EMUSubNetwork()
        known_emus = []
        unknown_emus = []

        # looping through the EMU rxns of the EMU subnetwork and collect the ones with the defined size
        for emu_rxn in model.emu_net.E:
            if emu_rxn.size == subnet_size:
                subnet.add_EMU_reaction(emu_rxn)

        subnet.V= sorted(subnet.V, key=lambda x: (x.size, str(x)))

        if not subnet:
            subnet_size += 1
            continue

        for emu in subnet.V:
            if emu.solved:
                known_emus.append(emu)
            elif not emu.solved:
                unknown_emus.append(emu)

        # initializing the A and B matrices for the equation: AX=BY
        # A is the multiplier for unknown MID variables
        A = ca.MX.zeros(len(unknown_emus), len(unknown_emus))
        # B is the multiplier for known MID variables
        B = ca.MX.zeros(len(unknown_emus), len(known_emus))

        for unknown_emu_index, unknown_emu in enumerate(unknown_emus):
            total_input_flux = ca.MX(0)

            # identifying the emu rxns producing the EMU with unknown MID
            for emu_rxn in subnet.E:
                if emu_rxn.end == unknown_emu:

                    # assigning values to A and B for the in emu rxns based on the mass balance equations
                    if emu_rxn.start in unknown_emus:
                        lhs_index = unknown_emus.index(emu_rxn.start)
                        A[unknown_emu_index, lhs_index] += (emu_rxn.rxn.v * emu_rxn.rxn_coef)


                    elif emu_rxn.start in known_emus:
                        lhs_index = known_emus.index(emu_rxn.start)
                        B[unknown_emu_index, lhs_index] -= (emu_rxn.rxn.v * emu_rxn.rxn_coef)

                    total_input_flux += (emu_rxn.rxn.v * emu_rxn.rxn_coef)

            # assigning values to A for the EMU rxns, in which unknown emu is precursor
            A[unknown_emu_index, unknown_emu_index] -= total_input_flux

        # creating matrix Y, which is MIDs for determined EMUs
        Y = ca.vertcat(*[emu.x for emu in known_emus])


        # solving the INST ODEs
        # for i, unknown_emu in enumerate(unknown_emus):
        M= []
        for unknown_emu in unknown_emus:
            met_id= unknown_emu.met.split('.')[0]
            comp= unknown_emu.met.split('.')[1] if '.' in unknown_emu.met else None


            if comp:
                if comp in comps_str:
                    idx= comps_str.index(comp)
                    comp_coef= comps[idx]
                else:
                    comp_coef= ca.MX.sym(comp)
                    comps.append(comp_coef)
                    comps_str.append(comp)
            else:
                comp_coef= 1

            if met_id in met_pools_str:
                idx= met_pools_str.index(met_id)
                met_pool= met_pools[idx]
            else:
                met_pool= ca.MX.sym(met_id)
                met_pools.append(met_pool)
                met_pools_str.append(met_id)

            M.append(met_pool * comp_coef)

        M_inv = ca.diag(1 / ca.vertcat(*M))

        X0 = ca.vertcat(*[unknown_emu.x0.T for unknown_emu in unknown_emus])

        X = ca.vertcat(*[unknown_emu.x for unknown_emu in unknown_emus])

        Xdot= M_inv @ (A @ X - B @ Y)

        for row, emu in enumerate(unknown_emus):
            emu.xdot= Xdot[row, :]
            emu.solved = True

            if emu in model.target_emus:
                target_emus_x_index[str(emu)]= (x_index_counter, x_index_counter + emu.size + 1)

            x_index_counter += (emu.size + 1)


        dXdt= ca.vertcat(*[emu.xdot.T for emu in unknown_emus])

        X = ca.vertcat(*[unknown_emu.x.T for unknown_emu in unknown_emus])
        X_subnets.append(X)

        dXdt_subnets.append(dXdt)

        X0_subnets.append(X0)

        subnet_size += 1

    X0= ca.vertcat(*X0_subnets)
    X= ca.vertcat(*X_subnets)
    dXdt= ca.vertcat(*dXdt_subnets)

    c = ca.vertcat(*met_pools)
    c_size = c.numel()

    if comps:
        alphas= ca.vertcat(*comps)
        alpha_size= alphas.numel()
        p= ca.vertcat(u, c, alphas)
    else:
        p= ca.vertcat(u, c)

    dae= {"x": X, "p": p, "ode": dXdt}

    time_points, emu2mid_measured, _ = model.inst_measured_MIDs

    integrator_opts= {"linear_multistep_method": "bdf",
                      "nonlinear_solver_iteration": "newton",
                      "abstol": 1e-6, "reltol": 1e-6, "max_num_steps": 1e6,
                      }
    F= ca.integrator("F", "cvodes", dae, 0, time_points, integrator_opts)


    J_mid = ca.MX(0)

    xk = F(x0= X0, p= p)["xf"]

    # mid_std_scale= ca.MX.sym('mid_std_scale')
    mid_std_scale= kwargs.get('mid_std_scale', 1)

    for emu_data in emu2mid_measured:
        emu_mid_sim= None
        comp_coefs= 0
        for emu in model.target_emus_dict[emu_data]:
            mid_sim= xk[target_emus_x_index[emu][0]:target_emus_x_index[emu][1],:]

            met= emu.split('_')[0]
            met_id= met.split('.')[0]
            comp= met.split('.')[1] if '.' in met else None


            # comp alphas are defined as variables
            if comp:
                idx= comps_str.index(comp)
                comp_coef= comps[idx]
            else:
                comp_coef= 1

            comp_coefs += comp_coef

            mid_sim *= comp_coef
            
            if emu_mid_sim is None:
                emu_mid_sim = mid_sim
            else:
                emu_mid_sim += mid_sim
        
        mid_sim/= comp_coefs
        
        emu_mid_meas_mean= ca.DM(emu2mid_measured[emu_data][0])
        emu_mid_meas_std= ca.DM(emu2mid_measured[emu_data][1])
        mid_res = emu_mid_sim - emu_mid_meas_mean
        J_mid += ca.sumsqr(mid_res / (emu_mid_meas_std * mid_std_scale))


    J = J_mid

    T= ca.DM(model.T)
    g1= null_space @ u
    g2= T @ null_space @ u

    if comps:
        g3= ca.sum1(alphas)
        g= ca.vertcat(g1, g2, g3)
    else:
        g= ca.vertcat(g1, g2)


    nlp= {"x": p, "f": J, "g": g}

    
    ipopt_opts = {
        "expand": True,
        "print_time": False,

        "ipopt": {
            "linear_solver": "mumps",
            "mu_strategy": "adaptive",

            "hessian_approximation": "limited-memory",
            "limited_memory_max_history": 10,

            "nlp_scaling_method": "gradient-based",
            "nlp_scaling_max_gradient": 100.0,

            "tol": 1e-8,
            "max_iter": 1000,

            "acceptable_tol": 1e-8,
            "acceptable_iter": 10,
            "acceptable_dual_inf_tol": 1e-8,
            "acceptable_compl_inf_tol": 1e-8,
            "acceptable_constr_viol_tol": 1e-8,
            "acceptable_obj_change_tol": 1e-8,

            "print_level": 0,
            "print_timing_statistics": "no",
            "sb": "yes",

        },
    }


    A_ub = np.vstack([-model.N, model.N, model.T @ model.N, -model.T @ model.N])
    b_ub = np.concatenate([model.lb, model.ub, model.vnet_ub, -model.vnet_lb])

    u_lb = np.zeros(u_size)
    u_ub = np.zeros(u_size)

    for i in range(u_size):
        # minimize u_i
        c = np.zeros(u_size)
        c[i] = 1
        res_min = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
        u_lb[i] = res_min.fun if res_min.success else -np.inf

        # maximize u_i
        c = np.zeros(u_size)
        c[i] = -1
        res_max = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
        u_ub[i] = -res_max.fun if res_max.success else np.inf


    c_lb = model.c_lb * ca.DM.ones(c_size)
    c_ub = model.c_ub * ca.DM.ones(c_size)

    if comps:

        alphas_lb= ca.DM.zeros(alpha_size)
        alphas_ub= ca.DM.ones(alpha_size)

        lbp= ca.vertcat(u_lb, c_lb, alphas_lb)
        ubp= ca.vertcat(u_ub, c_ub, alphas_ub)

    else:
        lbp = ca.vertcat(u_lb, c_lb)
        ubp = ca.vertcat(u_ub, c_ub)


    lbg1= ca.DM(model.lb)
    ubg1= ca.DM(model.ub)

    for i, v_mes in model.measured_rxn2flux.items():
        lbg1[i], ubg1[i] = v_mes[0], v_mes[1]


    lbg2= ca.DM(model.vnet_lb)
    ubg2= ca.DM(model.vnet_ub)


    if comps:
        lbg3= ca.DM([1])
        ubg3= ca.DM([1])

        lbg= ca.vertcat(lbg1, lbg2, lbg3)
        ubg= ca.vertcat(ubg1, ubg2, ubg3)
    else:
        lbg= ca.vertcat(lbg1, lbg2)
        ubg= ca.vertcat(ubg1, ubg2)

    if kwargs['optim_type'] == 'initial':

        if 'init_guess' in kwargs:
            p0= kwargs['init_guess']
        else:

            rng = np.random.default_rng(kwargs['seed'])

            u_init= rng.uniform(low= u_lb, high= u_ub, size= u_size)
            u0_np= make_u_feasible(u_init, model.N, model.T, model.lb, model.ub, model.vnet_lb, model.vnet_ub)
            u0= ca.DM(u0_np)

            c0= ca.DM(10 ** rng.uniform(low= np.log10(model.c_lb), high= np.log10(model.c_ub), size=c_size))

            if comps:
                alphas0= ca.DM.ones(alpha_size)/alpha_size
                p0= ca.vertcat(u0, c0, alphas0)

            else:
                p0= ca.vertcat(u0, c0)


        solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)
        with silence():
            sol = solver(x0=p0, lbx=lbp, ubx=ubp, lbg=lbg, ubg=ubg)

        

    elif kwargs['optim_type'] == 'reopt':

        prev_sol= kwargs['prev_sol']
        p0 = prev_sol['x_sol']
        
        ipopt_opts['ipopt'].update({
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-8,
            "warm_start_bound_frac": 1e-8,
            "warm_start_slack_bound_push": 1e-8,
            "warm_start_slack_bound_frac": 1e-8,
            "warm_start_mult_bound_push": 1e-8,

            "max_iter":200,
        })
        
        solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)
        with silence():
            sol = solver(x0=p0, lbx=lbp, ubx=ubp, lbg=lbg, ubg=ubg,
                        lam_x0=prev_sol['lam_x'], lam_g0=prev_sol['lam_g'])



    elif kwargs['optim_type'] == 'ci':

        prev_sol= kwargs['prev_sol']
        var_index= kwargs['var_index']
        fixed_value= kwargs['fixed_value']

        p0 = prev_sol['x_sol']

        if kwargs['var_type'] == 'vnet':
            lbg[v_size + var_index] = fixed_value
            ubg[v_size + var_index] = fixed_value

        elif kwargs['var_type'] == 'c':
            lbp[u_size + var_index] = fixed_value
            ubp[u_size + var_index] = fixed_value

        elif kwargs['var_type'] == 'alpha':
            lbp[u_size + c_size + var_index] = fixed_value
            ubp[u_size + c_size + var_index] = fixed_value
        
        ipopt_opts['ipopt'].update({
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-9,
            "warm_start_bound_frac": 1e-9,
            "warm_start_slack_bound_push": 1e-9,
            "warm_start_slack_bound_frac": 1e-9,
            "warm_start_mult_bound_push": 1e-9,
            "fixed_variable_treatment": "make_constraint",
            "mu_init": 1e-6,

            "tol": 1e-4,
            "max_iter": 20,

            "acceptable_tol": 1e-4,
            "acceptable_iter": 3,
            "acceptable_dual_inf_tol": 1e-2,
            "acceptable_compl_inf_tol": 1e-2,
            "acceptable_constr_viol_tol": 1e-8,
            "acceptable_obj_change_tol": 1e-4,

            "print_level": 0
            
        })

        solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)
        with silence():
            sol = solver(x0=p0, lbx=lbp, ubx=ubp, lbg=lbg, ubg=ubg, lam_x0=prev_sol['lam_x'], lam_g0=prev_sol['lam_g'])


    
    if sol:
        
        x_sol = np.array(sol["x"]).squeeze()
        obj= float(sol["f"])

        u_sol= x_sol[:u_size]
        c_sol= x_sol[u_size:u_size + c_size]

        if comps:
            alpha_sol= x_sol[u_size + c_size:]
            comp_alphas_sol= (comps_str, alpha_sol)

        v_sol= model.N@u_sol
        v_net_sol= model.T@v_sol

        
        met_pools_sol= (met_pools_str, c_sol)
        
        lam_x= np.array(sol["lam_x"]).squeeze()
        lam_g= np.array(sol["lam_g"]).squeeze()


        sol_clean= {'obj': obj, 'mid_std_scale': mid_std_scale, 'x_sol': x_sol,
            'v_sol': v_sol, 'vnet_sol': v_net_sol, #'v_xch_sol': v_xch_sol,
            'c_sol': c_sol,'met_pools_sol': met_pools_sol,
            'lam_x': lam_x, 'lam_g': lam_g}
        if comps:
            sol_clean['alpha_sol']= alpha_sol
            sol_clean['comp_alphas_sol']= comp_alphas_sol
        
    else:
        sol_clean= {}
        obj= 0

    
    for i, rxn in enumerate(model.rxns):
        del rxn.v

    for emu in model.emu_net.V:
        del emu.x0
        del emu.x
        if hasattr(emu, 'xdot'):
            del emu.xdot




    if obj == 0:
        sol_clean['obj']=1e4
        
        if kwargs['optim_type'] == 'reopt':
            prev_sol['obj'] *= (prev_sol['mid_std_scale']**2)
            prev_sol['mid_std_scale']= 1
            return prev_sol

        kwargs['mid_std_scale']= mid_std_scale * 10
        kwargs['init_guess']= p0

        if kwargs['mid_std_scale'] > 1e4:
            return {}

        # print("for {} scale increased to {}".format(kwargs['seed'], kwargs['mid_std_scale']))
        model.set_tracer(m_fun='est')
        new_sol= inst_estimate(model, **kwargs)
        if mid_std_scale > 1:
            return new_sol


        elif mid_std_scale == 1:
            if bool(new_sol)==0:
                return sol_clean

            else:
                while new_sol['mid_std_scale'] > 1:
                    kwargs['mid_std_scale']= new_sol['mid_std_scale'] / 10

                    # print("for {} scale decreased to {}".format(kwargs['seed'], kwargs['mid_std_scale']))

                    kwargs['optim_type']= 'reopt'
                    kwargs.update({'prev_sol': new_sol})
                    model.set_tracer(m_fun='est')
                    new_sol= inst_estimate(model, **kwargs)
                return new_sol

    else:
        return sol_clean





def gen_power_grid(c_lb, c_ub, power= 2, reverse=False):

    d= max(10, int(c_ub- c_lb))
    n_bins= min(d, 20)

    u = np.linspace(0, 1, n_bins + 1)

    if not reverse:
        u_transformed = u ** power
    else:
        u_transformed = 1 - (1 - u) ** power

    bins = c_lb + (c_ub - c_lb) * u_transformed

    if reverse:
        bins= bins[::-1]

    return bins[1:]



def calc_ci_one_var_pos(model, var_type, i):

    org_sol = model.estimation_solution

    if var_type == 'vnet':
        org_value= org_sol['vnet_sol'][i]
        ci_ub= model.vnet_ub[i]
            

    elif var_type == 'c':
        org_value= org_sol['c_sol'][i]
        ci_ub= model.c_ub

    elif var_type == 'alpha':
        org_value= org_sol['alpha_sol'][i]
        ci_ub= model.c_ub

        
    ci_lb= org_value
    obj_lb= org_sol['obj']
    fixed_value= ci_ub
    th= chi2.ppf(0.95, 1)
    n_iter = 0


    state= 'max_bound'


    while n_iter <= 20:

        optim_kwargs= {
            'optim_type': 'ci',
            'prev_sol': org_sol,
            'var_index': i, 
            'fixed_value': fixed_value,
            'var_type': var_type}
        

        model.set_tracer(m_fun='est')
        new_sol = inst_estimate(model, **optim_kwargs)

        new_obj= new_sol['obj']

        if var_type == 'vnet':
            new_x= new_sol['vnet_sol'][i]
        elif var_type == 'c':
            new_x= new_sol['c_sol'][i]
        elif var_type== 'alpha':
            new_x= new_sol['alpha_sol'][i]

        if new_obj == 0:
            new_obj = 1e4
            new_x= fixed_value

        delta= new_obj - org_sol['obj']

        gap= delta - th

        if abs(gap) < 0.1 or (n_iter == 0 and delta < th):
            break

        else:

            if state == 'max_bound':

                grid= gen_power_grid(ci_lb, ci_ub)
                fixed_value= grid[n_iter]
                state = 'grid'

            elif state == 'grid':

                if gap > 0:
                    ci_ub= new_x                
                    obj_ub= new_obj
                    
                    if max(obj_ub - th, th - obj_lb) < th:
                        fixed_value = ci_lb + ((ci_ub-ci_lb)*(th-obj_lb)/(obj_ub-obj_lb))

                    else:
                        fixed_value = (ci_ub + ci_lb)/2

                    state = 'final'
                    n_iter = 0


                elif gap < 0:
                    ci_lb = new_x
                    obj_lb= new_obj
                    fixed_value= grid[n_iter]


            elif state == 'final':

                if gap > 0:
                    ci_ub= new_x                
                    obj_ub= new_obj

                elif gap < 0:
                    ci_lb = new_x
                    obj_lb= new_obj


                if max(obj_ub - th, th - obj_lb) < th:
                    fixed_value = ci_lb + ((ci_ub-ci_lb)*(th-obj_lb)/(obj_ub-obj_lb))

                else:
                    fixed_value = (ci_ub + ci_lb)/2


            n_iter += 1
        
        if (ci_ub-ci_lb) < 0.01:
            break


    return new_x


def calc_ci_one_var_neg(model, var_type, i):

    org_sol = model.estimation_solution

    if var_type == 'vnet':
        org_value= org_sol['vnet_sol'][i]
        ci_lb= model.vnet_lb[i]


    elif var_type == 'c':
        org_value= org_sol['c_sol'][i]
        ci_lb= model.c_lb

    elif var_type == 'alpha':
        org_value= org_sol['alpha_sol'][i]
        ci_lb= model.c_lb
        
        
    ci_ub= org_value
    obj_ub= org_sol['obj']
    fixed_value= ci_lb

    th= chi2.ppf(0.95, 1)
    n_iter = 0

    state= 'max_bound'

    while n_iter <= 20:

        optim_kwargs= {
            'optim_type': 'ci',
            'prev_sol': org_sol,
            'var_index': i, 
            'fixed_value': fixed_value,
            'var_type': var_type}
        

        model.set_tracer(m_fun='est')
        new_sol = inst_estimate(model, **optim_kwargs)

        new_obj= new_sol['obj']
        if var_type == 'vnet':
            new_x= new_sol['vnet_sol'][i]
        elif var_type == 'c':
            new_x= new_sol['c_sol'][i]
        elif var_type == 'alpha':
            new_x= new_sol['alpha_sol'][i]


        if new_obj == 0:
            new_obj = 1e4
            new_x= fixed_value

        delta= new_obj - org_sol['obj']

        
        gap= delta - th

        if abs(gap) < 0.1 or (n_iter == 0 and delta < th):
            break

        else:

            if state == 'max_bound':
                grid= gen_power_grid(ci_lb, ci_ub, reverse=True)
                fixed_value= grid[n_iter]
                state = 'grid'


            elif state == 'grid':

                if gap > 0:
                    ci_lb= new_x
                    obj_lb= new_obj
                    
                    if max(obj_lb-th, th-obj_ub) < th:
                        fixed_value = ci_ub - ((ci_ub-ci_lb)*(th-obj_ub)/(obj_lb-obj_ub))
                    else:
                        fixed_value = (ci_ub + ci_lb)/2
                        
                    state = 'final'
                    n_iter= 0

                elif gap < 0:
                    ci_ub = new_x
                    obj_ub= new_obj
                    fixed_value= grid[n_iter]


            elif state == 'final':

                if gap > 0:
                    ci_lb= new_x
                    obj_lb= new_obj
                
                elif gap < 0:
                    ci_ub = new_x
                    obj_ub= new_obj

                if max(obj_lb-th, th-obj_ub) < th:
                    fixed_value = ci_ub - ((ci_ub-ci_lb)*(th-obj_ub)/(obj_lb-obj_ub))
                else:
                    fixed_value = (ci_ub + ci_lb)/2

        n_iter += 1

        if (ci_ub-ci_lb) < 0.01:
            break


    return new_x



def calc_ci_vnet_one_var(args):
    model, i = args
    d_pos = calc_ci_one_var_pos(model, 'vnet', i)
    d_neg = calc_ci_one_var_neg(model, 'vnet', i)

    print("Rxn {}\t{:.3f}\t{:.3f}\t{:.3f}".format(i, d_neg, model.estimation_solution['vnet_sol'][i], d_pos))

    return (d_neg, d_pos)


def calc_ci_c_one_var(args):
    model, i = args
    d_pos = calc_ci_one_var_pos(model, 'c', i)
    d_neg = calc_ci_one_var_neg(model, 'c', i)

    print("Pool {}\t{:.3f}\t{:.3f}\t{:.3f}".format(i, d_neg, model.estimation_solution['c_sol'][i], d_pos))

    return (d_neg, d_pos)



def calc_ci_alpha_one_var(args):
    model, i = args
    d_pos = calc_ci_one_var_pos(model, 'alpha', i)
    d_neg = calc_ci_one_var_neg(model, 'alpha', i)

    print("alpha {}\t{:.3f}\t{:.3f}\t{:.3f}".format(i, d_neg, model.estimation_solution['alpha_sol'][i], d_pos))
    # print('_' * 100)

    return (d_neg, d_pos)



def calc_ci(model):



    indices = range(len(model.estimation_solution['vnet_sol']))  
    with ProcessPoolExecutor(max_workers= model.n_workers) as executor:
        cis_vnet= list(executor.map(calc_ci_vnet_one_var, [(model, i) for i in indices]))  

    model.estimation_solution['cis_vnet']= cis_vnet




    indices = range(len(model.estimation_solution['c_sol']))  
    with ProcessPoolExecutor(max_workers=model.n_workers) as executor:
        cis_c= list(executor.map(calc_ci_c_one_var, [(model, i) for i in indices]))


    model.estimation_solution['cis_c']= cis_c


    indices = range(len(model.estimation_solution['alpha_sol']))  
    with ProcessPoolExecutor(max_workers=model.n_workers) as executor:
        cis_c= list(executor.map(calc_ci_alpha_one_var, [(model, i) for i in indices]))


    model.estimation_solution['cis_alpha']= cis_c






















# def calc_ci_one_var(model, var_type, i, dir_type):


#     d= 0
#     prev_sol = model.estimation_solution
#     step_size_ratio= 0.1

#     if var_type == 'vnet':
#         org_value= prev_sol['vnet_sol'][i]
#     elif var_type == 'c':
#         org_value= prev_sol['c_sol'][i]
    

#     if dir_type == 'pos' and var_type == 'vnet':
#         print("Rxn: {}\tValue: {:.2f}".format(i, org_value))
#     elif dir_type == 'pos' and var_type == 'c':
#         print("Met Pool: {}\tValue: {:.2f}".format(i, org_value))

#     th= chi2.ppf(0.90, 1)
#     delta= 0
#     rb= False

#     while delta < th and not rb:

#         step_size= abs(org_value) * step_size_ratio
#         d += step_size

#         if dir_type == 'pos':
#             fixed_value = org_value + d
#         elif dir_type == 'neg':
#             fixed_value = org_value - d

#         if var_type == 'vnet':
#             if fixed_value < model.vnet_lb[i]:
#                 fixed_value = model.vnet_lb[i]
#                 rb = True
#             elif fixed_value > model.vnet_ub[i]:
#                 fixed_value = model.vnet_ub[i]
#                 rb= True

#         if var_type == 'c':
#             if fixed_value < model.c_lb:
#                 fixed_value = model.c_lb
#                 rb= True
#             elif fixed_value > model.c_ub:
#                 fixed_value= model.c_ub
#                 rb= True


#         optim_kwargs= {
#             'optim_type': 'ci',
#             'prev_sol': prev_sol,
#             'var_index': i, 
#             'fixed_value': fixed_value,
#             'var_type': var_type}
        

#         model.set_tracer(m_fun='est')
#         new_sol = inst_estimate(model, **optim_kwargs)

#         diff=  new_sol['obj'] - prev_sol['obj']
#         delta= new_sol['obj'] - model.estimation_solution['obj']

#         if var_type == 'vnet':
#             new_x= new_sol['vnet_sol'][i]
#         elif var_type == 'c':
#             new_x= new_sol['c_sol'][i]

#         print("Step: {:.3f}\tNew Value: {:.3f}\tObj: {:.3f}\tDiff: {:.3f}\tDelta: {:.3f}".format(step_size, new_x, new_sol['obj'], diff, delta))
        
        
#         if diff < 0.01:
#             step_size_ratio *= 5
#         elif diff >=0.01 and diff >= 0.1:
#             step_size_ratio *= 2
#         elif diff >= 0.5:
#             step_size_ratio /= 2

#         gap= delta- th

#         if abs(gap) < 0.02 or rb:
#             break
#         elif gap < 0:
#             prev_sol = new_sol
#         elif gap > 0:
#             step_size_ratio = max(10e-4, step_size_ratio * (th - prev_sol['obj']) / diff)
#             delta= 0
#             d -= step_size

#     return new_x





# def calc_ci_optimization(model, var_index, step_size, prev_sol):

#     model.set_tracer(m_fun='est')

#     for rxn in model.rxns:
#         rxn.v = ca.MX.sym(rxn.rxn_id)

#     for emu in model.emu_net.V:
#         if emu.met in model.tracer:
#             emu.x0 = ca.DM(emu.MID).T
#             emu.x = ca.DM(emu.MID).T
#         else:
#             emu.x0 = ca.MX(emu.MID).T
#             emu.x = ca.MX.sym(str(emu), 1, emu.size + 1)

#     met_pools = []
#     met_pools_str = []

#     v = ca.vertcat(*[rxn.v for rxn in model.rxns])

#     target_emus_x_index = dict()
#     x_index_counter = 0

#     subnet_size = 1
#     subnet_number = 1

#     max_subnet_size = max([emu.size for emu in model.target_emus])

#     X_subnets = []
#     dXdt_subnets = []
#     X0_subnets = []

#     while subnet_size <= max_subnet_size:
#         subnet = EMUSubNetwork()
#         known_emus = []
#         unknown_emus = []

#         for emu_rxn in model.emu_net.E:
#             if emu_rxn.size == subnet_size:
#                 subnet.add_EMU_reaction(emu_rxn)

#         if not subnet:
#             subnet_size += 1
#             continue

#         for emu in subnet.V:
#             if emu.solved:
#                 known_emus.append(emu)
#             elif not emu.solved:
#                 unknown_emus.append(emu)

#         A = ca.MX.zeros(len(unknown_emus), len(unknown_emus))
#         B = ca.MX.zeros(len(unknown_emus), len(known_emus))

#         for unknown_emu_index, unknown_emu in enumerate(unknown_emus):
#             total_input_flux = ca.MX(0)

#             for emu_rxn in subnet.E:
#                 if emu_rxn.end == unknown_emu:

#                     if emu_rxn.start in unknown_emus:
#                         lhs_index = unknown_emus.index(emu_rxn.start)
#                         A[unknown_emu_index, lhs_index] += (emu_rxn.rxn.v * emu_rxn.rxn_coef)


#                     elif emu_rxn.start in known_emus:
#                         lhs_index = known_emus.index(emu_rxn.start)
#                         B[unknown_emu_index, lhs_index] -= (emu_rxn.rxn.v * emu_rxn.rxn_coef)

#                     total_input_flux += (emu_rxn.rxn.v * emu_rxn.rxn_coef)

#             A[unknown_emu_index, unknown_emu_index] -= total_input_flux

#         Y = ca.vertcat(*[emu.x for emu in known_emus])

#         M = []
#         for unknown_emu in unknown_emus:
#             met_id = unknown_emu.met.split('_')[0]
#             comp = unknown_emu.met.split('_')[1] if '_' in unknown_emu.met else None

#             if len(model.comp_alphas) > 0:
#                 comp_coef= model.comp_alphas[comp]
#             else:
#                 comp_coef = 1

#             if met_id in met_pools_str:
#                 idx = met_pools_str.index(met_id)
#                 met_pool = met_pools[idx]
#             else:
#                 met_pool = ca.MX.sym(met_id)
#                 met_pools.append(met_pool)
#                 met_pools_str.append(met_id)

#             M.append(met_pool * comp_coef)

#         M_inv = ca.diag(1 / ca.vertcat(*M))

#         X0 = ca.vertcat(*[unknown_emu.x0.T for unknown_emu in unknown_emus])
#         X = ca.vertcat(*[unknown_emu.x for unknown_emu in unknown_emus])
#         Xdot = M_inv @ (A @ X - B @ Y)

#         for row, emu in enumerate(unknown_emus):
#             emu.xdot = Xdot[row, :]
#             emu.solved = True

#             if emu in model.target_emus:
#                 target_emus_x_index[str(emu)] = (x_index_counter, x_index_counter + emu.size + 1)

#             x_index_counter += (emu.size + 1)

#         dXdt = ca.vertcat(*[emu.xdot.T for emu in unknown_emus])

#         X = ca.vertcat(*[unknown_emu.x.T for unknown_emu in unknown_emus])
#         X_subnets.append(X)

#         dXdt_subnets.append(dXdt)

#         X0_subnets.append(X0)

#         subnet_size += 1

#     X0 = ca.vertcat(*X0_subnets)
#     X = ca.vertcat(*X_subnets)
#     dXdt = ca.vertcat(*dXdt_subnets)

#     c = ca.vertcat(*met_pools)

#     p = ca.vertcat(v, c)

#     dae = {"x": X, "p": p, "ode": dXdt}

#     time_points, emu2mid_measured = model.inst_measured_MIDs

#     integrator_opts = {"linear_multistep_method": "bdf",
#                        "nonlinear_solver_iteration": "newton",
#                        "abstol": 1e-5, "reltol": 1e-5, "max_num_steps": 2000}

#     F = ca.integrator("F", "cvodes", dae, 0, time_points, integrator_opts)

#     J_mid = ca.MX(0)

#     xk = F(x0=X0, p=p)["xf"]

#     for emu in model.target_emus:
#         emu_mid_sim = xk[target_emus_x_index[str(emu)][0]:target_emus_x_index[str(emu)][1], :]
#         emu_mid_meas = ca.DM(emu2mid_measured[str(emu)])
#         mid_res = emu_mid_sim - emu_mid_meas
#         J_mid += (ca.sumsqr(mid_res) / (model.def_mid_sd ** 2))

#     J_v = ca.MX(0)
#     for i, v_mes in model.measured_rxn2flux.items():
#         v_res = v[i] - ca.MX(v_mes)
#         J_v += ca.sumsqr(v_res)

#     J = J_mid + J_v

#     S = ca.DM(model.S)
#     g = S @ v

#     nlp = {"x": p, "f": J, "g": g}



#     ipopt_opts = {
#         "expand": True,
#         "print_time": False,
#         "ipopt": {
#             "linear_solver": "mumps",
#             "mu_strategy": "adaptive",

#             "hessian_approximation": "limited-memory",
#             "limited_memory_max_history": 20,

#             "nlp_scaling_method": "gradient-based",
#             "nlp_scaling_max_gradient": 100.0,

#             "warm_start_init_point": "yes",
#             "warm_start_bound_push": 1e-8,
#             "warm_start_bound_frac": 1e-8,
#             "warm_start_slack_bound_push": 1e-8,
#             "warm_start_slack_bound_frac": 1e-8,
#             "warm_start_mult_bound_push": 1e-8,
#             "fixed_variable_treatment": "make_constraint",


#             "tol": 1e-4,
#             "max_iter": 100,

#             "acceptable_tol": 1e-3,
#             "acceptable_iter": 3,
#             "acceptable_dual_inf_tol": 1e-2,
#             "acceptable_compl_inf_tol": 1e-2,
#             "acceptable_constr_viol_tol": 1e-8,
#             "acceptable_obj_change_tol": 1e-3,

#             "print_level": 0,
#             "print_timing_statistics": "no",
#             "sb": "yes"
#         }
#     }

#     solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)

#     v_lb = ca.DM(model.lb)
#     v_ub = ca.DM(model.ub)

#     c_size = c.numel()
#     v_size = v.numel()

#     c_lb = model.c_lb * ca.DM.ones(c_size)
#     c_ub = model.c_ub * ca.DM.ones(c_size)

#     lbg = np.zeros(model.nMets)
#     ubg = np.zeros(model.nMets)

#     lbx = ca.vertcat(v_lb, c_lb)
#     ubx = ca.vertcat(v_ub, c_ub)



#     x0 = prev_sol['x_sol'].copy()

#     x0[var_index] += step_size
#     reached_bound= False
#     if x0[var_index] <= lbx[var_index]:
#         x0[var_index] = lbx[var_index]
#         reached_bound= True
#     elif x0[var_index] >= ubx[var_index]:
#         x0[var_index] = ubx[var_index]
#         reached_bound= True
    
    
#     lbx[var_index]= x0[var_index]
#     ubx[var_index]= x0[var_index]


#     sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, lam_x0=prev_sol['lam_x'], lam_g0=prev_sol['lam_g'])


#     x_sol = np.array(sol["x"]).squeeze()
#     obj= float(sol["f"])
#     v_sol= x_sol[:v_size]
#     c_sol= x_sol[v_size:]

#     met_pools_sol= (met_pools_str, c_sol)

#     lam_x= np.array(sol["lam_x"]).squeeze()
#     lam_g= np.array(sol["lam_g"]).squeeze()

#     return {'obj': obj, 'x_sol': x_sol, 'v_sol': v_sol,
#             'met_pools_sol': met_pools_sol, 'lam_x': lam_x, 'lam_g': lam_g, 'rb': reached_bound}











# from scipy.optimize import least_squares, linprog, LinearConstraint, minimize, Bounds

#
# # def flux_balance_analysis(model):
# #     # objective_met= model.mets.index(model.target_emu.met)
# #     # objective_rxn= np.where(model.S[objective_met])[0].item()
# #     c= np.zeros(model.nRxns)
# #     c[2]= -1    # corresponding to v2 for this example
# #     bounds= list((lb, ub) for lb, ub in zip(model.lb, model.ub))
# #     result= linprog(c, A_eq= model.S, b_eq= np.zeros(model.nMets),
# #                     bounds= bounds)
# #     return result.x
# #
# #
# # def ss_residuals(u, model, mid_exp, rxn2flux):
# #     v= model.stoichiometric_null_space @ u
# #     simulation_options= {'fluxes': v, 'simulation_type': 'ss', 'net_xch_fluxes': True}
# #     mid_sim= model.simulate_MIDs(**simulation_options)
# #     mid_res_all= []
# #     for emu, emu_mid_sim in mid_sim.items():
# #         mid_res= emu_mid_sim[:len(mid_exp[str(emu)])] - mid_exp[str(emu)]
# #         mid_res_all.extend(mid_res)
# #     v_res= np.array([v[i] - v_mes for i, v_mes in rxn2flux.items()])
# #     return np.concatenate((mid_res_all, v_res))
# #
# #
# # def inst_residuals(x, model, mid_exp, rxn2flux, time_span, comp_alpha):
# #     u_size= model.stoichiometric_null_space.shape[1]
# #     v = model.stoichiometric_null_space @ x[:u_size]
# #     c= x[u_size:]
# #     met_pools= {emu: const for emu,const in zip(model.req_c, c)}
# #     simulation_options = {'fluxes': v, 'simulation_type': 'inst', 'met_pools': met_pools,
# #                           'time_span': time_span, 'comp_alpha': comp_alpha, 'net_xch_fluxes': True}
# #     print(v,c)
# #     mid_sim = model.simulate_MIDs(**simulation_options)
# #     mid_res_all = []
# #     for emu, emu_mid_sim in mid_sim.items():
# #         t_mid_exp_tup= mid_exp[str(emu)]
# #         for t, emu_mid_exp in t_mid_exp_tup:
# #             emu_mid_sim_t= emu_mid_sim.inst_fun(t)
# #             mid_res = emu_mid_sim_t - emu_mid_exp
# #             mid_res_all.extend(mid_res)
# #     v_res = np.array([v[i] - v_mes for i, v_mes in rxn2flux.items()])
# #     return np.concatenate((mid_res_all, v_res))
#
#
# def ss_obj(u, model, mid_exp, rxn2flux):
#     v = model.stoichiometric_null_space @ u
#
#     simulation_options = {'fluxes': v, 'simulation_type': 'ss'}
#     mid_sim = model.simulate_MIDs(**simulation_options)
#
#     mid_res_all = []
#     for emu, emu_mid_sim in mid_sim.items():
#         mid_res = emu_mid_sim[:len(mid_exp[str(emu)])] - mid_exp[str(emu)]
#         mid_res_all.extend(mid_res)
#     v_res = np.array([(v[i] - v_mes)**2 for i, v_mes in rxn2flux.items()])
#
#     total_loss= np.sum(mid_res_all) + np.sum(v_res)
#
#     return total_loss
#
#
# def inst_obj(x, model, mid_exp, rxn2flux, time_span, comp_alpha):
#     u_size= model.stoichiometric_null_space.shape[1]
#
#     # v = model.stoichiometric_null_space @ x[:u_size]
#     v = model.stoichiometric_null_space @ x
#
#
#     # c= x[u_size:]
#     c_size = len(model.req_c)
#     c = model.def_met_const * np.ones(c_size)
#
#
#     met_pools= {emu: const for emu,const in zip(model.req_c, c)}
#     simulation_options = {'fluxes': v, 'simulation_type': 'inst', 'met_pools': met_pools,
#                           'time_span': time_span, 'comp_alpha': comp_alpha}
#     mid_sim = model.simulate_MIDs(**simulation_options)
#     mid_res_all = []
#     for emu, emu_mid_sim in mid_sim.items():   # TODO: I changed the returning dict, it should be checked that this works
#         t_mid_exp_tup= mid_exp[str(emu)]
#         for t, emu_mid_exp in t_mid_exp_tup:
#             emu_mid_sim_t= emu_mid_sim.inst_fun(t)
#             mid_res = emu_mid_sim_t - emu_mid_exp
#             cov_matrix= np.eye(mid_res.size)/(model.def_mid_sd**2)
#             mid_loss= mid_res@cov_matrix@mid_res
#             mid_res_all.append(mid_loss.item())
#
#     v_res = np.array([(v[i] - v_mes)**2 / (model.def_v_sd ** 2) for i, v_mes in rxn2flux.items()])
#
#     sum_mid_loss= np.sum(mid_res_all)
#     sum_v_loss= np.sum(v_res)
#     total_loss =  sum_mid_loss+ sum_v_loss
#
#     print('mid loss: {:.6f}\tflux loss: {:.6f}\ttotal loss: {:.6f}'.format(sum_mid_loss, sum_v_loss, total_loss))
#
#     return total_loss
#
#
#
#
# def estimate_flux(model, rxn2flux, **kwargs):
#
#     # steady_state = LinearConstraint(model.S, lb=np.zeros(model.nMets), ub=np.zeros(model.nMets))
#     # # v0= flux_balance_analysis(model)
#     #
#     # v0= np.array([0.13, 3.71, 3.29, 4.54, 12.78, 0.11, 8.36, 10.29, 21.67, 3.23, 29.94, 30.39, 27.92, 3.08, 0.61,
#     #              2.36, 1.18, 2.04, 9.83, 2.66, 6.32, 14.74, 14.70, 2.74, 10.53, 16.15, 12.19, 4.90, 1.07, 214.28,
#     #              213.58, 30.01, 30.72, 3.13, 3.13, 1.37, 0.40, 0.43, 0.02, 0.67, 0.37, 2.85, 2.75, 2.01, 1.74,
#     #              15.05, 15.20, 0.10, 7.85, 7.76, 0.50, 1.05, 0.03, 0.04, 0.04, 0.02, 4.30, 4.27, 10.0, 0.24])
#
#     # flux_bounds= Bounds(model.lb, model.ub)
#     #
#     # solution= minimize(objective_function,
#     #                  v0,
#     #                  args= (model, mids_measured, rxn2flux),
#     #                  method= 'SLSQP',    # COBYLA, COBYQA, SLSQP and trust-constr
#     #                  constraints= [steady_state],
#     #                  bounds= flux_bounds,
#     #                  tol= 1e-12)
#
#
#     # v0 = np.random.uniform(low=model.lb, high= model.ub, size=model.nRxns)
#
#
#     # feas_sol= linprog(np.zeros(len(v0)),
#     #                   A_eq= model.S, b_eq= np.zeros(model.S.shape[0]),
#     #                   bounds= list(zip(model.lb, model.ub)))
#     #
#     # # prediction_options= {'xch_fluxes':{'co2in':10}, 'obj_fun': 'biomass'}
#     # # fba_sol= model.fba(**prediction_options)
#     #
#     # v0_feas= feas_sol.x
#
#
#     u0, _, _, _ = np.linalg.lstsq(model.stoichiometric_null_space, v0, rcond=None)
#
#     estimation_type= kwargs.get('estimation_type', 'ss')
#     mids_measured = kwargs['measured_MIDs']
#
#
#     ### using least_square
#     # if estimation_type == 'ss':
#     #     solution= least_squares(ss_residuals,u0,
#     #                             args=(model, mids_measured, rxn2flux),
#     #                             method= 'trf',                           # trf, dogbox, lm
#     #                             ftol= 1e-12, xtol= 1e-12, gtol= 1e-12,
#     #                             bounds=(-1000,1000))
#     #     v_solution = model.stoichiometric_null_space @ solution.x
#     #
#     #     return v_solution
#     #
#     # elif estimation_type == 'inst':
#     #     time_span = kwargs.get('time_span', None)
#     #     comp_alpha = kwargs.get('comp_alpha', None)
#     #
#     #     if not model.req_c:
#     #         inst_simulation_options= {
#     #             'simulation_type': 'inst',
#     #             'time_span': time_span,
#     #             'comp_alpha': comp_alpha,
#     #             'fluxes': model.stoichiometric_null_space @ u0,
#     #             'presolve': True}
#     #         model.simulate_MIDs(**inst_simulation_options)
#     #
#     #     c0 = model.def_met_const * np.ones(len(model.req_c))
#     #     x0= np.concatenate((u0, c0))
#     #
#     #     solution = least_squares(inst_residuals, x0,
#     #                              args=(model, mids_measured, rxn2flux, time_span, comp_alpha),
#     #                              method='trf',  # trf, dogbox, lm
#     #                              ftol=1e-12, xtol=1e-12, gtol=1e-12)
#     #
#     #     u_size = model.stoichiometric_null_space.shape[1]
#     #     v_solution = model.stoichiometric_null_space @ solution.x[:u_size]
#     #     c_solution = solution.x[u_size:]
#     #
#     #
#     #     return v_solution, c_solution
#
#
#     v_size= model.nRxns
#     vnet_size = v_size - len(model.rev_rxns)
#     T = np.eye(vnet_size)
#     vnet_lb = np.zeros(vnet_size)
#     vnet_ub = model.max_flux * np.ones(vnet_size)
#
#     for count, rxn in enumerate(model.rev_rxns):
#         rxn_index = model.rxns.index(rxn)
#         back_rxn_index = rxn_index + 1
#         temp = np.zeros(vnet_size)
#         temp[rxn_index - count] = -1
#         T = np.insert(T, rxn_index + 1, temp, axis=1)
#         vnet_lb[rxn_index - count] = -model.max_flux
#
#     rxn2T= dict()
#
#
#     ### using minimize
#     if estimation_type == 'ss':
#
#         # # solving with trust-const
#         # irrev_const = LinearConstraint(model.stoichiometric_null_space,
#         #                                model.lb, model.ub)
#         # vnet_const = LinearConstraint(T @ model.stoichiometric_null_space, vnet_lb, vnet_ub)
#         #
#         # solution= minimize(ss_obj, u0,
#         #                    args= (model, mids_measured, rxn2flux),
#         #                    method= 'trust-constr',    # COBYLA, COBYQA, SLSQP and trust-constr
#         #                    constraints= [irrev_const, vnet_const],
#         #                    tol= 1e-12)
#
#         # solving with SLSQP
#         def v_const_lb_ss(u):
#             return model.stoichiometric_null_space @ u - model.lb
#
#         def vnet_const_lb_ss(u):
#             return T @ model.stoichiometric_null_space @ u - vnet_lb
#
#         def vnet_const_ub_ss(u):
#             return -T @ model.stoichiometric_null_space @ u + vnet_ub
#
#         cons= ({'type': 'ineq', 'fun': v_const_lb_ss},
#                {'type': 'ineq', 'fun': vnet_const_lb_ss},
#                {'type': 'ineq', 'fun': vnet_const_ub_ss})
#
#         solution= minimize(ss_obj, u0,
#                            args= (model, mids_measured, rxn2flux),
#                            method= 'SLSQP',    # COBYLA, COBYQA, SLSQP and trust-constr
#                            constraints= cons)
#
#         v_solution = model.stoichiometric_null_space @ solution.x
#         vnet_solution = T @ model.stoichiometric_null_space @ solution.x
#
#         return v_solution, vnet_solution
#
#     elif estimation_type == 'inst':
#         time_span = kwargs.get('time_span', None)
#         comp_alpha = kwargs.get('comp_alpha', None)
#
#         if not model.req_c:
#             inst_simulation_options= {
#                 'simulation_type': 'inst',
#                 'time_span': time_span,
#                 'comp_alpha': comp_alpha,
#                 'fluxes': model.stoichiometric_null_space @ u0,
#                 'presolve': True}
#             model.simulate_MIDs(**inst_simulation_options)
#
#         c_size= len(model.req_c)
#         u_size= model.stoichiometric_null_space.shape[1]
#         c0 = model.def_met_const * np.ones(c_size)
#         x0= np.concatenate((u0, c0), axis=0)
#
#         # Nu >= 0, c >= 0
#         A1= np.concatenate((np.concatenate((model.stoichiometric_null_space, np.zeros((v_size, c_size))), axis=1),
#                             np.concatenate((np.zeros((c_size, u_size)), np.eye(c_size)), axis= 1)), axis=0)
#
#         A1_lb= np.concatenate((model.lb, np.zeros(c_size)), axis=0)
#         A1_ub= np.concatenate((model.ub, 100 * np.ones(c_size)), axis=0)
#
#         # irrev_const = LinearConstraint(A1, np.zeros(v_size+c_size))
#         # irrev_const = LinearConstraint(A1, np.concatenate((model.lb, np.zeros(c_size)), axis=0),
#         #                                np.concatenate((model.ub, 100 * np.ones(c_size)), axis=0))
#
#         # v_lb =< TNu =< v_ub
#         A2= T @ np.concatenate((model.stoichiometric_null_space, np.zeros((v_size, c_size))), axis=1)
#         # vnet_const = LinearConstraint(A2, vnet_lb, vnet_ub)
#         #
#         #
#         # irrev_const = LinearConstraint(model.stoichiometric_null_space,
#         #                                model.lb, model.ub)
#         # vnet_const = LinearConstraint(T @ model.stoichiometric_null_space, vnet_lb, vnet_ub)
#
#         # solution= minimize(inst_obj, x0,
#         #                    args= (model, mids_measured, rxn2flux, time_span, comp_alpha),
#         #                    method= 'trust-constr',    # COBYLA, COBYQA, SLSQP and trust-constr
#         #                    constraints= [irrev_const, vnet_const],
#         #                    tol= 1e-12, options= {'disp': True})
#
#         # solving with SLSQP
#
#
#
#         def x_const_lb_inst(x):
#             # return A1 @ x - A1_lb
#             return model.stoichiometric_null_space @ x - model.lb
#
#         def x_const_ub_inst(x):
#             # return -A1 @ x + A1_ub
#             return -model.stoichiometric_null_space @ x + model.ub
#
#         def vnet_const_lb_inst(x):
#             # return A2 @ x - vnet_lb
#             return T @ model.stoichiometric_null_space @ x - vnet_lb
#
#         def vnet_const_ub_inst(x):
#             # return -A2 @ x + vnet_ub
#             return -T @ model.stoichiometric_null_space @ x + vnet_ub
#
#         cons = ({'type': 'ineq', 'fun': x_const_lb_inst},
#                 {'type': 'ineq', 'fun': x_const_ub_inst},
#                 {'type': 'ineq', 'fun': vnet_const_lb_inst},
#                 {'type': 'ineq', 'fun': vnet_const_ub_inst})
#
#         solution= minimize(inst_obj, u0,
#                            args= (model, mids_measured, rxn2flux, time_span, comp_alpha),
#                            method= 'SLSQP',    # COBYLA, COBYQA, SLSQP and trust-constr
#                            constraints= cons)
#
#
#         v_solution = model.stoichiometric_null_space @ solution.x[:u_size]
#         vnet_solution = T @ model.stoichiometric_null_space @ solution.x[:u_size]
#         c_solution = solution.x[u_size:]
#
#         return v_solution, vnet_solution, c_solution
#
#
#     else:
#         return 0









# def inst_estimate(model, seed, **kwargs):

#     null_space = ca.DM(model.stoichiometric_null_space)
#     u_size = null_space.shape[1]
#     u= ca.MX.sym('u', u_size)
#     v= ca.mtimes(null_space, u)

#     for i, rxn in enumerate(model.rxns):
#         rxn.v= v[i]

#     # for rxn in model.rxns:
#     #     rxn.v= ca.MX.sym(rxn.rxn_id)

#     for emu in model.emu_net.V:
#         if emu.met in model.tracer:
#             emu.x0= ca.DM(emu.MID).T
#             emu.x= ca.DM(emu.MID).T
#         else:
#             emu.x0= ca.MX(emu.MID).T
#             emu.x= ca.MX.sym(str(emu), 1, emu.size + 1)

#     met_pools= []
#     met_pools_str= []

#     # v = ca.vertcat(*[rxn.v for rxn in model.rxns])
#     # c= ca.vertcat(*[emu.c for emu in model.emu_net.V])

#     target_emus_x_index = dict()
#     x_index_counter = 0

#     subnet_size = 1
#     subnet_number = 1

#     max_subnet_size = max([emu.size for emu in model.target_emus])

#     X_subnets= []
#     dXdt_subnets= []
#     X0_subnets = []

#     while subnet_size <= max_subnet_size:
#         subnet = EMUSubNetwork()
#         known_emus = []
#         unknown_emus = []

#         # looping through the EMU rxns of the EMU subnetwork and collect the ones with the defined size
#         for emu_rxn in model.emu_net.E:
#             if emu_rxn.size == subnet_size:
#                 subnet.add_EMU_reaction(emu_rxn)

#         if not subnet:
#             subnet_size += 1
#             continue

#         for emu in subnet.V:
#             if emu.solved:
#                 known_emus.append(emu)
#             elif not emu.solved:
#                 unknown_emus.append(emu)

#         # initializing the A and B matrices for the equation: AX=BY
#         # A is the multiplier for unknown MID variables
#         A = ca.MX.zeros(len(unknown_emus), len(unknown_emus))
#         # B is the multiplier for known MID variables
#         B = ca.MX.zeros(len(unknown_emus), len(known_emus))

#         for unknown_emu_index, unknown_emu in enumerate(unknown_emus):
#             total_input_flux = ca.MX(0)

#             # identifying the emu rxns producing the EMU with unknown MID
#             for emu_rxn in subnet.E:
#                 if emu_rxn.end == unknown_emu:

#                     # assigning values to A and B for the in emu rxns based on the mass balance equations
#                     if emu_rxn.start in unknown_emus:
#                         lhs_index = unknown_emus.index(emu_rxn.start)
#                         A[unknown_emu_index, lhs_index] += (emu_rxn.rxn.v * emu_rxn.rxn_coef)


#                     elif emu_rxn.start in known_emus:
#                         lhs_index = known_emus.index(emu_rxn.start)
#                         B[unknown_emu_index, lhs_index] -= (emu_rxn.rxn.v * emu_rxn.rxn_coef)

#                     total_input_flux += (emu_rxn.rxn.v * emu_rxn.rxn_coef)

#             # assigning values to A for the EMU rxns, in which unknown emu is precursor
#             A[unknown_emu_index, unknown_emu_index] -= total_input_flux

#         # creating matrix Y, which is MIDs for determined EMUs
#         Y = ca.vertcat(*[emu.x for emu in known_emus])


#         # solving the INST ODEs
#         # for i, unknown_emu in enumerate(unknown_emus):
#         M= []
#         for unknown_emu in unknown_emus:
#             met_id= unknown_emu.met.split('_')[0]
#             comp= unknown_emu.met.split('_')[1] if '_' in unknown_emu.met else None

#             # if hasattr(model, 'met_pools'):
#             #     met_const_value = model.met_pools[met_id]
#             # else:
#             #     met_const_value= model.def_met_const

#             if model.comp_alphas != None:
#                 comp_coef= model.comp_alphas[comp]
#             else:
#                 comp_coef = 1

#             if met_id in met_pools_str:
#                 idx= met_pools_str.index(met_id)
#                 met_pool= met_pools[idx]
#             else:
#                 met_pool= ca.MX.sym(met_id)
#                 met_pools.append(met_pool)
#                 met_pools_str.append(met_id)

#             M.append(met_pool * comp_coef)

#         # M_inv = ca.diag(1 / ca.DM(M))
#         M_inv = ca.diag(1 / ca.vertcat(*M))

#         X0 = ca.vertcat(*[unknown_emu.x0.T for unknown_emu in unknown_emus])

#         X = ca.vertcat(*[unknown_emu.x for unknown_emu in unknown_emus])

#         Xdot= M_inv @ (A @ X - B @ Y)

#         for row, emu in enumerate(unknown_emus):
#             emu.xdot= Xdot[row, :]
#             emu.solved = True

#             if emu in model.target_emus:
#                 target_emus_x_index[str(emu)]= (x_index_counter, x_index_counter + emu.size + 1)

#             x_index_counter += (emu.size + 1)


#         dXdt= ca.vertcat(*[emu.xdot.T for emu in unknown_emus])

#         X = ca.vertcat(*[unknown_emu.x.T for unknown_emu in unknown_emus])
#         X_subnets.append(X)

#         dXdt_subnets.append(dXdt)

#         X0_subnets.append(X0)

#         subnet_size += 1

#     X0= ca.vertcat(*X0_subnets)
#     X= ca.vertcat(*X_subnets)
#     dXdt= ca.vertcat(*dXdt_subnets)

#     c = ca.vertcat(*met_pools)
#     T= ca.DM(model.T)
#     vnet= T @ v
#     p= ca.vertcat(u, c)

#     dae= {"x": X, "p": p, "ode": dXdt}
#     # dae = {"x": X, "p": v, "ode": dXdt}
#     # dae = {"x": X, "p": u, "ode": dXdt}


#     time_points, emu2mid_measured = model.inst_measured_MIDs


#     integrator_opts= {"linear_multistep_method": "bdf",
#                       "nonlinear_solver_iteration": "newton",
#                       "abstol": 1e-4, "reltol": 1e-4, "max_num_steps": 2000}

#     F= ca.integrator("F", "cvodes", dae, 0, time_points, integrator_opts)


#     J_mid = ca.MX(0)

#     # xk = F(x0=X0, p=u)["xf"]
#     # xk = F(x0=X0, p=v)["xf"]
#     xk = F(x0= X0, p= p)["xf"]

#     for emu in model.target_emus:
#         emu_mid_sim= xk[target_emus_x_index[str(emu)][0]:target_emus_x_index[str(emu)][1],:]
#         emu_mid_meas= ca.DM(emu2mid_measured[str(emu)])
#         mid_res = emu_mid_sim - emu_mid_meas
#         J_mid += (ca.sumsqr(mid_res) / (model.def_mid_sd ** 2))


#     J_v = ca.MX(0)
#     for i, v_mes in model.measured_rxn2flux.items():
#         v_res = v[i] - ca.MX(v_mes)
#         J_v += ca.sumsqr(v_res)

#     J = J_mid + J_v

#     # S = ca.DM(model.S)
#     # g = S @ v  

#     g1= null_space @ u
#     g2= T @ null_space @ u
#     g= ca.vertcat(g1, g2)

#     nlp = {"x": p, "f": J, "g": g}
#     # nlp = {"x": v, "f": J, "g": g}
#     # nlp = {"x": u, "f": J, "g": g}

#     ipopt_opts = {
#         "expand": True,
#         "print_time": False,
#         "ipopt": {
#             "linear_solver": "mumps",
#             "mu_strategy": "adaptive",

#             "hessian_approximation": "limited-memory",
#             "limited_memory_max_history": 10,

#             "nlp_scaling_method": "gradient-based",
#             "nlp_scaling_max_gradient": 100.0,

#             "tol": 1e-6,
#             "max_iter": 500,

#             "acceptable_tol": 1e-4,
#             "acceptable_iter": 5,
#             "acceptable_dual_inf_tol": 1e-3,
#             "acceptable_compl_inf_tol": 1e-3,
#             "acceptable_constr_viol_tol": 1e-8,
#             "acceptable_obj_change_tol": 1e-6,

#             "print_level": 5,
#             "print_timing_statistics": "no",
#             "sb": "yes"
#         }
#     }
    
#     solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)

#     # v_lb = ca.DM(model.lb)
#     # v_ub = ca.DM(model.ub)

#     # vnet_lb= ca.DM(model.vnet_lb)
#     # vnet_ub= ca.DM(model.vnet_ub)

#     c_size = c.numel()
#     v_size= v.numel()

#     c_lb = model.c_lb * ca.DM.ones(c_size)
#     c_ub = model.c_ub * ca.DM.ones(c_size)

#     # lbg = np.zeros(model.nMets)
#     # ubg = np.zeros(model.nMets)

#     lbg1= ca.DM(model.lb)
#     ubg1= ca.DM(model.ub)

#     lbg2= ca.DM(model.vnet_lb)
#     ubg2= ca.DM(model.vnet_ub)

#     lbg= ca.vertcat(lbg1, lbg2)
#     ubg= ca.vertcat(ubg1, ubg2)


#     # v0 = ca.DM.ones(model.nRxns, 1)
#     # u0 = ca.DM.ones(u_size, 1)
#     # sol = solver(x0=v0, lbx=v_lb, ubx=v_ub, lbg=lbg, ubg=ubg)


#     rng = np.random.default_rng(seed)
#     # v0 = ca.DM(rng.uniform(low=model.lb, high= model.ub, size=model.nRxns))
#     # v0 = ca.DM(rng.uniform(low=model.vnet_lb, high= model.vnet_ub, size=model.vnet_size))
#     u0= ca.DM(rng.uniform(low=-100, high= 100, size= u_size))
#     c0= ca.DM(rng.uniform(model.c_lb, high= model.c_ub, size=c_size))

#     # x0 = ca.vertcat(v0, c0)
#     x0 = ca.vertcat(u0, c0)

#     u_lb= -ca.inf * ca.DM.ones(u_size)
#     u_ub= ca.inf * ca.DM.ones(u_size)
    
#     lbx = ca.vertcat(u_lb, c_lb)
#     ubx = ca.vertcat(u_ub, c_ub)
    
    
#     # lbx = ca.vertcat(v_lb, c_lb)
#     # ubx = ca.vertcat(v_ub, c_ub)
#     # lbx= ca.vertcat(vnet_lb, c_lb)
#     # ubx= ca.vertcat(vnet_ub, c_ub)


#     # feas_sol= linprog(np.zeros(len(v0)),
#     #                   A_eq= model.S, b_eq= np.zeros(model.S.shape[0]),
#     #                   bounds= list(zip(model.lb, model.ub)))
#     # v0_feas= feas_sol.x
#     # u0, _, _, _ = np.linalg.lstsq(model.stoichiometric_null_space, v0_feas, rcond=None)
#     # u0= ca.DM(u0)

#     sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
#     # sol = solver(x0=x0, lbg=lbg, ubg=ubg)
#     # sol = solver(x0=v0, lbx=v_lb, ubx=v_ub, lbg=lbg, ubg=ubg)
#     # u_sol = np.array(sol["x"]).squeeze()
#     x_sol = np.array(sol["x"]).squeeze()
#     obj= float(sol["f"])
#     u_sol= x_sol[:u_size]
#     v_sol= model.stoichiometric_null_space @ u_sol
#     vnet_sol= model.T @ v_sol
#     # v_sol= x_sol[:v_size]
#     c_sol= x_sol[u_size:]

#     met_pools_sol= (met_pools_str, c_sol)

#     lam_x= np.array(sol["lam_x"]).squeeze()
#     lam_g= np.array(sol["lam_g"]).squeeze()

#     return {'obj': obj, 'x_sol': x_sol, 'v_sol': v_sol, 'vnet_sol': vnet_sol,
#             'met_pools_sol': met_pools_sol, 'lam_x': lam_x, 'lam_g': lam_g}






        #
        # elif not isotopically_stationary:
        #
        #     # solving the INST ODEs
        #     time_span = model.time_span
        #
        #     # for i, unknown_emu in enumerate(unknown_emus):
        #     M = []
        #     for unknown_emu in unknown_emus:
        #         met_id = unknown_emu.met.split('_')[0]
        #         comp = unknown_emu.met.split('_')[1] if '_' in unknown_emu.met else None
        #         met_const_value = model.met_pools[met_id] if model.met_pools else model.def_met_const
        #         comp_coef = model.comp_alphas.get(comp, 1)
        #         M.append(met_const_value * comp_coef)
        #
        #         if presolve and met_id not in req_c:
        #             req_c.append(met_id)
        #
        #     M_inv = np.diag([1 / m for m in M])
        #     X0 = np.array([unknown_emu.MID for unknown_emu in unknown_emus])
        #
        #     def emu_balance(t, X_flat, known_emus, M_inv, A, B, org_shape):
        #         X = X_flat.reshape(org_shape)
        #         Y = np.array([known_emu.inst_fun(t) for known_emu in known_emus])
        #         dxdt = M_inv @ (A @ X - B @ Y)
        #         return dxdt.flatten()
        #
        #     sol = solve_ivp(emu_balance, time_span, X0.flatten(), method='BDF',
        #                     args=(known_emus, M_inv, A, B, X0.shape), dense_output=True, rtol=1e-12, atol=1e-12)
        #
        #     for emu in model.emu_net.V:
        #         if emu in unknown_emus:
        #             index = unknown_emus.index(emu)
        #             mdv_size = subnet_size + 1
        #             emu.inst_fun = lambda t, idx=index, size=mdv_size, solution=sol: solution.sol(t)[
        #                                                                              idx * size: idx * size + size]
        #             emu.solved = True
        #
        #             if emu in model.target_emus:
        #                 target_emus_MIDs[str(emu)] = emu
        #
        # subnet_size += 1



    # return target_emus_MIDs
