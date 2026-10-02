import numpy as np
from core.emu import EMUSubNetwork
from scipy.integrate import solve_ivp


def simulate_MIDs(model, isotopically_stationary= True, verbose= False, presolve= False):
    target_emus_MIDs= dict()

    subnet_size= 1
    subnet_number= 1

    max_subnet_size= max([emu.size for emu in model.target_emus])

    if presolve:
        req_c= []

    while subnet_size <= max_subnet_size:
        subnet= EMUSubNetwork()
        known_emus = []
        unknown_emus = []


        # looping through the EMU rxns of the EMU subnetwork and collect the ones with the defined size
        # TODO: implement the connectivity
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
        A= np.zeros((len(unknown_emus), len(unknown_emus)))
        # B is the multiplier for known MID variables
        B= np.zeros((len(unknown_emus), len(known_emus)))



        for unknown_emu_index, unknown_emu in enumerate(unknown_emus):
            total_input_flux = 0

            # identifying the emu rxns producing the EMU with unknown MID
            for emu_rxn in subnet.E:
                if emu_rxn.end == unknown_emu:

                    # assigning values to A and B for the in emu rxns based on the mass balance equations
                    if emu_rxn.start in unknown_emus:
                        lhs_index= unknown_emus.index(emu_rxn.start)
                        A[unknown_emu_index, lhs_index] += (emu_rxn.rxn.flux * emu_rxn.rxn_coef)


                    elif emu_rxn.start in known_emus:
                        lhs_index= known_emus.index(emu_rxn.start)
                        B[unknown_emu_index, lhs_index] -= (emu_rxn.rxn.flux * emu_rxn.rxn_coef)

                    total_input_flux += (emu_rxn.rxn.flux * emu_rxn.rxn_coef)

            # assigning values to A for the EMU rxns, in which unknown emu is precursor
            A[unknown_emu_index, unknown_emu_index] -= total_input_flux

        # creating matrix Y, which is MIDs for determined EMUs
        Y= np.vstack([emu.MID for emu in known_emus])


        if isotopically_stationary:
            # solving the equations for isotopically stationary
            X= np.linalg.solve(A, B @ Y)

            for emu in model.emu_net.V:
                if emu in unknown_emus:
                    index = unknown_emus.index(emu)
                    solved_MID= X[index]
                    emu.MID= solved_MID
                    emu.solved = True


                    if emu in model.target_emus:
                        if verbose:
                            print(emu)
                            for index, MID in enumerate(solved_MID):
                                print('M{}:\t{:.2f}%'.format(index, MID * 100))

                        target_emus_MIDs[emu] = solved_MID


        elif not isotopically_stationary:

            # solving the INST ODEs
            time_span = model.time_span

            # for i, unknown_emu in enumerate(unknown_emus):
            M= []
            for unknown_emu in unknown_emus:
                met_id= unknown_emu.met.split('.')[0]
                comp= unknown_emu.met.split('.')[1] if '.' in unknown_emu.met else None
                met_const_value = model.met_pools[met_id] if model.met_pools else model.def_met_const
                comp_coef= model.comp_alphas.get(comp, 1)
                M.append(met_const_value * comp_coef)

                if presolve and met_id not in req_c:
                    req_c.append(met_id)

            M_inv= np.diag([1/m for m in M])
            X0 = np.array([unknown_emu.MID for unknown_emu in unknown_emus])

            def emu_balance(t, X_flat, known_emus, M_inv, A, B, org_shape):
                X= X_flat.reshape(org_shape)
                Y= np.array([known_emu.inst_fun(t) for known_emu in known_emus])
                dxdt= M_inv @ (A @ X - B @ Y)
                return dxdt.flatten()

            sol = solve_ivp(emu_balance, time_span, X0.flatten(), method='BDF',
                            args=(known_emus, M_inv, A, B, X0.shape), dense_output=True, rtol=1e-12, atol=1e-12)


            for emu in model.emu_net.V:
                if emu in unknown_emus:
                    index = unknown_emus.index(emu)
                    mdv_size= subnet_size + 1
                    emu.inst_fun= lambda t, idx= index, size= mdv_size, solution= sol: solution.sol(t)[idx * size: idx * size + size]
                    emu.solved = True

                    if emu in model.target_emus:
                        target_emus_MIDs[str(emu)]= emu



        subnet_size += 1

    if presolve:
        model.req_c= req_c

    return target_emus_MIDs