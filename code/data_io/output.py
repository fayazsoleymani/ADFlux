import pandas as pd
import os
from plots.inst_plot import plot_save_inst_estimated_measured
import pickle

def output_estimation(model, sol, target_EMUs_MIDs, sol_number):


    output_dir = os.path.join(model.model_dir,
                              '_'.join(['estimation_results', model.inst_measured_MIDs[2]]),
                              str(sol_number))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_v = pd.DataFrame({
        'Rxns': [str(rxn) for rxn in model.rxns],
        'Fluxes': sol['v_sol'],
    })
    df_v.to_csv(os.path.join(output_dir, 'estimated_fluxes.csv'), index=False)

    df_c= pd.DataFrame({
        'Mets': sol['met_pools_sol'][0],
        'PoolSize': sol['met_pools_sol'][1],
    })
    df_c.to_csv(os.path.join(output_dir, 'estimated_pool_sizes.csv'), index=False)

    if 'comp_alphas_sol' in model.estimation_solution:
        df_alpha= pd.DataFrame({
            'Mets': sol['comp_alphas_sol'][0],
            'alpha': sol['comp_alphas_sol'][1],
        })
        df_alpha.to_csv(os.path.join(output_dir, 'estimated_comp_alphas.csv'), index=False)


    with open(os.path.join(output_dir, 'obj_val.txt'), 'w') as f:
        f.write(str(sol['obj']) + '\n')


    with open(os.path.join(output_dir, 'sol.pkl'), 'wb') as f:
        pickle.dump(sol, f)
        

    for target_met in model.target_emus_dict:
        plot_save_inst_estimated_measured(target_met, model, target_EMUs_MIDs, output_dir)


def output_estimation_ci(model, output_dir):
    df_v = pd.DataFrame({
        'Rxns': [rxn for rxn in model.irrev_rxns],
        'Fluxes': model.estimation_solution['vnet_sol'],
        'LB': [ci[0] for ci in model.estimation_solution['cis_vnet']],
        'UB': [ci[1] for ci in model.estimation_solution['cis_vnet']]})
    df_v.to_csv(os.path.join(output_dir, 'estimated_fluxes_ci.csv'), index=False)


    df_c= pd.DataFrame({
        'Mets': model.estimation_solution['met_pools_sol'][0],
        'PoolSize': model.estimation_solution['met_pools_sol'][1],
        'LB': [ci[0] for ci in model.estimation_solution['cis_c']],
        'UB': [ci[1] for ci in model.estimation_solution['cis_c']]})
    df_c.to_csv(os.path.join(output_dir, 'estimated_pool_sizes_ci.csv'), index=False)

    df_alpha= pd.DataFrame({
        'Mets': model.estimation_solution['comp_alphas_sol'][0],
        'alphas': model.estimation_solution['comp_alphas_sol'][1],
        'LB': [ci[0] for ci in model.estimation_solution['cis_alpha']],
        'UB': [ci[1] for ci in model.estimation_solution['cis_alpha']]})
    df_alpha.to_csv(os.path.join(output_dir, 'estimated_comp_alphas_ci.csv'), index=False)




