from copy import deepcopy
import numpy as np
from core.emu import generate_EMU_network, EMU
from core.simulation import simulate_MIDs
from data_io.input import read_model, read_fluxes_from_csv, read_inst_MIDs, read_inst_pools, read_inst_MIDs_mean_std
from plots.inst_plot import plot_inst_emu
from scipy.linalg import null_space
from core.reaction import Rxn
from core.optim import ss_estimate, inst_estimate, calc_ci
from scipy.optimize import linprog
import copy
from cobra import Model, Reaction, Metabolite
from cobra.sampling import OptGPSampler
import pandas as pd
import os
import casadi as ca
from data_io.output import output_estimation, output_estimation_ci
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import csv


def run_inst_estimate(args):
    model, optim_kwargs= args
    return inst_estimate(model, **optim_kwargs)



class MetabolicModel():

    def __init__(self, name, model_dir, n_workers= 5):
        self.name= name
        self.model_dir = model_dir
        self.mets = []
        self.rxns = []
        self.lb= np.array([])
        self.ub= np.array([])
        self.S= np.zeros((0, 0))
        self.max_flux= 100
        self.rotational_mols= dict()
        self.tracer= dict()
        self.exchange_rxns= []
        self.uptake_mets= []
        self.export_rxns= dict()
        self.count= 0
        self.biomass=[]
        self.def_met_const= 0.1
        self.req_c= None
        self.rev_rxns= []
        self.def_mid_sd= 0.1
        self.def_v_std= 0.01,
        self.time_span= [0, 1]
        self.comp_alphas= {}
        self.c_lb= 1e-3
        self.c_ub= 1e3
        self.irrev_rxns= []
        self.n_workers= n_workers

    def read_model_from_csv(self, path):
        read_model(self, os.path.join(self.model_dir, path))


    def read_fluxes_from_csv(self, path):
        read_fluxes_from_csv(self, os.path.join(self.model_dir, path))

    def read_inst_measured_MIDs_from_csv(self, path, name, std_range= (-np.inf, np.inf)):
        read_inst_MIDs(self, os.path.join(self.model_dir, path), name, std_range)

    def read_inst_measured_MIDs_from_csv_mean_std(self, path, name):
        read_inst_MIDs_mean_std(self, os.path.join(self.model_dir, path), name)

    def read_inst_measured_pools_from_csv(self, path, std_range=(-np.inf, np.inf)):
        read_inst_pools(self, os.path.join(self.model_dir, path), std_range)


    def add_reaction(self, rxn):
        # adding a Rxn object to the model

        self.rxns.append(rxn)

        if rxn.rxn_id == 'biomass':
            self.biomass.append(rxn)

        self.S= np.hstack((self.S, np.zeros((self.nMets, 1))))

        for index, metabolite in enumerate(rxn.reactants + rxn.products):
            # check if the metabolite exist in the model
            if metabolite not in self.mets and metabolite not in self.excluded_mets:
                self.mets.append(metabolite)
                self.S = np.vstack((self.S, np.zeros((1, self.nRxns))))

            elif metabolite in self.excluded_mets:
                self.exchange_rxns.append(rxn)
                if metabolite in rxn.reactants:
                    self.uptake_mets.append(rxn.reactants[0])
                elif metabolite in rxn.products:
                    self.export_rxns[metabolite]= [rxn]
                continue

            # modifying the stoichiometric matrix
            metabolite_index= self.mets.index(metabolite)
            self.S[metabolite_index, -1] += rxn.stoich_coefs[index]

        # adding lower and upper bounds
        lb= 0
        self.lb= np.append(self.lb, np.array([lb]))
        self.ub= np.append(self.ub, np.array([self.max_flux]), axis=0)

        # converting the reaction to irreversible if it's reversible by adding the reverse rxn to the model
        if rxn.rev == 1:

            self.rev_rxns.append(rxn)

            reactants_coefs = [-1 * coef for coef in rxn.stoich_coefs[:len(rxn.reactants)]]
            products_coefs = [-1 * coef for coef in rxn.stoich_coefs[len(rxn.reactants):]]
            new_rxn = Rxn(rxn_id=rxn.rxn_id + '_b',
                          reactants=rxn.products,
                          products=rxn.reactants,
                          stoich_coefs= products_coefs + reactants_coefs,
                          atom_mapping= (rxn.atom_mapping[1], rxn.atom_mapping[0]) if rxn.atom_mapping else None)

            self.add_reaction(new_rxn)

    def add_exchange_rxn(self, met):
        # self, rxn_id, reactants, products, stoich_coefs, atom_mapping, rev = 0, flux = None)
        # met_index= self.mets.index(met)
        self.add_reaction(Rxn('Ex_'+ met, [met], [], [-1], None, 1))

    def add_all_exchange_rxns(self):
        for index, met in enumerate(self.mets):
            if all(self.S[index] >= 0) or all(self.S[index] <= 0):
                self.add_exchange_rxn(met)


    def add_rotational_mols(self, rotational_mols):
        for mol in rotational_mols:
            if mol[0] not in self.rotational_mols:
                self.rotational_mols[mol[0]]= mol[1]


    def generate_EMU_network(self, target_emus):
        # constructing the EMU network for the target EMU
        emu2emucomp= {target_emu: [] for target_emu in target_emus}
        target_emus_model= []
        for target_emu in target_emus:

            met_target, pos= "_".join(target_emu.split('_')[0:-1]), target_emu.split("_")[-1]
            for met_model in self.mets:
                if "." in met_model:
                    met, comp= met_model.split(".")[0], met_model.split(".")[1]
                else:
                    met, comp= met_model, ""

                if met_target == met:
                    target_emu_model= met_model+'_'+pos
                    emu2emucomp[target_emu].append(target_emu_model)
                    target_emus_model.append(target_emu_model)

        self.target_emus_dict= emu2emucomp
        self.target_emus= [EMU(target_emu) for target_emu in target_emus_model]
        self.single_influx= [met for met_index, met in enumerate(self.mets) if len(np.where(self.S[met_index] > 0)[0]) == 1]# if met not in self.uptake_mets]
        self.emu_net= generate_EMU_network(self)


    def simulate_MIDs(self, **kwargs):

        tracer_input= kwargs.get('tracer', None)
        self.set_tracer(tracer_input, m_fun= 'sim')
        fluxes= kwargs.get('fluxes', None)
        if fluxes is not None:
            for i, rxn in enumerate(self.rxns):
                rxn.flux = fluxes[i]

        verbose= kwargs.get('verbose', False)
        simulation_type= kwargs.get('simulation_type', None)

        if simulation_type == 'ss':
            iso_staytionary= True
        elif simulation_type == 'inst':
            iso_staytionary= False
            self.met_pools = kwargs.get('met_pools', None)
            if 'time_span' in kwargs:
                self.time_span = kwargs['time_span']
            if 'comp_alphas' in kwargs:
                self.comp_alphas= kwargs['comp_alphas']
            elif not bool(self.comp_alphas):
                self.comp_alphas= {}
        else:
            return 0

        presolve = kwargs.get('presolve', False)
        target_EMUs_MID= simulate_MIDs(self, iso_staytionary, verbose, presolve)


        if simulation_type == 'inst' and verbose:
            for emu in target_EMUs_MID.values():
                plot_inst_emu(emu, self.time_span)

        return target_EMUs_MID



    def set_est_params(self, **kwargs):

        rxn2flux= dict()
        measured_fluxes= kwargs['measured_fluxes']
        for rxn, flux in measured_fluxes.items():
            rxn_index= [rxn.rxn_id for rxn in self.rxns].index(rxn)
            rxn2flux[rxn_index]= flux
            # self.lb[rxn_index], self.ub[rxn_index]= flux, flux
        self.measured_rxn2flux= rxn2flux

        self.comp_alphas= kwargs.get('comp_alphas', {})
        self.n_init= int(kwargs.get('n_init', 1))

        self.construct_vnet()



    def estimate_fluxes_inst(self, **kwargs):

        tracer_input= kwargs.get('tracer', None)
        self.set_tracer(tracer_input, m_fun= 'est')

        print("Initial estimation ...")

        optim_kwargs= {'optim_type': 'initial'}


        if self.n_init == 1:
            optim_kwargs.update({'seed': 2})
            self.estimation_solution = inst_estimate(self, **optim_kwargs)
            top_sols= [self.estimation_solution]

        else:

            initial_seed= 0
            seeds = range(initial_seed, self.n_init+initial_seed)
            

            kwargs_list= [{**optim_kwargs, 'seed': s} for s in seeds]
            args = [(self, kw) for kw in kwargs_list]
            
            results = []
            total = len(args)
            completed = 0


            for arg in kwargs_list:
                
                result= inst_estimate(self, **arg)
                results.append(result)
            
                completed += 1
                print("Progress: {}/{}, Obj: {:.2f}".format(completed, total, result['obj']))

                self.set_tracer(tracer_input, m_fun= 'est')

            top_sols = sorted(results, key=lambda r: float(r["obj"]))[:10]
            n_runs= sum(1 for r in results if r['obj'] < 1e4)
        
            self.estimation_solution = min(top_sols, key=lambda r: float(r["obj"]))


        print('Estimated v: ', [round(flux, 2).item() for flux in self.estimation_solution['v_sol']])
        print('Estimated v_net: ', [round(flux, 2).item() for flux in self.estimation_solution['vnet_sol']])
        print('Obj:\t{:.6f}'.format(self.estimation_solution['obj']))
        if self.n_init > 1:
            print("nRuns: {}".format(n_runs))

            # if estimation_type == 'inst':
        print('Estimated pool sizes:', end='\t')
        for met, c in zip(self.estimation_solution['met_pools_sol'][0],
                            self.estimation_solution['met_pools_sol'][1]):
            print('{}:{:.3f}  '.format(met, c), end='\t')

        if 'comp_alphas_sol' in self.estimation_solution:
            print('\nEstimated comp alphas:', end='\t')
            for comp, alpha in zip(self.estimation_solution['comp_alphas_sol'][0],
                                self.estimation_solution['comp_alphas_sol'][1]):
                print('{}:{:.3f}  '.format(comp, alpha), end='\t')
            
        print('\n')
        print("_" * 50)

        for sol_number, top_sol in enumerate(top_sols):

            inst_simulation_options= {
                'fluxes': top_sol['v_sol'],
                'simulation_type': 'inst',
                'met_pools': {k: v for k, v in zip(top_sol['met_pools_sol'][0], top_sol['met_pools_sol'][1])},
                'verbose': False,
            }
            if 'comp_alphas_sol' in self.estimation_solution:
                inst_simulation_options['comp_alphas']= {k: v for k, v in zip(top_sol['comp_alphas_sol'][0],
                                                                              top_sol['comp_alphas_sol'][1])}
            

            target_EMUs_MIDs= self.simulate_MIDs(**inst_simulation_options)
            output_estimation(self, top_sol, target_EMUs_MIDs, sol_number)


    def calc_ci_sol(self, sol):
        results_dir= os.path.join(self.model_dir, "_".join(['estimation_results', sol]))
        with open(os.path.join(results_dir, "sol.pkl"), 'rb') as f:
            self.estimation_solution= pickle.load(f)
        print("Calculating CIs ...")
        calc_ci(self)
        output_estimation_ci(self, results_dir)


    def reopt(self, sol):
        results_dir= os.path.join(self.model_dir, "_".join(['estimation_results', sol]))
        with open(os.path.join(results_dir, "sol.pkl"), 'rb') as f:
            org_sol= pickle.load(f)
        print('Reoptimizing ...')

        reoptim_kwargs= {'optim_type': 'reopt'}
        reoptim_kwargs.update({'prev_sol': org_sol})
        self.estimation_solution = inst_estimate(self, **reoptim_kwargs)

    def reopt_other(self, sol_file):
        v= []
        c= {}
        with open(os.path.join(self.model_dir, sol_file)) as file:
            reader= csv.reader(file, delimiter= ',')
            for row in reader:
                type= row[0]
                name= row[1]
                value= float(row[2])
                if type== 'v_net':
                    v.append(value)
                elif type== 'v_xch':
                    v_net= v[-1]
                    v_xch= value

                    v_f= v_xch-min(-v_net, 0)
                    v_b= v_xch-min(v_net, 0)

                    v[-1]= v_f
                    v.append(v_b)

                elif type== 'c':
                    c[name]= value
        v= np.array(v)
        reoptim_kwargs= {'optim_type': 'reopt_other'}
        reoptim_kwargs.update({'prev_sol': (v, c)})
        self.estimation_solution = inst_estimate(self, **reoptim_kwargs)




    def set_tracer(self, tracer_input=None, m_fun= 'main'):
        if tracer_input:
            self.tracer= tracer_input
        for emu in self.emu_net.V:
            if emu.met in self.tracer:
                emu2MID(emu, self.tracer[emu.met])
                emu.solved = True
                if m_fun == 'sim':
                    emu.inst_fun= lambda t, e=emu: e.MID

            else:
                emu.MID= np.zeros_like(emu.MID)
                emu.MID[0]= 1
                emu.solved = False
                if m_fun == 'sim':
                    emu.inst_fun= None


    def fba(self, **kwargs):

        A_eq, b_eq= self.S, np.zeros(self.nMets)
        lb= copy.deepcopy(self.lb)
        ub= copy.deepcopy(self.ub)
        for rxn, flux in kwargs['xch_fluxes'].items():
            rxn_index= [rxn.rxn_id for rxn in self.rxns].index(rxn)
            lb[rxn_index], ub[rxn_index]= flux, flux
        bounds= list(zip(lb, ub))
        c= np.zeros(self.nRxns)
        obj_index= [rxn.rxn_id for rxn in self.rxns].index(kwargs['obj_fun'])
        c[obj_index]= -1
        sol= linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        print("obj: {:.5f}".format(-sol.fun))
        return sol.x

    def convert2cobra(self, obj):
        cobra_model= Model(self.name)
        self.obj= obj
        for j, rxn in enumerate(self.rxns):
            cobra_rxn= Reaction(rxn.rxn_id)
            cobra_rxn.name= rxn.rxn_id
            cobra_rxn.lower_bound= self.lb[j]
            cobra_rxn.upper_bound= self.ub[j]


            stoich= {}
            for met in  rxn.reactants + rxn.products:
                if met not in self.excluded_mets:
                    met_index= self.mets.index(met)
                    stoich[Metabolite(met)]= self.S[met_index, j]

            cobra_rxn.add_metabolites(stoich)
            cobra_model.add_reactions([cobra_rxn])

        cobra_model.objective= obj

        self.cobra_model= cobra_model



    def obtain_req_c(self):
        inst_simulation_options = {
            'simulation_type': 'inst',
            'time_span': self.time_span,
            'presolve': True}
        self.simulate_MIDs(**inst_simulation_options)


    def construct_vnet(self):
        v_size= self.nRxns
        vnet_size = v_size - len(self.rev_rxns)
        self.T = np.eye(vnet_size)
        self.vnet_lb = np.zeros(vnet_size)
        self.vnet_ub = self.max_flux * np.ones(vnet_size)
        self.vnet_size= vnet_size
        for count, rxn in enumerate(self.rev_rxns):
            rxn_index = self.rxns.index(rxn)
            back_rxn_index = rxn_index + 1
            temp = np.zeros(vnet_size)
            temp[rxn_index - count] = -1
            self.T = np.insert(self.T, rxn_index + 1, temp, axis=1)
            self.vnet_lb[rxn_index - count] = -self.max_flux
        self.irrev_rxns= []
        for rxn in self.rxns:
            if '_b' not in str(rxn):
                self.irrev_rxns.append(str(rxn))


    def set_time_span(self, time_span):
        self.time_span = time_span


    def set_excluded_mets(self, mets):
        self.excluded_mets = mets




    @property
    def N(self):
        return null_space(self.S)

    @property
    def nMets(self):
        return self.S.shape[0]

    @property
    def nRxns(self):
        return self.S.shape[1]



    def __str__(self):
        temp_str= ''
        temp_str += self.name + '\n'
        temp_str += "#Mets\t:" + str(self.nMets) + '\n'
        temp_str += "#Rxns\t:" + str(self.nRxns) + '\n'
        return temp_str



def emu2MID(emu, binary2mid):

    specific2mid= dict()
    for binary_val, percentage in binary2mid.items():
        specific_part= "".join([binary_val[pos] for pos in emu.pos])
        if specific_part not in specific2mid:
            specific2mid[specific_part] = percentage
        else:
            specific2mid[specific_part]+=percentage

    mid2percentage= {k:0 for k in range(emu.size+1)}
    for binary_val, percentage in specific2mid.items():
        mid2percentage[binary_val.count('1')] += percentage

    emu.MID= np.array([mid2percentage[i] for i in range(emu.size+1)])

