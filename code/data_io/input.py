import csv

import pandas as pd
from core.reaction import Rxn
import re
import numpy as np
import os



def read_model(model, file_path):

    with open(file_path, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader, None)
        for row in reader:
            if row[0][0].startswith('#'):
                continue
            temp_rxn, rotational_mols= parse_row_to_rxn(row)
            model.add_reaction(temp_rxn)
            if rotational_mols:
                model.add_rotational_mols(rotational_mols)
    


def read_fluxes_from_csv(model, file_path):
    with open(file_path, newline='') as file:
        reader= csv.reader(file, delimiter=',')
        next(reader, None)
        for row in reader:
            rxn_id= row[0]
            flux= float(row[1])
            rxn= model.rxns[[rxn.rxn_id for rxn in model.rxns].index(rxn_id)]
            rxn.flux= flux


def parse_row_to_rxn(row):

    rotational_mols= []
    rxn_id= row[0]
    lhs, rhs = row[1], row[2]
    rev= int(row[3])

    reactants, reactants_stoich_coefs, lhs_atom_map, rot_mols= parse_side(lhs, -1)
    if rot_mols:
        rotational_mols.extend(rot_mols)
    products, products_stoich_coefs, rhs_atom_map, rot_mols= parse_side(rhs, 1)
    if rot_mols:
        rotational_mols.extend(rot_mols)
    stoich_coefs= reactants_stoich_coefs + products_stoich_coefs

    atom_map= (lhs_atom_map, rhs_atom_map)

    return Rxn(rxn_id= rxn_id,
               reactants= reactants,
               products= products,
               stoich_coefs= stoich_coefs,
               atom_mapping= atom_map,
               rev= rev), rotational_mols


def parse_side(side, lhs_rhs_coef):
    comps= side.split('+')
    mets, stoich_coefs, atom_maps, rotational_mols= [], [], [], []
    for comp in comps:
        if comp:

            atoms = re.search(r"\(([^)]*)\)$", comp)
            if atoms:
                atoms= atoms.group(1)
                comp= comp.replace('(' + atoms + ')', '')
                if ';' in atoms:
                    new_atoms = atoms.split(';')[0]
                    rotational_mols.append((comp, len(new_atoms)))
                    atom_maps.append(new_atoms)
                else:
                    atom_maps.append(atoms)
            met, stoich_coef= parse_component_to_met_coef(comp, lhs_rhs_coef)

            if met not in mets:
                mets.append(met)
                stoich_coefs.append(stoich_coef)
            else:
                met_index= mets.index(met)
                stoich_coefs[met_index] += stoich_coef
    return mets, stoich_coefs, atom_maps, rotational_mols


def parse_component_to_met_coef(component, lhs_rhs_coef):
    coef = re.findall(r"^\d+(?:\.\d+)?", component)
    # coef = re.findall(r"^\d+", component)
    met = component.replace(coef[0], '', 1) if coef else component
    stoich_coef = float(coef[0]) if coef else 1.0
    # met = [met] * int(stoich_coef)
    return met, stoich_coef * lhs_rhs_coef



def read_inst_MIDs(model, file_path, name, std_range):
    min_std= std_range[0]
    max_std= std_range[1]
    
    time_points= set()
    with open(file_path, newline='') as file:
        reader= csv.reader(file, delimiter=',')
        next(reader, None)
        emu2mid= dict()
        emu2t2mid= dict()
        
        for row in reader:
            emu= row[0]
            t= float(row[1])
            time_points.add(t)
            mid= np.array([float(mid) for mid in row[2].split(';')])
            if mid.sum() == 0:
                continue
            else:
                mid /= mid.sum()

            if emu not in emu2t2mid:
                emu2t2mid[emu]= dict()
            if t not in emu2t2mid[emu]:
                emu2t2mid[emu][t]= []

            emu2t2mid[emu][t].append(mid)

        time_points= list(sorted(time_points))
        for emu, t2mids in emu2t2mid.items():
            if emu in model.target_emus_dict:    
                mid_means= []
                mid_stds= []
                for t, mids in t2mids.items():
                    mid_mean= np.mean(np.stack(mids), axis= 0)

                    if len(mids) > 1:
                        mid_std= np.std(np.stack(mids), axis= 0, ddof=1)
                    else:
                        mid_std= np.std(np.stack(mids), axis= 0)

                    mid_means.append(mid_mean.tolist())
                    mid_stds.append(mid_std.tolist())

                    

                
                mid_stds= np.array(mid_stds)
                mid_stds = np.broadcast_to(np.clip(mid_stds.max(axis=0, keepdims=True), min_std, max_std),mid_stds.shape).copy().T

                emu2mid[emu]= (np.array(mid_means).T, mid_stds)


        model.inst_measured_MIDs= (time_points, emu2mid, name)


def read_inst_MIDs_mean_std(model, file_path, name):
    
    time_points= set()
    with open(file_path, newline='') as file:
        reader= csv.reader(file, delimiter=',')
        next(reader, None)
        emu2mid= dict()
        emu2t2mid= dict()
        
        for row in reader:
            emu= row[0]
            t= float(row[1])
            time_points.add(t)
            mid_mean= np.array([float(mid) for mid in row[2].split(';')])
            mid_std= np.array([float(mid) for mid in row[3].split(';')])

            if emu not in emu2t2mid:
                emu2t2mid[emu]= dict()
            if t not in emu2t2mid[emu]:
                emu2t2mid[emu][t]= []

            emu2t2mid[emu][t]= (mid_mean, mid_std)

        time_points= list(sorted(time_points))
        for emu, t2mids in emu2t2mid.items():
            if emu in model.target_emus_dict:    
                mid_means= []
                mid_stds= []
                for t, mids in t2mids.items():
                    mid_mean= mids[0]
                    mid_std= mids[1]

                    mid_means.append(mid_mean.tolist())
                    mid_stds.append(mid_std.tolist())

                emu2mid[emu]= (np.array(mid_means).T, np.array(mid_stds).T)

        model.inst_measured_MIDs= (time_points, emu2mid, name)


def read_inst_pools(model, file_path, std_range):
    met2pools_stat= dict()
    met2pools= dict()
    all_pools= []
    min_std= std_range[0]
    max_std= std_range[1]


    with open(file_path, newline='') as file:
        reader= csv.reader(file, delimiter=',')
        next(reader, None)
        for row in reader:
            met= row[0]
            t= int(row[1])
            pools= [float(pool)*1e-3 for pool in row[2].split(";")]

            if met not in met2pools:
                met2pools[met]= [pools]
            else:
                met2pools[met].append(pools)
        

        for met, pools in met2pools.items():
            pool_reps= np.mean(np.array(pools), axis=0)
            pools_mean= pool_reps.mean()
            pools_std= np.clip(np.std(pool_reps, ddof= 1), min_std, max_std)
            met2pools_stat[met]= (pools_mean, pools_std)
            all_pools.append(pools_mean)

        model.c_lb= np.min(all_pools)
        model.c_ub= np.max(all_pools)

    model.measured_pools= met2pools_stat





















