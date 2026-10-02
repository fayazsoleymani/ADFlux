import numpy as np
# import torch
from core.model import MetabolicModel
from DL.lstm import EMUSurrogateLSTM
from plots.inst_plot import *
import os
from DL.tcn import EMUSurrogateTCN
from DL.dataset import EMUDataset, make_loaders
from DL.tet import train_model, predict_one
from DL.plots import plot_losses
from DL.i_o import save_checkpoint, load_checkpoint
import csv


def main():

    model = MetabolicModel('c_ohadii', '../data/models/c_ohadii', n_workers= 5)

    model.set_excluded_mets(['CO2_in', 'ACE_in', 'ALAprot', 'ARGprot', 'ASXprot', 'GLXprot',
                             'GLYprot', 'HISprot', 'ILEprot', 'LEUprot', 'LYSprot', 'METprot',
                             'PHEprot', 'PROprot', 'SERprot', 'THRprot', 'TYRprot', 'VALprot',
                             'CYSprot', 'TAG', 'G3P_TAG', 'PUnucOUT', 'PYnucOUT', 'Thymidine', 'WALL'])
    model.read_model_from_csv('c_ohadii.csv')
    model.generate_EMU_network(['S7P_1234567', 'UDPG_123456', 'ADPG_123456', 'GLU_12345', 'PG_12',
                                'SBP_1234567', 'TP_123', 'MAL_1234', 'ASP_1234','G6P_123456',
                                'F6P_123456', 'G1P_123456', 'AKG_12345', 'P5P_12345', 'SUC_1234',
                                'RUBP_12345', 'PGA_123',  'PEP_123'])
    model.set_tracer({'CO2_in': {'0': 0, '1': 1},
                      'ACE_in': {'00': 1}})
    model.set_time_span((0, 50))


    # inst estimation
    model.read_inst_measured_MIDs_from_csv('c_ohadii_50_measured_MIDs_comp.csv', name= '50',
                                           std_range= (0.01, 1))
    estimation_options= {
        'measured_fluxes': {'ACETATE-IN': (0, 1e-5)}, 'n_init': 10}
    model.set_est_params(**estimation_options)
    model.estimate_fluxes_inst()
    # model.calc_ci_sol(sol= 'x')

    print("FINISHED")



if __name__ == "__main__":
    main()