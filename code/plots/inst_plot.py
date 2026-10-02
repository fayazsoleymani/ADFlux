import numpy as np
import matplotlib.pyplot as plt
import os

def plot_inst_emu(emu, time_span):
    time_points= np.linspace(time_span[0], time_span[1], 1000)
    mdvs= np.array([emu.inst_fun(t) for t in time_points])
    mdv_size= emu.size

    plt.figure(figsize=[5,5])
    for i in range(mdv_size+1):
        plt.plot(time_points, mdvs[:, i], label=str(f"M+{i}"))

        t_ss = time_points[-1]
        mid_ss = mdvs[-1, i]

        # Add text near the endpoint
        plt.text(
            t_ss + 0.01 * (time_span[1] - time_span[0]),  # slight offset
            mid_ss,
            f"{mid_ss:.4f}",
            va='center',
            fontsize=6,
        )

    plt.title(f"EMU {str(emu)}")
    plt.xlabel('Time')
    plt.ylabel('MID')
    plt.legend()
    plt.tight_layout()
    plt.show()





def plot_save_inst_estimated_measured(target_met, model, target_EMUs_MIDs, output_dir):

    time_points = np.linspace(model.time_span[0], model.time_span[1], 1000)


    mdvs= None
    comp_coefs= 0
    for emu_str in model.target_emus_dict[target_met]:

        met= emu_str.split('_')[0]
        met_id= met.split('.')[0]
        comp= met.split('.')[1] if '.' in met else None

        if bool(model.comp_alphas) and comp in model.comp_alphas:
            comp_coef= model.comp_alphas[comp]
        else:
            comp_coef = 1
        comp_coefs+= comp_coef

        emu= target_EMUs_MIDs[emu_str]
        mdv= np.array([emu.inst_fun(t) for t in time_points])

        mdv *= comp_coef
        if mdvs is None:
            mdvs = mdv
        else:
            mdvs += mdv

    mdvs/= comp_coefs
    mdv_size = emu.size


    time_points_measured = np.asarray(model.inst_measured_MIDs[0])
    MID_points = np.asarray(model.inst_measured_MIDs[1][target_met][0])
    MID_points_std = np.asarray(model.inst_measured_MIDs[1][target_met][1])

    plt.figure(figsize=[5, 5])

    for i in range(mdv_size + 1):
        # ODE curve
        line, = plt.plot(time_points, mdvs[:, i], label=f"M+{i}")


        plt.errorbar(
            time_points_measured,
            MID_points[i, :],
            yerr=MID_points_std[i, :],
            fmt="o",
            color=line.get_color(),
            markersize=4,
            capsize=3,
            alpha=0.8,
            zorder=3,
        )


    plt.title(f"EMU {target_met}")
    plt.xlabel("Time")
    plt.ylabel("MID")
    plt.ylim(0, 1)
    # plt.legend()
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{target_met}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()