import os

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd


def main():
    # kitchen has ent0, bsz 64
    # antmaze has ent 0.1, bsz 4096
    # spirl/ssp is best of tuned bsz, filtered or unfiltered
    # sfp is much too expensive to tune...
    create_sfp_csvs()
    matplotlib.rcParams.update({'font.size': 20})

    savepath = 'plots/antmaze_umaze.pdf'
    paths = ['csvs/sas_antmaze_umaze.csv', 'csvs/ssp_antmaze_umaze.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SSP']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/antmaze_umaze_obs.pdf'
    paths = ['csvs/sas_antmaze_umaze.csv', 'csvs/spirl_antmaze_umaze_filtered_bsz4096.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SPiRL']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/antmaze_medium.pdf'
    paths = ['csvs/sas_antmaze_medium.csv', 'csvs/ssp_antmaze_medium.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SSP']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/antmaze_medium_obs.pdf'
    paths = ['csvs/sas_antmaze_medium.csv', 'csvs/spirl_antmaze_medium_unfiltered.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SPiRL']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/antmaze_large.pdf'
    paths = ['csvs/sas_antmaze_large.csv', 'csvs/ssp_antmaze_large.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SSP']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/antmaze_large_obs.pdf'
    paths = ['csvs/sas_antmaze_large.csv', 'csvs/spirl_antmaze_large_filtered_bsz4096.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SPiRL']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/kitchen.pdf'
    paths = ['csvs/sas_kitchen.csv', 'csvs/ssp_kitchen.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SSP']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=4)

    savepath = 'plots/kitchen_obs.pdf'
    paths = ['csvs/sas_kitchen.csv', 'csvs/spirl_kitchen.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SPiRL']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=4)

    savepath = 'plots/coinrun.pdf'
    paths = ['csvs/coinrun_bsz4096_ent0.5.csv', 'csvs/ssp_coinrun.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SSP']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=20)

    savepath = 'plots/coinrun_obs.pdf'
    paths = ['csvs/coinrun_bsz4096_ent0.5.csv', 'csvs/spirl_coinrun.csv']
    run_indices = [[0, 1, 2, 3, 4]] * len(paths)
    labels = ['SaS', 'SPiRL']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=20)

    # ablations
    savepath = 'plots/antmaze_vocabsize.pdf'
    # paths = ['csvs/antmaze_medium_bsz4096_ent0.1.csv', *['csvs/antmaze_medium_vocabsize.csv' for _ in range(3)]]
    # run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    paths = ['csvs/sas_antmaze_medium.csv', *['csvs/sas_antmaze_vocabsize.csv' for _ in range(3)]]
    run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    labels = [16, 8, 32, 64]
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/antmaze_k.pdf'
    # paths = ['csvs/antmaze_medium_bsz4096_ent0.1.csv', *['csvs/antmaze_medium_k.csv' for _ in range(3)]]
    # run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    paths = ['csvs/sas_antmaze_medium.csv', *['csvs/sas_antmaze_k.csv' for _ in range(3)]]
    run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    labels = [16, 8, 32, 64]
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/antmaze_length.pdf'
    # paths = ['csvs/antmaze_medium_bsz4096_ent0.1.csv', *['csvs/antmaze_medium_length.csv' for _ in range(3)]]
    # run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    paths = ['csvs/sas_antmaze_medium.csv', *['csvs/sas_antmaze_length.csv' for _ in range(3)]]
    run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    labels = [10, 5, 15, 20]
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/antmaze_tokalg.pdf'
    # paths = ['csvs/antmaze_medium_bsz4096_ent0.1.csv', *['csvs/antmaze_medium_tokalg.csv' for _ in range(2)]]
    # run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(1, 3)]]
    paths = ['csvs/sas_antmaze_medium.csv', *['csvs/sas_antmaze_tokalg.csv' for _ in range(2)]]
    run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(1, 3)]]
    labels = ['BPE', 'WordPiece', 'Unigram']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    savepath = 'plots/kitchen_vocabsize.pdf'
    # paths = ['csvs/kitchen_bsz64_ent0.csv', *['csvs/kitchen_vocabsize.csv' for _ in range(3)]]
    # run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    paths = ['csvs/sas_kitchen.csv', *['csvs/sas_kitchen_vocabsize.csv' for _ in range(3)]]
    run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    labels = [16, 8, 32, 64]
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=4)

    savepath = 'plots/kitchen_k.pdf'
    # paths = ['csvs/kitchen_bsz64_ent0.csv', *['csvs/kitchen_k.csv' for _ in range(3)]]
    # run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    paths = ['csvs/sas_kitchen.csv', *['csvs/sas_kitchen_k.csv' for _ in range(3)]]
    run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    labels = [18, 9, 36, 72]
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=4)

    savepath = 'plots/kitchen_length.pdf'
    # paths = ['csvs/kitchen_bsz64_ent0.csv', *['csvs/kitchen_length.csv' for _ in range(3)]]
    # run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    paths = ['csvs/sas_kitchen.csv', *['csvs/sas_kitchen_length.csv' for _ in range(3)]]
    run_indices = [list(range(5)), *[list(range(i, 15, 3)) for i in range(3)]]
    labels = [10, 5, 15, 20]
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=4)

    savepath = 'plots/kitchen_tokalg.pdf'
    # paths = ['csvs/kitchen_bsz64_ent0.csv', *['csvs/kitchen_tokalg.csv' for _ in range(3)]]
    # run_indices = [list(range(5)), *[list(range(i, 10, 2)) for i in range(2)]]
    paths = ['csvs/sas_kitchen.csv', *['csvs/sas_kitchen_tokalg.csv' for _ in range(3)]]
    run_indices = [list(range(5)), *[list(range(i, 10, 2)) for i in range(2)]]
    labels = ['BPE', 'WordPiece', 'Unigram']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=4)

    # transfer
    savepath = 'plots/antmaze_transfer.pdf'
    paths = ['csvs/antmaze_transfer_bsz4096_ent0.1.csv']
    run_indices = [list(range(i, 20, 4)) for i in range(4)]
    labels = ['1%', '10%', '25%', '100%']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1)

    # clusters and locomotion
    savepath = 'plots/hopper_k.pdf'
    paths = ['csvs/hopper_k.csv']
    run_indices = [list(range(10, 15)), *[list(range(i, 15, 3)) for i in range(3)]]
    labels = [12, 6, 24, 48]
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=3500)

    # data quality
    savepath = 'plots/hopper_quality.pdf'
    paths = ['csvs/hopper_quality.csv']
    run_indices = [list(range(i*5, (i+1)*5)) for i in range(3)]
    labels = ['Random', 'Medium', 'Expert']
    plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=3500)


def create_sfp_csvs():
    paths = []
    savepaths = ['csvs/sfp_coinrun.npz', 'csvs/sfp_kitchen.npz', 'csvs/sfp_antmaze_medium.npz', 'csvs/sfp_antmaze_large.npz', 'csvs/sfp_antmaze_umaze.npz'] 
    for path, savepath in zip(paths, savepaths):
        arr = merge_sfp_csvs(path)
        np.savez(savepath, arr)


def merge_sfp_csvs(path):
    # SFP csv reading
    returns = []
    for i in range(5):
        cpath = path.replace('0', str(i))
        df = pd.read_csv(cpath, sep='\s+')
        returns.append(df['AverageTestEpRet'].tolist())
    min_length = min(len(r) for r in returns)
    if 'kitchen' in path:
        max_steps = 2_000_000
    elif 'coinrun' in path:
        max_steps = 1_500_000
    else:
        max_steps = 10_000_000
    steps = max_steps / min_length * np.arange(0, min_length)
    returns = [r[:min_length] for r in returns]
    mean = np.mean(np.array(returns), axis=0)
    std = np.std(np.array(returns), axis=0)
    arr = np.stack([steps, mean, std], axis=0)
    return arr


def plot_figure(savepath, paths, run_indices, labels, ymin=0, ymax=1):
    data_dicts = [csv_to_data_dict(path) for path in paths]
    if len(paths) == 1:
        data_arrs = [group_runs(dd, ri) for dd in data_dicts for ri in run_indices]  # for single-csv multi-group plots
    else:
        data_arrs = [group_runs(dd, ri) for dd, ri in zip(data_dicts, run_indices)]  # for multi-csv plots

    # incorporate opal with dotted line
    if 'obs' not in savepath and 'kitchen' in savepath and 'SSP' in labels:
        ssp_p_arr = data_arrs[-1].copy()
        ssp_p_arr = np.concatenate([np.zeros_like(ssp_p_arr[:, 0:1]), ssp_p_arr], axis=1)
        ssp_p_arr[1, :] = 0.8
        ssp_p_arr[2, :] = 0.2
        data_arrs.append(ssp_p_arr)
        labels.append('SSP-p')
    if 'obs' in savepath and 'antmaze' in savepath:
        if 'medium' in savepath:
            opal_arr = data_arrs[-1].copy()
            opal_arr[1, :] = 0.816
            opal_arr[2, :] = 0.037
            data_arrs.append(opal_arr)
            labels.append('OPAL')
        elif 'large' in savepath:
            opal_arr = data_arrs[-1].copy()
            opal_arr[1, :] = 0.0
            opal_arr[2, :] = 0.0
            data_arrs.append(opal_arr)
            labels.append('OPAL')
    if 'obs' not in savepath and 'SSP' in labels:
        if 'umaze' in paths[0]:
            sfp_path = 'csvs/sfp_antmaze_umaze.npz'
        elif 'medium' in paths[0]:
            sfp_path = 'csvs/sfp_antmaze_medium.npz'
        elif 'large' in paths[0]:
            sfp_path = 'csvs/sfp_antmaze_large.npz'
        elif 'kitchen' in paths[0]:
            sfp_path = 'csvs/sfp_kitchen.npz'
        elif 'coinrun' in paths[0]:
            sfp_path = 'csvs/sfp_coinrun.npz'
        sfp_arr = np.load(sfp_path)['arr_0']
        data_arrs.append(sfp_arr)
        labels.append('SFP')
        if 'coinrun' not in savepath:
            sac_arr = sfp_arr.copy()
            sac_arr[1, :] = 0.0
            sac_arr[2, :] = 0.0
            data_arrs.append(sac_arr)
            labels.append('SAC')
        sacd_arr = sfp_arr.copy()
        sacd_arr[1, :] = 0.0
        sacd_arr[2, :] = 0.0
        data_arrs.append(sacd_arr)
        labels.append('SAC-d')

    colors = matplotlib.colormaps['Set1'].colors
    # plot multiple arrs with corresponding labels
    fig, ax = plt.subplots()
    for label, arr, color in zip(labels, data_arrs, colors):
        # make plots more consistent looking across methods
        if (label == 'SSP' or label == 'SPiRL') and 'coinrun' in savepath: 
            arr[1, :] = exponential_smooth(arr[1, :], alpha=0.2)
            arr[2, :] = exponential_smooth(arr[2, :], alpha=0.2)
            arr = np.concatenate([np.zeros_like(arr[:, 0:1]), arr], axis=1)
        elif label != 'SAC' and label != 'SAC-d' and label != 'SFP' and label != 'OPAL' and label != 'SSP-p':
            # add zeros to the beginning when SaS or ablations
            arr = np.concatenate([np.zeros_like(arr[:, 0:1]), arr], axis=1)
        elif label == 'SFP':
            arr[1, :] = exponential_smooth(arr[1, :], alpha=0.1)
            arr[2, :] = exponential_smooth(arr[2, :], alpha=0.1)

        # limit steps to SaS saving
        # subsample arrays down to every 1_000_000 steps for antmaze, 100_000 for others
        if 'antmaze' in savepath:
            steps = list(range(0, 10_000_000, 1_000_000))
        elif 'kitchen' in savepath:
            steps = list(range(0, 2_000_000, 100_000))
        elif 'coinrun' in savepath:
            steps = list(range(0, 1_500_000, 100_000))
        elif 'hopper' in savepath:
            steps = list(range(0, 3_000_000, 100_000))
        ixs = get_subsampled_ixs(arr, steps)
        arr = arr[:, ixs]

        plot_mean_std(ax, arr, label, color, ymin=ymin, ymax=ymax)
    ax.legend()
    ax.set_ylabel('Return')
    ax.set_xlabel('Steps')
    fig.tight_layout(pad=0.25)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    plt.close(fig)


def get_subsampled_ixs(arr, steps):
    ixs = []
    for step2 in steps:
        ix = np.argmin(np.abs(arr[0] - step2))
        ixs.append(ix)
    return ixs


# plot of eval metrics
def plot_mean_std(ax, arr, label, color, ymin=0, ymax=1):
    step = arr[0]
    mean = arr[1]
    std = arr[2]
    if label == 'OPAL' or label == 'SSP-p':
        ax.plot(step, np.clip(mean, ymin, ymax), label=label, color=color, linestyle='dotted', linewidth=4)
    else:
        ax.plot(step, np.clip(mean, ymin, ymax), label=label, color=color, linewidth=4)
    ax.fill_between(step, np.clip(mean-std, ymin, ymax), np.clip(mean+std, ymin, ymax), color=color, alpha=0.1)


def group_runs(data_dict, run_indices):
    # takes list of run indices and groups those keys
    key_stump = list(data_dict.keys())[1].split(' ')[0].split('-')[0]
    data = []

    if len(data_dict.keys())-1 == len(run_indices):
        # no need to filter when exactly as many indices as keys
        for k in data_dict:
            if 'Step' not in k:
                data.append(data_dict[k])
    else:
        for k in data_dict:
            for ri in run_indices:
                if f'{key_stump}-{ri} ' in k or f'seed{ri}' in k:
                    data.append(data_dict[k])

    data = np.array(data)
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    data = np.stack([data_dict['Step'], mean, std])
    return data


def csv_to_data_dict(path):
    assert os.path.splitext(path)[1] == '.csv'
    df = pd.read_csv(path)
    for k in df:
        if 'MIN' in k or 'MAX' in k:
            df = df.drop(k, axis=1)

    # first partition 'Step' column, then make new dataframe with merges
    steps = df['Step']
    divisions = []
    current_division = []
    for ix, step in enumerate(steps):
        if ix == 0:
            # always add first step
            current_division.append(ix)
        else:
            # if first 3 digits agree with last step in current_division, then append, else append current_division
            if str(step)[:3] == str(last_step)[:3]:
                current_division.append(ix)
            else:
                divisions.append(current_division)
                current_division = [ix]
        last_step = step
    divisions.append(current_division)

    # merge rows with steps with the same leading 3 digits
    df[df.isna()] = 0.0  # set NaNs to 0, then merge rows based on max
    data_dict = {k: [] for k in df.keys()}
    for div in divisions:
        step = df.iloc[div[0]]['Step']
        row = df.iloc[div[0]:div[-1]+1].max(axis=0)
        data_dict['Step'].append(step)
        for k in data_dict:
            if k != 'Step':
                data_dict[k].append(row[k])
    data_dict = {k: np.array(v) for k, v in data_dict.items()}
    return data_dict


def exponential_smooth(a, alpha=0.3):
    smoothed = [a[0]]
    for i in range(1, len(a)):
        new = (1 - alpha) * smoothed[i-1] + alpha * a[i]
        smoothed.append(new)
    return smoothed


if __name__ == '__main__':
    main()
