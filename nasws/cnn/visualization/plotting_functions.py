# put the plotting methods in this folder.
# generate the paper required images.

import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .tool import *
#
# base_colors = {
#     'nasbench101': '#AA5939',
#     'nasbench201': '#29526D',
#     'darts_nds': '#277553',
# }


def _obtain_base_color():
    cmaps = plt.get_cmap('tab10', 10)
    names = ['nasbench101', 'nasbench201', 'darts_nds']
    return {n: cmaps(ind) for ind, n in enumerate(names)}

base_colors = _obtain_base_color()


def icml_plot_tags_and_skdt(
        exp_dir, str_filter, validate_key, target_fn, plot_path, search_spaces,
        space_keys=['nasbench101', 'nasbench201', 'darts_nds'],
        target_tags=['Train/lr', 'Valid/loss'], plot_tags=None,
        spaces_y_lims = {
            'nasbench101': [-0.1, 0.4],
            'darts_nds': [-0.2, 0.4],
            'nasbench201': [0.1, 0.8]
        },
        legend_axis=[0],
        ):
    """Plot the results , learning rate
    
    Parameters
    ----------
    exp_dir : [type]
        [description]
    str_filter : [type]
        [description]
    validate_key : [type]
        [description]
    target_fn : [type]
        [description]
    plot_path : [type]
        [description]
    search_spaces : [type]
        [description]
    space_keys : list, optional
        [description], by default ['nasbench101', 'nasbench201', 'darts_nds']
    target_tags : list, optional
        [description], by default ['Train/lr', 'Valid/loss']
    plot_tags : [type], optional
        [description], by default None
    
    Returns
    -------
    [type]
        [description]
    """

    accs, kdt, best_models, p_random, res_dicts = load_data_from_experiment_root_dir(
        exp_dir, str_filter, original_args=True, target_fn=target_fn)

    print(res_dicts.keys())
    for space in space_keys:

        lrs = list(sorted(list(res_dicts[space].keys())))
        print(lrs)
        cmaps = plt.get_cmap('tab10', len(lrs))

        plot_tags = plot_tags or target_tags

        fig, axes = plt.subplots(1, len(target_tags) + 1, figsize=(9, 3))


        for plot_ind, tag in enumerate(target_tags):
            ax = axes[plot_ind]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(tag)
            for ind, lr in enumerate(lrs):
                steps, data = plot_data_x_y(res_dicts[space][lr], [tag])
                ax.plot(steps[tag], data[tag], color=cmaps(ind), label=str(lr))
            if plot_ind in legend_axis:
                ax.legend()

        axes[1].set_xlabel('epochs')

        ax = axes[len(target_tags)]
        ax.set_ylabel('S-KdT')

        def _data_fn(x): return x[0][0], x[0][1]
        data_fn = _data_fn  # return data, error. if error does not exists, return 0

        def fn_get_searched_ids(x):
            arch_ids, arch_supernet_perfs = x[5], x[4]
            s_arch_ids, _ = sort_hash_perfs(arch_ids, arch_supernet_perfs)
            return s_arch_ids[-5:], 0

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plot the sparse kdt.
        for ind, lr in enumerate(lrs):
            steps, skdt, pvalue = plot_data_sparse_kdt(
                res_dicts[space][lr], search_spaces, data_fn)
            steps, top_k_arch, _ = plot_data_sparse_kdt(
                res_dicts[space][lr], search_spaces, fn_get_searched_ids)
            ax.errorbar(steps, skdt, np.array(pvalue)/2,
                        color=cmaps(ind), label=str(lr))
            # ax.legend()
            print(f'{lr}: kdt {skdt[-1]} +- {pvalue[-1]}')
            print(
                f"mean performances {np.mean(search_spaces[space].query_gt_perfs(top_k_arch[-1]))}")
        ax.set_ylim(spaces_y_lims[space])
        # plt.show()
        print('save fig to ', plot_path + space + '.pdf')
        plt.savefig(plot_path + space + '.pdf', bbox_inches='tight')




def icml_plot_loss_perf_kdt(exp_dir, str_filter, validate_key, target_fn, plot_path, search_spaces,
        space_keys=['nasbench101', 'nasbench201', 'darts_nds'],
        spaces_y_lims = {
            'nasbench101': [-0.1, 0.4],
            'darts_nds': [-0.2, 0.4],
            'nasbench201': [0.1, 0.8]
        },
        perf_y_lims = {
             'nasbench101': [0.915, 0.945],
                'darts_nds': [93.00, 95.00],
                'nasbench201': [87.5, 94.]
        },
        cmaps_index_input=None,
        legend_axis=[0],
        load_data_kwargs={}
        ):
    """Plot the results , learning rate
    
    Parameters
    ----------
    exp_dir : [type]
        [description]
    str_filter : [type]
        [description]
    validate_key : [type]
        [description]
    target_fn : [type]
        [description]
    plot_path : [type]
        [description]
    search_spaces : [type]
        [description]
    spaces_y_lims : [list]
    Returns
    -------
    [type]
        [description]
    """

    accs, kdt, best_models, p_random, res_dicts = load_data_from_experiment_root_dir(
        exp_dir, str_filter, original_args=True, target_fn=target_fn, **load_data_kwargs)

    print(res_dicts.keys())
    for space in space_keys:

        lrs = list(sorted(list(res_dicts[space].keys())))
        print(lrs)
        cmaps = plt.get_cmap('tab10', 10)
        # cmaps_index = [s* (10 // len(lrs) ) for s in range(len(lrs))]
        # cmaps_index = list(range(len(lrs)))
        cmaps_index = cmaps_index_input or list(range(len(lrs)))
        if 'nasbench101' in space:
            target_tags = ['Valid/loss']
        else:
            target_tags = ['Valid/top_1_acc']
        plot_tags = ['valid loss', 'perf']

        fig, axes = plt.subplots(1, 3, figsize=(9,3))

        for plot_ind, tag in enumerate(target_tags):
            ax = axes[plot_ind]
            print(f"plot tag {tag} with tags {lrs}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(plot_tags[plot_ind])
            for ind, lr in enumerate(lrs):
                steps, data = plot_data_x_y(res_dicts[space][lr], [tag])
                ax.plot(steps[tag], data[tag], color=cmaps(cmaps_index[ind]), label=str(lr))
                
            if plot_ind in legend_axis:
                ax.legend()
    
        # Final Arch Performance 
        ax = axes[1]

        for ind, lr in enumerate(lrs):
            def _process_model_archs(x):
                TOP_K = 10
                if space == 'nasbench201':
                    s_model_ids = x[5][-TOP_K:]
                else:
                    s_model_ids = list(sorted(x[5][-TOP_K:]))
                    s_model_ids = s_model_ids[-3:]
                perf = np.mean(search_spaces[space].query_gt_perfs(s_model_ids))
                std = np.std(search_spaces[space].query_gt_perfs(s_model_ids))
                return perf, std

            steps, perf, std = plot_data_sparse_kdt(res_dicts[space][lr], search_spaces, _process_model_archs, **load_data_kwargs)
            ax.plot(steps, perf, color=cmaps(cmaps_index[ind]), label=str(lr))
            # ax.errorbar(steps, perf, std, color=cmaps(cmaps_index[ind]), label=str(lr))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        axes[1].set_xlabel('epochs')
        ax.set_ylabel(plot_tags[1])
        ax.set_ylim(perf_y_lims[space])
        if 1 in legend_axis:
            ax.legend()
    
        ax = axes[2]
        ax.set_ylabel('S-KdT')

        def _data_fn(x): return x[0][0], x[0][1]
        data_fn = _data_fn  # return data, error. if error does not exists, return 0

        def fn_get_searched_ids(x):
            arch_ids, arch_supernet_perfs = x[5], x[4]
            s_arch_ids, _ = sort_hash_perfs(arch_ids, arch_supernet_perfs)
            return s_arch_ids[-5:], 0

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plot the sparse kdt.
        for ind, lr in enumerate(lrs):
            print(load_data_kwargs)
            steps, skdt, pvalue = plot_data_sparse_kdt(
                res_dicts[space][lr], search_spaces, data_fn, **load_data_kwargs)
            steps, top_k_arch, _ = plot_data_sparse_kdt(
                res_dicts[space][lr], search_spaces, fn_get_searched_ids, **load_data_kwargs)
            ax.plot(steps, skdt, 
                        color=cmaps(cmaps_index[ind]), label=str(lr))
            # ax.legend()
            print(f'{lr}: kdt {skdt[-1]} +- {pvalue[-1]}')
            print(
                f"mean performances {np.mean(search_spaces[space].query_gt_perfs(top_k_arch[-1]))}")
        ax.set_ylim(spaces_y_lims[space])
        if 2 in legend_axis:
            ax.legend()    
        print('save fig to ', plot_path + space + '.pdf')
        plt.savefig(plot_path + space + '.pdf', bbox_inches='tight')
    return plt


def plot_rank_in_bet_group(rank_change_data, save_path, show=True):
    from visualization.plot_rank_change import get_cmap_as_rgbs
    fig, ax = plt.subplots(figsize=(1.5, 3))

    # Hide the right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    y_dim, x_dim = len(rank_change_data), 2
    rgb = get_cmap_as_rgbs(nb_points=y_dim+30)
    rgb = rgb[10:y_dim+10]
    x_values = [0, 1]
    for i in range(y_dim):
        ax.plot(x_values, rank_change_data[i], linestyle='--', linewidth=1, color=rgb[i])
    ax.set_xticks(x_values)
    # ax.set_xticklabels([str(i) for i in x_values[:-1]] + ['ground-truth'])
    # ax.legend(bbox_to_anchor=(1.3, 0.5), loc='center')
    if show:
        plt.show()
    if save_path:
        print(save_path)
        plt.savefig(save_path, bbox_inches='tight')


def cvpr_process_and_plot_cifar10_results_over_three_search_spaces(
    root_folder, data_folders, folders_tags, search_spaces, plot_path, process_data_folder_fn=None,
    spaces_y_lims = {
    # 'nasbench101': [-0.1, 0.4],
    # 'darts_nds': [-0.2, 0.4],
    'nasbench201': [0.5, 0.9],
    'nasbench102': [0.5, 0.9]
    },):
    def process_data_folder(l):
        return os.path.join(root_folder, l.replace('/runs', ''), 'args.json')
    process_data_folder_fn = process_data_folder_fn or process_data_folder
    from matplotlib import pyplot as plt
    cmaps = plt.get_cmap('tab10', 10)

    for space in search_spaces.keys():
        plot_datas = []
        if space not in data_folders:
            continue

        for tag, all_list in zip(folders_tags, data_folders[space]):
            # do one plot at one time.
            # process the path to args.json
            all_list = [process_data_folder_fn(l) for l in all_list]
            # get the data
            plot_datas.append(load_data_from_experiment_root_dirs(
                all_list, target_fn=lambda x: tag, original_args=True)
                )
        fig, axes = plt.subplots(1, 1, figsize=(3,3))
        ax = axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('sparse Kendall Tau')
        ax.set_xlabel('epoch')
        if space in spaces_y_lims:
            ax.set_ylim(spaces_y_lims[space])
        def _data_fn(x): return x[0][0], x[0][1]
        data_fn = _data_fn  # return data, error. if error does not exists, return 0

        for ind, tag in enumerate(folders_tags):
            # wrong, this is only ploting 1 curve, you should instead do an average ploting ..
            # steps, skdt, pvalue, top_k_arch = [], [], [], []
            res_dicts = plot_datas[ind][-1]
            print(res_dicts.keys())
            steps, skdt, skdt_std = plot_data_sparse_kdt_avg(res_dicts[space][tag], search_spaces, data_fn)
            _, top_k_arch, _ = plot_data_sparse_kdt(res_dicts[space][tag], search_spaces, fn_get_searched_ids)

            final_step=-1
            if space == 'darts_nds':
                final_step = -1
                steps = steps[:final_step]
                skdt = skdt[:final_step]
                skdt_std = skdt_std[:final_step]
                # ax.plot(steps[:-4], skdt[:-4], color=cmaps(ind), label=str(tag))
            elif space == 'nasbench101':
                final_step = -1
                steps = steps[:final_step]
                skdt = skdt[:final_step]
                skdt_std = skdt_std[:final_step]
            else:    
                pass
            skdt = np.array(skdt)
            skdt_std = np.array(skdt_std)
            ax.plot(steps, skdt, color=cmaps(ind), label=str(tag))
            ax.fill_between(steps, skdt - skdt_std, skdt + skdt_std, color=cmaps(ind), alpha=0.3)
            ax.legend()
            # ax.legend()
            print(f'{tag}: kdt {skdt[final_step-1]} +- {skdt_std[final_step-1]}')
            print(f"mean performances {np.mean(search_spaces[space].query_gt_perfs(top_k_arch[final_step-1]))}") 
            print(f"std performances {np.std(search_spaces[space].query_gt_perfs(top_k_arch[final_step-1]))}") 
            print(f"best performances {np.max(search_spaces[space].query_gt_perfs(top_k_arch[final_step-1]))}") 
            print(f'best rank {MAXRANK[space] - np.max(top_k_arch[final_step-1])}')

        # simply get the 
        plt_file =  plot_path + space + '-skdt.pdf'
        print('save fig to ',plt_file)
        plt.savefig(plt_file, bbox_inches='tight')


def cvpr_plot_ablation_kdt_perf(data_folders, folders_tags, search_spaces, process_data_folder_fn=None):

    collect = {
        # key: 1, 2 ... as distance
        # data: [skdt, skdt-std, mean perf, std perf, best perf]
    }

    def process_data_folder(l):
        return l
    
    process_data_folder_fn = process_data_folder_fn or process_data_folder
    from matplotlib import pyplot as plt
    
    def _data_fn(x): return x[0][0], x[0][1]
    data_fn = _data_fn

    process_data_folder_fn = process_data_folder
    cmaps = plt.get_cmap('tab10', 10)

    for space in search_spaces.keys():
        plot_datas = []
        if space not in data_folders:
            continue

        for tag, all_list in zip(folders_tags, data_folders[space]):
            # do one plot at one time.
            # process the path to args.json
            all_list = [process_data_folder_fn(l) for l in all_list]
            # get the data
            plot_datas.append(load_data_from_experiment_root_dirs(
                all_list, target_fn=lambda x: tag, original_args=True)
                )
        for ind, tag in enumerate(folders_tags):
            # wrong, this is only ploting 1 curve, you should instead do an average ploting ..
            # steps, skdt, pvalue, top_k_arch = [], [], [], []
            res_dicts = plot_datas[ind][-1]
            print(res_dicts.keys())
            steps, skdt, skdt_std = plot_data_sparse_kdt_avg(res_dicts[space][tag], search_spaces, data_fn)
            _, top_k_arch, _ = plot_data_sparse_kdt(res_dicts[space][tag], search_spaces, fn_get_searched_ids)

            final_step=-1
            if space == 'darts_nds':
                final_step = -1
                steps = steps[:final_step]
                skdt = skdt[:final_step]
                skdt_std = skdt_std[:final_step]
                # ax.plot(steps[:-4], skdt[:-4], color=cmaps(ind), label=str(tag))
            elif space == 'nasbench101':
                final_step = -1
                steps = steps[:final_step]
                skdt = skdt[:final_step]
                skdt_std = skdt_std[:final_step]
            else:    
                final_step=0

            skdt = np.array(skdt)
            skdt_std = np.array(skdt_std)
            print(f'{tag}: kdt {skdt[final_step-1]} +- {skdt_std[final_step-1]}')
            print(f"mean performances {np.mean(search_spaces[space].query_gt_perfs(top_k_arch[final_step-1]))}") 
            print(f"std performances {np.std(search_spaces[space].query_gt_perfs(top_k_arch[final_step-1]))}") 
            print(f"best performances {np.max(search_spaces[space].query_gt_perfs(top_k_arch[final_step-1]))}") 
            print(f'best rank {MAXRANK[space] - np.max(top_k_arch[final_step-1])}')
            collect[tag] = (skdt[final_step-1], skdt_std[final_step-1],
                            np.mean(search_spaces[space].query_gt_perfs(top_k_arch[final_step-1])),
                            np.std(search_spaces[space].query_gt_perfs(top_k_arch[final_step-1])), 
                            np.max(search_spaces[space].query_gt_perfs(top_k_arch[final_step-1])),
                            search_spaces[space].query_gt_perfs(top_k_arch[final_step-1])
                            ) 

    return collect