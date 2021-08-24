# generate latex table based on given data.
import tabulate 
import numpy as np
from nasws.cnn.visualization.analysis import load_data_from_experiment_root_dir, initialize_space_data_dict, add_supernetacc_finalperf_kdt

def latex_cmidrule(num_chunks:list, separate=1, offset=1):
    """compute and return the c-mid rule.
    """
    cstart = offset + separate + 1
    l = ''
    for c in num_chunks:
        l += '\cmidrule{' + '{}-{}'.format(cstart, cstart + c-1) + '}'
        cstart += c + separate
    return l


def latex_header(content, num_chunks, separate=1, offset=0):
    """return the header of a given column.
    
    Parameters
    ----------
    content : list of str
        Table Header between toprule and cmidrule.
    num_chunks : list of int
        Number of cmidrule sub-header
    separate : int, optional
        separate the latex for better visualization and layout, by default 1
    offset : int, optional
        offset to enable left column, by default 0

    Returns
    -------
    content_header, tabular_header
    """
    
    tabular_header = 'l' * (separate + offset)
    content_header = '&' * offset 
    assert len(content) == len(num_chunks)
    for c, n in zip(content, num_chunks):
        content_header += '&' * separate +'\multicolumn{' + str(n) + '}{c}{' + c + "}&"
        tabular_header += "c"* n + " l " * separate
    return content_header + '\\\\', tabular_header


def latex_oneline(chunks, title='', separate=1, offset=0, right_sep=1):
    l = '&' * offset + title + '&' * right_sep
    for c in chunks:
        l += "&" * separate + '&'.join([str(e) for e in c]) + "&" * right_sep
    return l + '\\\\'


def latex_table_subsections_v1(headers, ):
    """Serves as a template to build this.
        
    """
    h = ['nasbench101', 'nasbench201', 'darts-nds']
    content_header, tabular_header = latex_header(h, [3,3,2], offset=1)
    subheader = latex_oneline([[100, 200, 300,], [100, 200, 300,], [100, 200,],], offset=0, title='Epoch', separate=1)
    cmidrule = latex_cmidrule([3,3,2], offset=1, separate=1)

    print("\\begin{"+ 'tabular' +"}", '{', tabular_header , '}')
    print(content_header)
    print(cmidrule)
    print(subheader)
    print('\midrule')
    print('\\end{' + 'tabular' + '}')

def map_mean_std_to_latex(data, precision_mean=3, precision_std=2):
    """Mapping the mean and std into latex format.

    
    Parameters
    ----------
    data : list: (N, 2) shape

    """
    res = []
    for d in data:
        if d[0] < 1:
            res.append('{:.3f}$\pm${:.2f}'.format(d[0], d[1]))
        else:
            res.append('{:2.2f}$\pm${:1.2f}'.format(d[0], d[1]))
    return res


def latex_table_subsection_generation(tabular_header, content_header, cmidrule, subheader, latex_lines):

    print("\\begin{"+ 'tabular' +"}", '{', tabular_header , '}')
    print(content_header)
    print(cmidrule)
    print(subheader)
    print('\midrule')
    for l in latex_lines:
        print(l)
    print('\\bottomrule')
    print('\\end{' + 'tabular' + '}')


PRINT_SPACE_ORDER = ['nasbench101', 'nasbench201', 'darts_nds']
PRINT_METRIC_ORDER = ['acc','acc-std', 'kdt', 'p-random', 'perf', 'perf-std']

def num_to_latex_str(metrics):
    if metrics is None or len(metrics) != len(PRINT_METRIC_ORDER):
        return ['-', '-', '-', '-'] 
    # print(metrics)
    return [map_mean_std_to_latex([metrics[:2]])[0], 
                '{:.3f}'.format(metrics[2]), 
                '{:.3f}'.format(metrics[3]),  
            map_mean_std_to_latex([[metrics[4], metrics[5]]])[0]]


def parse_all_statistics_for_big_table(data_dict, space_name):
    res = []
    for i in range(len(data_dict[space_name]['acc'])):
        _r = []
        for k in PRINT_METRIC_ORDER:
            _r.append(float(data_dict[space_name][k][i]))
        res.append(_r)
    # print('loading ', len(res), 'statistics for ', space_name)
    return res


def parse_folder_data_to_table(expdir, str_filter, space_names, filter_fn, search_spaces, load_data_kwargs={}):

    accs, kdt, best_models, p_random, res_dicts = load_data_from_experiment_root_dir(expdir, str_filter, original_args=True, target_fn=filter_fn, **load_data_kwargs)


    final_latex_data = {}
    for space_name in space_names:
        for k in res_dicts[space_name].keys():

            # create this final table accordingly.
            if k not in final_latex_data.keys():
                final_latex_data[k] = {space_name: []}
            
            # load the full statistics
            # note that 'add_supernet_acc_finalperf_kdt' is designed for another purpose, 
            # we only adopt this idea here to simpler the code
            # ideally, this one could be improved. 
            try:
                space_data_dict = add_supernetacc_finalperf_kdt(res_dicts[space_name][k], search_spaces,            initialize_space_data_dict(), final_result=True, **load_data_kwargs)
                final_latex_data[k][space_name] = parse_all_statistics_for_big_table(space_data_dict, space_name)[0]
            except ValueError as e:
                print(f'Error loading statistics of {k} {space_name}, message is :{e}')
                final_latex_data[k][space_name] = None
            # in short, there is only 1 config for each dataset, that's why [0] is hard-coded.
            
    for k in sorted(list(final_latex_data.keys())):
        chunks = []
        for space_name in space_names:
            if space_name in final_latex_data[k].keys():
                _c = num_to_latex_str(final_latex_data[k][space_name])
            else:
                _c = num_to_latex_str(None)
            chunks.append(_c)
        # print(len(chunks))
        print(latex_oneline(chunks, f'{k} ', separate=1, right_sep=0))
