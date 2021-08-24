from utils import DictAttr

darts_dict_config = {}
darts_dict_config['data'] = 'data/penn/'
darts_dict_config['emsize'] = 300
darts_dict_config['nhid'] = 300
darts_dict_config['nhidlast'] = 300
darts_dict_config['lr'] = 20
darts_dict_config['clip'] = 0.25
darts_dict_config['epochs'] = 50
darts_dict_config['batch_size'] = 256
darts_dict_config['bptt'] = 35
darts_dict_config['dropout'] = 0.75
darts_dict_config['dropouth'] = 0.25
darts_dict_config['dropoutx'] = 0.75
darts_dict_config['dropouti'] = 0.2
darts_dict_config['dropoute'] = 0.0
darts_dict_config['seed'] = 3
darts_dict_config['nonmono'] = 5
darts_dict_config['cuda'] = True
darts_dict_config['log_interval'] = 50
darts_dict_config['save'] = 'EXP_DARTS_SEARCH'
darts_dict_config['alpha'] = 0.0
darts_dict_config['beta'] = 1e-3
darts_dict_config['wdecay'] = 5e-7
darts_dict_config['small_batch_size'] = -1
darts_dict_config['max_seq_len_delta'] = 20
darts_dict_config['single_gpu'] = True
darts_dict_config['gpu'] = 0
darts_dict_config['unrolled'] = True
darts_dict_config['arch_wdecay'] = 1e-3
darts_dict_config['arch_lr'] = 3e-3
darts_dict_config['num_intermediate_nodes'] = 8
darts_dict_config['num_operations'] = 4
darts_dict_config['handle_hidden_mode'] = 'NONE'
darts_dict_config['main_path'] = 'NONE'

darts_configs = DictAttr(darts_dict_config)
