import IPython
import argparse

naoparser = argparse.ArgumentParser(description='NAO CIFAR-10')

naoparser.add_argument('--ratio', type=float, default=0.9)
naoparser.add_argument('--child_sample_policy', type=str, default='params')
naoparser.add_argument('--child_batch_size', type=int, default=64)
naoparser.add_argument('--child_eval_batch_size', type=int, default=500)
naoparser.add_argument('--child_epochs', type=int, default=200)
naoparser.add_argument('--child_layers', type=int, default=2, help='repeative layer, total layer = l * 3 + 2')
naoparser.add_argument('--child_nodes', type=int, default=5, help='SuperNet, number of nodes, == intermediate nodes.')
naoparser.add_argument('--child_channels', type=int, default=20, help='input channels.')
naoparser.add_argument('--child_cutout_size', type=int, default=None)
naoparser.add_argument('--child_grad_bound', type=float, default=5.0)
naoparser.add_argument('--child_lr_max', type=float, default=0.025)
naoparser.add_argument('--child_lr_min', type=float, default=0.001)
naoparser.add_argument('--child_keep_prob', type=float, default=1.0)
naoparser.add_argument('--child_drop_path_keep_prob', type=float, default=0.9)
naoparser.add_argument('--child_l2_reg', type=float, default=3e-4)
naoparser.add_argument('--child_use_aux_head', action='store_true', default=False)
naoparser.add_argument('--child_eval_epochs', type=str, default=50)
naoparser.add_argument('--child_stand_alone_epoch', type=str, default=40)
naoparser.add_argument('--child_arch_pool', type=str, default=None)
naoparser.add_argument('--child_lr', type=float, default=0.1)
naoparser.add_argument('--child_label_smooth', type=float, default=0.1, help='label smoothing')
naoparser.add_argument('--child_gamma', type=float, default=0.97, help='learning rate decay')
naoparser.add_argument('--child_decay_period', type=int, default=1, help='epochs between two learning rate decays')
naoparser.add_argument('--controller_seed_arch', type=int, default=100)
naoparser.add_argument('--controller_expand', type=int, default=8)
naoparser.add_argument('--controller_discard', action='store_true', default=False)
naoparser.add_argument('--controller_new_arch', type=int, default=300)
naoparser.add_argument('--controller_random_arch', type=int, default=400)
naoparser.add_argument('--controller_replace', action='store_true', default=False)
naoparser.add_argument('--controller_encoder_layers', type=int, default=1)
naoparser.add_argument('--controller_decoder_layers', type=int, default=1)
naoparser.add_argument('--controller_encoder_hidden_size', type=int, default=96)
naoparser.add_argument('--controller_encoder_emb_size', type=int, default=48)
naoparser.add_argument('--controller_mlp_layers', type=int, default=2)
naoparser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
naoparser.add_argument('--controller_decoder_hidden_size', type=int, default=96)
naoparser.add_argument('--controller_source_length', type=int, default=40, help='40 = 2 x 20, hard-coded for reduce cell (20) and conv cell (20)')
naoparser.add_argument('--controller_encoder_length', type=int, default=20, help='4 x 5 = 20. It is a hard-coded value. ')
naoparser.add_argument('--controller_decoder_length', type=int, default=40)
naoparser.add_argument('--controller_encoder_dropout', type=float, default=0)
naoparser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
naoparser.add_argument('--controller_decoder_dropout', type=float, default=0)
naoparser.add_argument('--controller_l2_reg', type=float, default=1e-4)
naoparser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
naoparser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
naoparser.add_argument('--controller_trade_off', type=float, default=0.8)
naoparser.add_argument('--controller_epochs', type=int, default=1000)
naoparser.add_argument('--controller_batch_size', type=int, default=100)
naoparser.add_argument('--controller_lr', type=float, default=0.001)
naoparser.add_argument('--controller_optimizer', type=str, default='adam')
naoparser.add_argument('--controller_grad_bound', type=float, default=5.0)

naoargs = naoparser.parse_args("") 

def default_nao_search_configs(nao_search_config, args):
    if 'nasbench101' in args.search_space:
        nodes = args.num_intermediate_nodes
        base_embed_size = 12 * (nodes - 1)
        # nao_search_config.controller_encoder_hidden_size = base_embed_size
        # nao_search_config.controller_decoder_hidden_size = base_embed_size
        nao_search_config.controller_encoder_emb_size = base_embed_size
        nao_search_config.controller_decoder_emb_size = base_embed_size
        from .utils_for_nasbench import NASBENCH_Node2ArchLength
        # setting vocabulary size.
        vocab_size = 1 + 2 + 3
        # vocab_size = nodes + 3 + 1
        nao_search_config.controller_encoder_vocab_size = vocab_size
        nao_search_config.controller_decoder_vocab_size = vocab_size
        nao_search_config.child_nodes = nodes
        
        length = NASBENCH_Node2ArchLength[nodes + 2]
        nao_search_config.controller_source_length = length
        nao_search_config.controller_encoder_length = length
        nao_search_config.controller_decoder_length = length

        if args.debug:
            # reduce time
            args.epochs = 8
            nao_search_config.child_epochs = 8
            nao_search_config.controller_seed_arch = 10
            nao_search_config.controller_random_arch = 10
            nao_search_config.controller_new_arch = 10
            nao_search_config.controller_epochs = 4
            nao_search_config.child_eval_epochs = "1"

        nao_search_config.controller_mlp_layers = 2
        nao_search_config.controller_mlp_hidden_size = 64
        nao_search_config.controller_encoder_hidden_size = 16
        nao_search_config.controller_decoder_hidden_size = 16
        
        nao_search_config.controller_discard = True

    elif 'nasbench201' in args.search_space:
        nodes = args.num_intermediate_nodes
        base_embed_size = 12 * (nodes - 1)
        # nao_search_config.controller_encoder_hidden_size = base_embed_size
        # nao_search_config.controller_decoder_hidden_size = base_embed_size
        nao_search_config.controller_encoder_emb_size = base_embed_size
        nao_search_config.controller_decoder_emb_size = base_embed_size
        from .utils_for_nasbench201 import NASBENCH201_Node2ArchLength, ALLOWED_OPS
        # setting vocabulary size.
        vocab_size = 1 + len(ALLOWED_OPS)
        # vocab_size = nodes + 3 + 1
        nao_search_config.controller_encoder_vocab_size = vocab_size
        nao_search_config.controller_decoder_vocab_size = vocab_size
        nao_search_config.child_nodes = nodes
        
        length = NASBENCH201_Node2ArchLength[nodes]
        nao_search_config.controller_source_length = length
        nao_search_config.controller_encoder_length = length
        nao_search_config.controller_decoder_length = length
        nao_search_config.controller_encoder_dropout = 0.1
        nao_search_config.controller_decoder_dropout = 0.1

        nao_search_config.controller_mlp_layers = 2
        nao_search_config.controller_mlp_hidden_size = 64
        nao_search_config.controller_encoder_hidden_size = 16
        nao_search_config.controller_decoder_hidden_size = 16
    
    elif 'darts' in args.search_space:
        nodes = args.num_intermediate_nodes # equal to B
        base_embed_size = 16 * (nodes - 1)
        nao_search_config.controller_encoder_hidden_size = base_embed_size
        nao_search_config.controller_decoder_hidden_size = base_embed_size
        nao_search_config.controller_encoder_emb_size = base_embed_size
        nao_search_config.controller_decoder_emb_size = base_embed_size
        from .utils_for_darts import DARTS_Node2ArchLength, PRIMITIVES
        # setting vocabulary size.
        vocab_size = 2 + len(PRIMITIVES) + nodes
        # vocab_size = nodes + 3 + 1
        nao_search_config.controller_encoder_vocab_size = vocab_size
        nao_search_config.controller_decoder_vocab_size = vocab_size
        nao_search_config.child_nodes = nodes
        
        length = DARTS_Node2ArchLength[nodes]
        nao_search_config.controller_source_length = length
        nao_search_config.controller_encoder_length = length
        nao_search_config.controller_decoder_length = length
        nao_search_config.controller_encoder_dropout = 0
        nao_search_config.controller_decoder_dropout = 0

        nao_search_config.controller_mlp_layers = 2
        nao_search_config.controller_mlp_hidden_size = 64

        nao_search_config.controller_discard = True

        if args.dataset == 'imagenet':
            nao_search_config.controller_new_arch = 10
            nao_search_config.controller_random_arch = 20
            nao_search_config.controller_seed_arch = 50
            nao_search_config.child_stand_alone_epoch = 20

    elif 'proxylessnas' in args.search_space:
        pass
    else:
        pass

    if args.debug:
        # reduce time
        args.epochs = 8
        args.save_every_epoch = 1
        nao_search_config.child_epochs = 8
        nao_search_config.child_stand_alone_epoch = 2
        nao_search_config.controller_seed_arch = 10
        nao_search_config.controller_random_arch = 10
        nao_search_config.controller_new_arch = 10
        nao_search_config.controller_epochs = 4
        nao_search_config.child_eval_epochs = "1"

    return nao_search_config