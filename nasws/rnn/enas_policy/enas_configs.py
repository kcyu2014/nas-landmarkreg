"""
define the arguments here.

Please change to
    * enas_config_default for node 8 case.
    * enas_config_small for node 2 case.
"""

from utils import DictAttr

enas_config_default = {
    # Add for only this code
    "comment": "enas-space8-debug",
    "save": "EXP",
    "num_intermediate_nodes": 8,
    "num_blocks": 8,
    "num_operations": 4,
    "shared_average_all": False,
    "max_epoch": 250,
    "main_path": None,

    "activation_regularization": False,
    "activation_regularization_amount": 2.0,
    "alpha_fisher": 1.0,
    "batch_size": 64,
    "cnn_hid": 64,
    "controller_grad_clip": 0,
    "controller_hid": 100,
    "controller_lr": 0.00035,
    "controller_lr_cosine": False,
    "controller_lr_max": 0.05,
    "controller_lr_min": 0.001,
    "controller_max_step": 2000,
    "controller_optim": "adam",
    "controller_update_fisher": False,
    "cuda": True,
    "data_dir": "data",
    "data_path": "data/penn",
    "dataset": "ptb",
    "derive_num_sample": 100,
    "discount": 1.0,
    "ema_baseline_decay": 0.95,
    "entropy_coeff": 0.0001,
    "entropy_mode": "reward",
    "fisher_clip_by_norm": 10.0,
    "lambda_fisher": 0.9,
    "load_path": "",
    "log_dir": "logs",
    "log_level": "DEBUG",
    "log_step": 50,
    "max_save_num": 4,
    "mode": "train",
    "model_dir": "logs/ptb_2018-06-16_08-54-31",
    "model_name": "ptb_2018-06-16_08-54-31",
    "momentum": False,
    "nb_batch_reward_controller": 1,
    "network_type": "rnn",
    "norm_stabilizer_fixed_point": 5.0,
    "norm_stabilizer_regularization": False,
    "norm_stabilizer_regularization_amount": 1.0,
    "num_batch_per_iter": 1,
    "num_gpu": 1,
    "policy_batch_size": 1,
    "ppl_square": False,
    "random_seed": 12345,
    "reward_c": 80,
    "save_epoch": 30,
    "set_fisher_zero_per_iter": -1,
    "shared_ce_fisher": True,
    "shared_cnn_types": [
        "3x3",
        "5x5",
        "sep 3x3",
        "sep 5x5",
        "max 3x3",
        "max 5x5"
    ],
    "shared_decay": 0.96,
    "shared_decay_after": 15,
    "shared_dropout": 0.4,
    "shared_dropoute": 0.1,
    "shared_dropouti": 0.65,
    "shared_embed": 1000,
    "shared_grad_clip": 0.25,
    "shared_hid": 1000,
    "shared_initial_step": 0,
    "shared_l2_reg": 1e-07,
    "shared_lr": 20.0,
    "shared_max_step": 70,
    "shared_num_sample": 1,
    "shared_optim": "sgd",
    "shared_rnn_activations": [
        "tanh",
        "ReLU",
        "identity",
        "sigmoid"
    ],
    "shared_rnn_max_length": 35,
    "shared_valid_fisher": True,
    "shared_wdrop": 0.5,
    "softmax_temperature": 5.0,
    "start_evaluate_diff": 1000,
    "start_using_fisher": 1000,
    "start_training_controller": 0,
    "stop_training_controller": 1000,
    "tanh_c": 2.5,
    "temporal_activation_regularization": False,
    "temporal_activation_regularization_amount": 1.0,
    "test_batch_size": 1,
    "tie_weights": True,
    "train_controller": True,
    "use_tensorboard": True
}


def _enas_config_small():
    base_dict = DictAttr(enas_config_default)
    base_dict.update_dict(
        {
            # Add for only this code
            "comment": "enas-test-1",
            "num_intermediate_nodes": 2,
            "num_blocks": 2,
            "controller_max_step": 20,
            "shared_max_step": 20,
            "num_batch_per_iter": 20,
            "log_step": 10,
        }
    )
    return base_dict


enas_config = DictAttr(enas_config_default)
enas_config_small = _enas_config_small()
