"""
This is a cnn search policy wrapper.
"""

#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/27 下午1:12
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================

import logging
from nasws.cnn.policy.random_policy.random_nas_weight_sharing_policy import RandomNASCNNSearchPolicy

from nasws.cnn.policy.darts_policy.darts_official_configs import darts_official_args
from nasws.cnn.policy.enas_policy.enas_micro.train_search import get_enas_microcnn_parser
from utils import DictAttr, wrap_config

# FIX this part later...
# sampler based policies.
from nasws.cnn.policy.darts_policy.darts_search_policy import DARTSMicroCNNSearchPolicy
# from nasws.cnn.policy.darts_policy.darts_search_policy import DARTSMicroCNNSearchPolicy
from nasws.cnn.policy.pcdarts_policy.pcdarts_search_policy import PCDARTSSearchPolicy
from nasws.cnn.policy.pcdarts_policy.darts_new_policy import DARTSCNNPolicy

from nasws.cnn.policy.gdas_policy.gdas_search_policy import GDASCNNSearchPolicy
from nasws.cnn.policy.enas_policy.enas_search_policy import ENASNasBenchSearch, ENASMicroCNNSearchPolicy, ENASNasBenchGroundtruthPolicy
from nasws.cnn.policy.nao_policy.nao_search_policy import NAOSearchPolicy, NAONasBenchGroundTruthPolicy
from nasws.cnn.policy.random_policy import *

# nasbench arguments
from nasws.cnn.policy.enas_policy.configs import enasargs
from nasws.cnn.policy.nao_policy.configs import naoargs
from nasws.cnn.policy.darts_policy.configs import dartsargs
from nasws.cnn.policy.pcdarts_policy.configs import pcdarts_args


class CNNCellSearchPolicyFactory:

    @staticmethod
    def factory(args):
        """

        Logic: search_policy -> 
                search_space is isolated from the policy.
        """

        if 'pcdarts' in args.search_policy:
            return PCDARTSSearchPolicy(args, pcdarts_args)

        if 'darts' in args.search_policy:
            return DARTSCNNPolicy(args, dartsargs)
        
        if 'gdas' in args.search_policy:
            return GDASCNNSearchPolicy(args)

        if args.search_policy in ['nao', 'nao-groundtruth']:
            """
            The experiment settting here is purely for NASBench experiments, 
            rerun in original space, need to do more operations.
            """

            if args.search_policy == 'nao':
                return NAOSearchPolicy(args)
            elif args.search_policy == 'nao-groundtruth':
                return NAONasBenchGroundTruthPolicy(args)
        
        # else: using SPOS by default.
        if 'nasbench101' in args.search_space:
            from nasws.cnn.policy.darts_policy.darts_search_policy import DARTSNasBenchSearch, DARTSMicroCNNSearchPolicy
            from nasws.cnn.policy.nao_policy.nao_search_policy import NasBenchNetSearchNAO
            from nasws.cnn.policy.darts_policy.fbnet_search_policy import FBNetNasBenchSearch
            from nasws.cnn.policy.random_policy.nasbench_weight_sharing_maml_policy import MAMLPlusNasBenchPolicy
            from nasws.cnn.policy.random_policy.nasbench_weight_sharing_policy import NasBenchWeightSharingPolicy, \
                NasBenchWeightSharingNewEval
            from nasws.cnn.policy.random_policy.nasbench_weight_sharing_fairnas_policy import NasBenchNetTopoFairNASPolicy

        elif 'darts' in args.search_space:
            # replace gradually with the spos search space as change model spec is done.
            return RandomNASCNNSearchPolicy(args)

        elif 'nasbench201' in args.search_space:
            if 'hamming' in args.search_policy:
                # Policy for other pipelines...
                return NASBench201HammingDistancePolicy(args)
            return RandomNASCNNSearchPolicy(args)

        elif args.search_space == 'original':
            """
            Experiment here should be done with the following setting:
                Each policy has its own model
                    Run their original search on it for 10 times, to get 10 best models
                    WS-Random to get another 10 random models
                in total, 3 * (10 + 10) = 60 training from scratch, in each of space
                ~ 180 GPU days + exp times. 
            """
            # intermediate node = 5 in this cases.
            if args.search_policy == 'enas':
                # get default parser
                enas_args = get_enas_microcnn_parser().parse_args('')
                wrap_config(enas_args, '',
                            keys=['seed', 'epochs', 'data'], args=args)
                return ENASMicroCNNSearchPolicy(args, enas_args)
            elif args.search_policy == 'darts':
                steps = 4 # this is fixed
                darts_config = darts_official_args
                wrap_config(darts_config, '', keys=['seed', 'epochs', 'data'], args=args)
                return DARTSMicroCNNSearchPolicy(args, darts_config)
            elif args.search_policy == 'nao':
                raise NotImplementedError("TODO This is not yet supported. "
                                          "please run the official code for 10 times, "
                                          "and parse the data manually.")
        else:
            raise NotImplementedError("not supported")
