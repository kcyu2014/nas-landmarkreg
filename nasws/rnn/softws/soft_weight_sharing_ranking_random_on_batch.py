import logging

from ..random_policy.weight_sharing_ranking_random_on_batch import WeightSharingRandomRank
from ..random_policy.fairnas_weight_sharing import FairSampleWeightSharingRandomRank
import nasws.rnn.softws.soft_weight_sharing_model as model_module


class SoftWeightSharingRandomRank(WeightSharingRandomRank):

    def __init__(self, args):
        super(WeightSharingRandomRank, self).__init__(args=args)
        logging.info(">>> Soft-weight sharing enabled! <<<")
        self.model_fn = model_module.RNNModelSoftWS
        self.initialize_run()

    def initialize_model(self, ntokens, genotype_id, genotype):
        # Add search_space into args for RNNModelSoftWS function.
        self.args.search_space = self.search_space
        super(SoftWeightSharingRandomRank, self).initialize_model(ntokens, genotype_id, genotype)


class SoftWeightSharingRandomRankFairNAS(FairSampleWeightSharingRandomRank):

    def __init__(self, args):
        super(SoftWeightSharingRandomRankFairNAS, self).__init__(args=args)
        logging.info(">>> Soft-weight sharing enabled! Also FairNAS<<<")
        self.model_fn = model_module.RNNModelSoftWS
        self.initialize_run()

    def initialize_model(self, ntokens, genotype_id, genotype):
        # Add search_space into args for RNNModelSoftWS function.
        self.args.search_space = self.search_space
        super(SoftWeightSharingRandomRankFairNAS, self).initialize_model(ntokens, genotype_id, genotype)
