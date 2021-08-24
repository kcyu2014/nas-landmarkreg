from nasws.cnn.policy.cnn_general_search_policies import CNNSearchPolicy
from nasws.cnn.search_space.darts.utils import change_model_spec


# this is also a merge between these two
class DARTSBenchWeightSharingPolicy(CNNSearchPolicy):
    """
    Darts Search Space supporter.

    """
    def __init__(self, args):
        super(DARTSBenchWeightSharingPolicy, self).__init__(args, sub_dir_path=None)
        self._change_model_fn = change_model_spec

# alias.
NDSBenchWeightSharingPolicy = DARTSBenchWeightSharingPolicy