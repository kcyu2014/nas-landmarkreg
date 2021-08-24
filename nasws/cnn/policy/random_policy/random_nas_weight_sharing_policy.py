"""
adopted from RandomNAS paper.

Simulate the 
"""

from ..cnn_general_search_policies import CNNSearchPolicy

# basically spos is the default behavior of CNNSearchpolicy...
class SinglePathOneShotPolicy(CNNSearchPolicy):

    def __init__(self, args) -> None:
        super().__init__(args)

    def run(self):
        raise NotImplementedError('To port the SPOS with so called latency and parameter sampler...')
    
class RandomNASCNNSearchPolicy(CNNSearchPolicy):

    def __init__(self, args) -> None:
        super().__init__(args)
