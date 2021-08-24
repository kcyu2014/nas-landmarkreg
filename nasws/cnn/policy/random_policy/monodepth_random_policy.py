from ..cnn_general_search_policies import CNNSearchPolicy



class MonodepthSearchPolicy(CNNSearchPolicy):

    def __init__(self, args) -> None:
        super().__init__(args)
        # load the config and params to train the dataset...
        # reset the train_fn... ?
        self.train_fn = None # should be identical to MnodepthTraining.train_one_epoch ...
        self.eval_fn = None

    def initialize_search_space(self):
        if self.args.search_space == 'nasbench201':
            pass
        elif self.args.search_space == 'nasbench201-upsample':
            pass
        else:
            raise ValueError(f'Search space {self.args.search_space} not supported yet!')

    def initialize_run(self, sub_dir_path):
        # load dataset need to be redo, criterion need to be redo as well...
        pass
    
    def model_to_gpus(self):
        pass

    def initialize_model(self, resume):
        self.mutator = None
        # set the mutator accordingly.

    def load_dataset(self, shuffle_test):
         # redo the dataset loader based on this!! from the config files....
         pass

    def random_sampler(self, model, architect, args):
        """ simple ! just do a simple reset ... it will do the trick """
        self.mutator.reset()
