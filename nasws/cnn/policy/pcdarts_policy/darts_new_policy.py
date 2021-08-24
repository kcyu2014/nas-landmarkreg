from ..differentiable_policy import DifferentiableCNNPolicy
from .architect import Architect

def update_darts_args(args, darts_args):
    return darts_args


class DARTSCNNPolicy(DifferentiableCNNPolicy):
    top_K_complete_evaluate = 200
    top_K_num_sample = 1000

    def __init__(self, args, darts_args) -> None:
        super(DARTSCNNPolicy, self).__init__(args)
        darts_args = update_darts_args(args, darts_args)
        self.darts_args = darts_args
        self.args.policy_args = darts_args
        self.policy_epochs = darts_args.epochs

    # otherwise, define the run profile
    def initialize_model(self):
        tmp_resume, self.args.resume = self.args.resume, False
        model, optimizer, scheduler = super().initialize_model()
        model.ALWAYS_FULL_PARAMETERS = True
        self.args.resume = tmp_resume
        # load architect
        # anyway this is simply a pointer to DataParallel..., always using self.model to avoid the problem of module. smt.
        self.architect = Architect(self.model, self.args.policy_args, None) # as a wrapper to train the arch parames
        if self.args.resume:
            self.resume_from_checkpoint()
        return model, optimizer, scheduler
    

