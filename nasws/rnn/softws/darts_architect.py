"""
This implements the DARTS way of optimizing soft-weights.

"""
import logging

import IPython
import torch
from torch.autograd import Variable

from ..darts_policy.architect import Architect, _clip
from torch.nn import Module


class SoftWSArchitect(Architect):
    def __init__(self, model, args):
        super(Architect, self).__init__()
        self.network_weight_decay = args.wdecay
        self.network_clip = args.clip

        # saving the model object of the class RNNModelSearch, that contains the DAG for training
        self.model = model
        self.model_new = self.model.new()

        # self.softws_parameters = model.softws_parameters()

        # initializing the architecture optimizer,
        # having as parameters to optimize the
        # architecture params saved in the model object
        # logging.warning("##### DEBUG $$$$$ Creating the optimizer")
        self.optimizer = torch.optim.Adam(self.model.softws_parameters(),
                                          lr=args.softws_lr,
                                          weight_decay=args.softws_wdecay)

    def get_update_parameters(self):
        # TODO Change to single vector? To avoid any potential issue.
        return self.model.softws_parameters()[self.model.genotype_id()]

    def _backward_step_unrolled(self,
                                hidden_train, input_train, target_train,
                                hidden_valid, input_valid, target_valid,
                                eta):
        genotype_id = self.model.genotype_id()
        # sync the copied model with the previous one.
        self.model_new.change_genotype(self.model.genotype(), self.model.genotype_id())

        # Unrolled? To detach and create a new model
        unrolled_model, clip_coef = self._compute_unrolled_model(hidden_train, input_train, target_train, eta)

        # TODO (change later) Doing the computation again, find out the reason later.
        tmp_loss, hidden_next = unrolled_model._loss(hidden_valid, input_valid, target_valid)
        grad_alpha = torch.autograd.grad(tmp_loss, unrolled_model.softws_parameters(), allow_unused=True)

        unrolled_loss, hidden_next = unrolled_model._loss(hidden_valid, input_valid, target_valid)
        unrolled_loss.backward()

        # TODO (change later) Manual copy the gradient into v.grad and assign to dalpha.
        # IPython.embed(header="Testing")
        for g, v in zip(grad_alpha, unrolled_model.softws_parameters()):
            v.grad = g
        dalpha = [v.grad for v in [unrolled_model.softws_parameters()[genotype_id]]]

        # update the soft-weight vector to the Genotype_id only! (or cluster)
        dtheta = [v.grad for v in unrolled_model.parameters()]
        _clip(dtheta, self.network_clip)
        vector = [dt.data for dt in dtheta]
        # second-order gradients.
        implicit_grads = self._hessian_vector_product(vector, hidden_train, input_train, target_train, r=1e-2)

        for g, ig in zip(dalpha, implicit_grads):

            g.data.sub_(eta * clip_coef, ig.data)

        # only assign to the weights should be.
        v = self.model.softws_parameters()[genotype_id]
        g = dalpha[0]
        if v.grad is None:
            v.grad = Variable(g.data)
        else:
            v.grad.data.copy_(g.data)

        # for v, g in zip(self.model.softws_parameters(), dalpha):
        #     if v.grad is None:
        #         v.grad = Variable(g.data)
        #     else:
        #         v.grad.data.copy_(g.data)
        # IPython.embed(header='checking backward of updating via DARTS sampler')

        return hidden_next

    def _debug_gradient_flow(self, unrolled_model):
        # debugging function here.

        # test 1, checking the normal model softws can be passed into the norm or not.
        softws = self.model.rnns[0]._softws_parameters
        weights = self.model.rnns[0].Ws # this should be a soft weights
        pesudo_loss = 0.0
        for w in weights:
            pesudo_loss += torch.sum(w)

        # check the gradient
        softws_grad = torch.autograd.grad(pesudo_loss, softws, retain_graph=True, allow_unused=True)
        assert softws_grad[self.model.genotype_id]
        IPython.embed(header='##Test 1, debugging the gradient flow of this function. PASSED!')


class SoftWSArchitectWPL(SoftWSArchitect):
    """
    TODO Add this later.
    Possibly add the
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO LATER.")

