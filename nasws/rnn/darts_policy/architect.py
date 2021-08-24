import IPython
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def _clip(grads, max_norm):
    total_norm = 0
    for g in grads:
        param_norm = g.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)
    return clip_coef


class Architect(object):

    def __init__(self, model, args):
        self.network_weight_decay = args.wdecay
        self.network_clip = args.clip

        # saving the model object of the class RNNModelSearch, that contains the DAG for training
        self.model = model
        self.model_new = self.model.new()

        # initializing the architecture optimizer,
        # having as parameters to optimize the
        # architecture params saved in the model object
        self.optimizer = torch.optim.Adam(self.get_update_parameters(), lr=args.arch_lr, weight_decay=args.arch_wdecay)

    def get_update_parameters(self):
        return self.model.arch_parameters()

    def _compute_unrolled_model(self, hidden, input, target, eta):
        # Unrolled model is created via this.

        # calling forward and loss on the model object
        loss, hidden_next = self.model._loss(hidden, input, target)

        # taking the network weights of the model, for RNN cell
        # means the embedding weights, the matrices of the RNN cell
        theta = _concat(self.model.parameters()).data

        grads = torch.autograd.grad(loss, self.model.parameters())
        clip_coef = _clip(grads, self.network_clip)
        dtheta = _concat(grads).data + self.network_weight_decay * theta
        # construct a new unrolled model from theta.
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, dtheta))
        return unrolled_model, clip_coef

    def step(self,
             hidden_train, input_train, target_train,
             hidden_valid, input_valid, target_valid,
             network_optimizer,
             unrolled):

        eta = network_optimizer.param_groups[0]['lr']
        self.optimizer.zero_grad()
        if unrolled:
            hidden = self._backward_step_unrolled(hidden_train, input_train, target_train, hidden_valid, input_valid,
                                                  target_valid, eta)
        else:
            hidden = self._backward_step(hidden_valid, input_valid, target_valid)
        # IPython.embed(header='check if the optimizer actually update.')
        self.optimizer.step()
        return hidden, None

    def _backward_step(self, hidden, input, target):
        loss, hidden_next = self.model._loss(hidden, input, target)
        loss.backward()
        return hidden_next

    def _backward_step_unrolled(self,
                                hidden_train, input_train, target_train,
                                hidden_valid, input_valid, target_valid,
                                eta):
        # Unrolled? To detach and create a new model
        unrolled_model, clip_coef = self._compute_unrolled_model(hidden_train, input_train, target_train, eta)
        unrolled_loss, hidden_next = unrolled_model._loss(hidden_valid, input_valid, target_valid)
        # torch.autograd.backward()
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        dtheta = [v.grad for v in unrolled_model.parameters()]
        _clip(dtheta, self.network_clip)
        vector = [dt.data for dt in dtheta]
        implicit_grads = self._hessian_vector_product(vector, hidden_train, input_train, target_train, r=1e-2)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta * clip_coef, ig.data)

        for v, g in zip(self.get_update_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        return hidden_next

    def _construct_model_from_theta(self, theta):
        model_new = self.model_new
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        # IPython.embed(header='check construct model')
        # return model_new.cuda()
        # This is updated in pytorch, 1.0, i.e. x.cuda() return None!
        # model_new.cuda()
        return model_new

    def _hessian_vector_product(self, vector, hidden, input, target, r=1e-2):
        """
        approximation of the Hessian of the train loss
        :param vector: gradients of arch params wrt the validation loss
        :param hidden: first hidden state for training
        :param input: input from train set
        :param target: target from train set
        :param r: hyper-parameter
        :return:
        """

        # computing the epsilon value (see paper, end page 4, note 2)
        R = r / _concat(vector).norm()

        # computing the w+ for the approx (see paper page 4)
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        # obtain train loss dependent on arch weights and w+
        # for the backward to obtain gradient wrt arch weights
        loss, _ = self.model._loss(hidden, input, target)

        # obtain the gradient wrt arch weights of train loss dependent on w+
        # see paper equation (7), first term of the numerator of the right member
        # IPython.embed(header='check hessian')
        grads_p = torch.autograd.grad(loss, self.get_update_parameters())

        # same process for w-
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)

        loss, _ = self.model._loss(hidden, input, target)

        # obtain the gradient wrt arch weights of train loss dependent on w-
        # see paper equation (7), second term of the numerator of the right member
        grads_n = torch.autograd.grad(loss, self.get_update_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        # computing the approx, paper equation (7), the right member.
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
