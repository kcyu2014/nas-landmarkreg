import torch
import nasws

class GDASArchitect:

    def __init__(self, model, args, module_forward_fn=None):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters,
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        self.module_forward_fn = module_forward_fn

    # def reset_logger(self):
    #     self.loggers = {
    #         'arch_objs': nasws.cnn.utils.AverageMeter(),
    #         'arch_top1': nasws.cnn.utils.AverageMeter(),
    #         'arch_top5': nasws.cnn.utils.AverageMeter()
    #     }
    
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        # get a random minibatch from the search queue with replacement
        arch_loss, logits, _ = self.module_forward_fn(self.model, input_valid, target_valid)
        arch_loss.backward()
        self.optimizer.step()
        # record
        arch_prec1, arch_prec5 = nasws.cnn.utils.accuracy(logits.data, target_valid.data, topk=(1, 5))
        # self.loggers['arch_objs'].update(arch_loss.item(),  input_valid.size(0))
        # self.loggers['arch_top1'].update(arch_prec1.item(), input_valid.size(0))
        # self.loggers['arch_top5'].update(arch_prec5.item(), input_valid.size(0))  
        return arch_loss
