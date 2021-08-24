import logging
import copy
import warnings
import torch
import torch.nn as nn


class Supernet(nn.Module):

    # use to distinguish landmark is enabled or not.
    landmark_loss_mode = False
    _model_spec = None
    _arch_parameters = []
    _redundant_modules = None
    _unused_modules = None
    ALWAYS_FULL_PARAMETERS = False

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
    
    @property
    def model_spec_cache(self):
        """Stores the current model_spec for oneshot algorithms"""
        return self._model_spec

    @model_spec_cache.setter
    def model_spec_cache(self, spec):
        self._model_spec = spec
    
    @property
    def warmup(self):
        if self.args.current_epoch:
            return self.args.current_epoch < self.args.supernet_warmup_epoch
        else:
            warnings.warn(f'No current_epoch set in the args. Possible reason is you get the warmup mode '
                          'before using it in policy.run() function. Return True here.')
            return True
    @property
    def forward_mode(self):
        """ Define the forward mode of a super-net, in any case, it support these options
            
            | Mode       |     Explanation |
            | -----------| ----------------|
            | oneshot    |     traditional oneshot forward, need the model_spec to function|
            | darts      |     Darts forward without training the arch_parameters.| 
            | softoneshot|     SoftOneShot, combining Darts with Oneshot approach to train supernet.  |
            ---------------------------
        """

        choices = ['oneshot', 'darts', 'softoneshot']
        # define this based on the policy.
        if 'darts' in self.args.search_policy or 'gdas' in self.args.search_policy:
            mode = 'darts' if 'darts' in self.args.search_policy else 'gdas'
            if self.training:
                if self.warmup:
                    return self.args.supernet_warmup_method
                else:
                    return mode if not self.landmark_loss_mode else self.args.landmark_forward_mode
            else:
                # to compute the kendall tau we need to use oneshot or softoneshot, 
                # for now, we could simply use 
                # TODO, we could simply do this to avoid any potential violations
                return self.args.supernet_eval_method
        else:
            # for other policy, we do not need to worry about the forward mode, just use SPOS safely.
            return 'oneshot'
    
    def set_landmark_mode(self, val):
        self.landmark_loss_mode = val

    def new(self, parallel=None):
        return copy.deepcopy(self)

    # regulate the forward function here.
    def forward(self, inputs):
        """Supernet forward function that supports multiple modes for various policy

        Parameters
        ----------
        inputs : tensor
            Image for classification - size (N, 3, W, H).

        Returns
        -------
        tuple
            logits, logits_aux
        """
        # logging.debug(f'Supernet Forward mode {self.forward_mode}')
        
        return {
            'oneshot': self.forward_oneshot, 
            'softoneshot': self.forward_softoneshot,
            'darts': self.forward_darts,
            'gdas': self.forward_gdas,
        }[self.forward_mode](inputs)

    def forward_oneshot(self, inputs):
        raise NotImplementedError()

    def forward_softoneshot(self, inputs):
        raise NotImplementedError()
    
    def forward_darts(self, inputs):
        raise NotImplementedError()

    def forward_gdas(self, inputs):
        raise NotImplementedError()

    @property
    def arch_parameters(self):
        """Parameter as buffers to be tracked in DARTS-based method.

        Returns
        -------
        list[torch.Tensor().cuda()requires_grad_()]
            A list of leaf tensor that can be passed into Optimizer directly. 
            Note that this is automatically moved to cuda() (first Cuda device available.)
            DataParallel should not be called under this section anyway.

        TODO check DataParallel in future for imagenet exps.
            This parameters is developed to mimic a normal nn.Parameters() but using buffer, so
            ideally there will be no clitch to apply DataParallel. 
                but let's see if there are some weird bugs.
        
        """
        if len(self._arch_parameters) == 0:
            # raise RuntimeError('You shoudl assign arch_parameters before usage')
            warnings.warn('No arch parameters registered to this module, you should set it first before using!')
        return self._arch_parameters

    @arch_parameters.setter
    def arch_parameters(self, params):
        # set the parameters and also do the registration
        if isinstance(params, torch.Tensor):
            params = [params]
        logging.debug('Assign arch parameters, copy to cuda directly. ')
        self._arch_parameters = [p.cuda().requires_grad_() for p in params]
        for i, p in enumerate(params):
            self.register_buffer(f'arch_parameter_{i}', p)
    
    """ Define the dynamic modules to save some GPU memory... Now it only works with NASBench101 net anyway. """
    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            # IPython.embed(header='check-redudant modules')
            for m in self.modules():
                if m.__str__().startswith('MixedVertex'): # this should be changed to match other supernet implementations.
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def unused_modules_off(self):
        if self._unused_modules is None or self.ALWAYS_FULL_PARAMETERS:
            return

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        self.unused_modules_back()
        d = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.unused_modules_off()
        return d