from monodepth.models import MidasNet, MidasNetSearch
from monodepth.optim.optimizer import MGDA, GDA
import torch
import torch.optim as optim
from monodepth.validation import Validation
from monodepth.utils.checkpointer import Checkpointer
from monodepth.utils.reporter import Reporter
from monodepth.optim.scheduler import ConstLR
from monodepth.loss import (
    ScaleAndShiftInvariantLoss,
    ProcrustesLoss,
    TrimmedProcrustesLoss,
)
from monodepth.data import multi_data_loader
from monodepth.data.datasets import MegaDepth, ReDWeb, Movies3d
from monodepth.data.datasets.transforms import (
    RandomCrop,
    RandomFliplr,
    Rescale,
    PrepareForNet,
    Resize,
    NormalizeImage,
)
from torchvision.transforms import Compose
from monodepth.metric import DisparityRmse
from collections import OrderedDict
from monodepth.nas.api import NonZeroRandomMutator, NonZeroRandomMutatorSync


class TrainParams:
    def __init__(self, config, output, args, do_resume=False, init_from=None):
        # torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # device
        self.device = torch.device("cuda")
        self.args = args
        # model
        self.model = MidasNetSearch(backbone="resnext101_wsl", args=args)
        if 'spos' in args.supernet_train_method:
            if args.mutator == 'nonzero':
                mutator = NonZeroRandomMutator(self.model)
            elif args.mutator == 'nonzero_sync':
                mutator = NonZeroRandomMutatorSync(self.model)
            else:
                raise NotImplementedError(f"Mutator {args.mutator} is not implemented for SPOS method")
        else:
            raise NotImplementedError(f"Method {args.supernet_train_method} is not implemented for MonoDepth")
        self.mutator = mutator
        # tasks
        self.tasks = OrderedDict()

        self.tasks["ReDWeb"] = {
            "dataset": ReDWeb(
                config["datasets"]["ReDWeb"]["train"],
                config["datasets"]["ReDWeb"]["datapath"],
                transform=Compose(
                    [
                        Resize(384, 384, keep_aspect_ratio=True),
                        RandomCrop(384, 384),
                        RandomFliplr(),
                        Rescale(),
                        NormalizeImage(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                        PrepareForNet(),
                    ]
                ),
                mode="train",
            ),
            "loss": TrimmedProcrustesLoss(),
        }

        self.dataloader = multi_data_loader.MultiDataLoader(
            {k: v["dataset"] for k, v in self.tasks.items()},
            batch_size=8,
            num_workers=4,
        )

        # optimization
        algo = optim.Adam(
            [
                {"params": self.model.scratch.parameters()},
                {"params": self.model.pretrained.parameters(), "lr": args.backbone_learning_rate},
            ],
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
        if args.backbone_learning_rate == 0.0:
            for param in self.model.pretrained.parameters():
                param.requires_grad = False
        
        if args.backbone_weights:
            # load the pretrained weights weights accordingly
            state = torch.load(args.backbone_weights)
            self.model.load_state_dict(state, strict=False)

        # this should make the training super fast
        self.epoch_length = 3240
        self.num_epochs = args.epochs

        # wrapping the optimizer
        self.optimizer = GDA(algorithm=algo, num_tasks=len(self.tasks))
        
        if args.learning_rate_scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(algo, args.epochs // 3, gamma=0.1)
 
        elif args.learning_rate_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(algo, float(args.epochs), eta_min=args.learning_rate_min)

        print(f'Using {self.scheduler} with initial learning rate {args.learning_rate}.')

        # checkpointing
        self.checkpointer = Checkpointer(
            output + "/checkpoint.pt",
            self.model,
            self.optimizer.algo,
            self.scheduler,
            frequency=10,
            do_resume=do_resume,
        )

        # reporting
        self.reporter = Reporter(
            use_console=True, tensorboard_out=output + "/log", img_index=[], logger_out=output + '/log_search.txt'
        )

        # validation
        validation_tasks = OrderedDict()
        validation_tasks["ReDWeb"] = {
            "dataset": ReDWeb(
                config["datasets"]["ReDWeb"]["validation"],
                config["datasets"]["ReDWeb"]["datapath"],
                transform=Compose(
                    [
                        Resize(
                            384,
                            384,
                            resize_target=False,
                            keep_aspect_ratio=True,
                            ensure_multiple_of=32,
                            resize_method="minimal",
                        ),
                        NormalizeImage(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                        PrepareForNet(),
                    ]
                ),
            ),
            "metric": DisparityRmse(),
        }

        self.validation = Validation(
            self.model, validation_tasks, self.reporter, self.device, frequency=args.save_every_epoch
        )
