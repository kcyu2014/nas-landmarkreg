import shutil
import torch
import torch.optim as optim

from monodepth.models import MidasNet, MidasNetSearch
from monodepth.optim.optimizer import MGDA, GDA
from monodepth.validation import Validation
from monodepth.utils.checkpointer import Checkpointer, load_json, save_args
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
from nni.nas.pytorch.fixed import FixedArchitecture


class TrainParams:
    # keeping the hyper-parameters untouch in this code.
    def __init__(self, config, output, args, do_resume=False, init_from=None):
        # torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # device
        self.device = torch.device("cuda")
        self.args = args
        # model
        self.model = MidasNetSearch(backbone="resnext101_wsl", args=args)
        # mutator = 
        # load the fix_architecture.json
        
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
                {"params": self.model.pretrained.parameters(), "lr": 0.00001},
            ],
            lr=0.0001,
            betas=(0.9, 0.999),
            weight_decay=0, # specify here to remind myself
        )

        self.epoch_length = 3240
        self.num_epochs = 300

        self.optimizer = GDA(algorithm=algo, num_tasks=len(self.tasks))
        self.scheduler = torch.optim.lr_scheduler.StepLR(algo, 100, gamma=0.1)

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
            use_console=True, tensorboard_out=output + "/log", img_index=[], logger_out=output + '/log_retrain.txt'
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

        #loading the architecture
        arch = load_json(args.arch_json)
        self.reporter.info(f'Loading json architecture {args.arch_json}')
        self.reporter.info(f'Architecture: {arch}')
        # self.checkpointer
        self.mutator = FixedArchitecture(self.model, arch)
        self.args.save_every_epoch = 1000 # disable the landmark computation...
        shutil.copyfile(args.arch_json, output + '/arch.json')
        save_args(output + '/args.json', args)

