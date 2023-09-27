# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py.
Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.
Train on ImageNet with default parameters:
.. code-block: bash
    python imagenet.py 
or show all options you can change:
.. code-block: bash
    python imagenet.py --help
"""
import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
import deeplake

class ImageNetLightningModel(LightningModule):
    """
    >>> ImageNetLightningModel(data_path='missing')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ImageNetLightningModel(
      (model): ResNet(...)
    )
    """
    
    # pull out resnet names from torchvision models
    MODEL_NAMES = sorted(
        name
        for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )
    
    def __init__(
        self,
        arch: str = os.environ['ARCH'],
        pretrained: bool = False,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        workers: int = 0,
        num_threads:int = None, 
        prefetch_factor:int = 2,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.num_threads = num_threads
        self.prefetch_factor = prefetch_factor
        print('*' * 80)
        print(f'*************** Loading model {self.arch}')
        print('*' * 80)
        self.model = models.__dict__[self.arch](pretrained=self.pretrained)
        
        self.train_dataset = deeplake.load(
            "hub://activeloop/imagenet-val", read_only=True
        )
        self.val_dataset = deeplake.load(
            "hub://activeloop/imagenet-val", read_only=True
        )
        
        self.train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.val_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, target = batch
        target = target.squeeze()
        output = self(images)
        
        print (f"output_shape: {output.shape}, target_shape: {target.shape}")

        loss_train = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train
    
    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        target = target.squeeze()
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log(f"{prefix}_loss", loss_val, on_step=True, on_epoch=True)
        self.log(f"{prefix}_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")
    
    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        train_loader = self.train_dataset.dataloader() \
            .transform({"images": self.train_transform, "labels": None}) \
            .batch(self.batch_size, drop_last=True) \
            .shuffle(False) \
            .pytorch(num_workers=self.workers, 
                     num_threads=self.num_threads,
                     prefetch_factor=self.prefetch_factor,
                     decode_method = {'images': 'pil'},
                     distributed=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = self.train_dataset.dataloader() \
            .transform({"images": self.val_transform, "labels": None}) \
            .batch(self.batch_size, drop_last=True) \
            .pytorch(num_workers=self.workers, 
                     num_threads=self.num_threads,
                     prefetch_factor=self.prefetch_factor,
                     decode_method = {'images': 'pil'},
                     distributed=True)
        return val_loader
    
    def test_dataloader(self):
        return self.val_dataloader()
    
    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")
    
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("ImageNetLightningModel")
        parser.add_argument(
            "-a",
            "--arch",
            metavar="ARCH",
            default="resnet18",
            choices=ImageNetLightningModel.MODEL_NAMES,
            help=("model architecture: " + " | ".join(ImageNetLightningModel.MODEL_NAMES) + " (default: resnet18)"),
        )
        parser.add_argument(
            "-j", "--workers", default=10, type=int, metavar="N", help="number of data loading workers (default: 4)"
        )
        parser.add_argument(
            "-t", "--num-threads", default=None, type=int, metavar="N", help="number of threads used to fetch and decompress images (default: Automatically determined)"
        )
        parser.add_argument(
            "-p", "--prefetch-factor", default=2, type=int, metavar="N", help="Number of batches to transform and collate in advance per worker. Defaults to 2"
        )
        parser.add_argument(
            "-g", "--num-gpu", default=1, type=int, metavar="G", help="number of gpus (default: 1)"
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            default=256,
            type=int,
            metavar="N",
            help="mini-batch size (default: 256), this is the total batch size of all GPUs on the current node"
            " when using Data Parallel or Distributed Data Parallel",
        )
        parser.add_argument(
            "--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="lr"
        )
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=1e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-4)",
            dest="weight_decay",
        )
        parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
        return parent_parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)
    
    # if args.accelerator == "ddp":
    #     # When using a single GPU per process and per
    #     # DistributedDataParallel, we need to divide the batch size
    #     # ourselves based on the total number of GPUs we have
    #     args.batch_size = int(args.batch_size / max(1, args.gpus))
    #     args.workers = int(args.workers / max(1, args.gpus))
   
    model = ImageNetLightningModel(**vars(args))
    trainer = pl.Trainer(max_epochs=10,  accelerator="gpu", devices=args.num_gpu, strategy='ddp')
    # pl.Trainer.from_argparse_args(args)
    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)
num_epochs_ = None

if 'EPOCHS' in os.environ:
    try:
        num_epochs_ = int(os.environ['EPOCHS'])
    except:
        pass


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    # parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument(
        "-e", "--evaluate", dest="evaluate", action="store_true",
        help="evaluate model on validation set"
    )
    parent_parser.add_argument("--seed", type=int, default=42, help="seed for initializing training.")
    parser = ImageNetLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler="simple", deterministic=True, max_epochs=num_epochs_ or 90)
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    run_cli()

