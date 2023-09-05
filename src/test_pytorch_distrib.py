# Copyright 2022 CRS4 (http://www.crs4.it/)
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

import os
from clize import run
import deeplake
from torchvision import transforms
from tqdm import tqdm
from IPython import embed
import torch


def open_dataset(
    src,
    aws_access_key_id,
    aws_secret_access_key,
    endpoint_url,
):
    if aws_access_key_id and aws_secret_access_key and endpoint_url:
        creds = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "endpoint_url": endpoint_url,
        }
    else:
        creds = None

    ds = deeplake.load(src, creds=creds)
    return ds


def loop_read_enterprise(
    ds,
    lr,
    epochs=3,
    shuffle=False,
    distributed=False,
    backend="gloo",
):
    tform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
            transforms.RandomResizedCrop(
                224, scale=(0.1, 1.0), ratio=(0.8, 1.25), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if distributed:
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    dp = (
        ds.dataloader()
        .transform({"images": tform, "labels": None})
        .batch(2)
        .shuffle(shuffle)
        .pytorch(
            decode_method={"images": "pil"},
            prefetch_factor=4,
            distributed=distributed,
            # num_workers=2,
        )
    )
    
    pbar = tqdm(dp)
    print(f"tqdm rank: {lr}")
    for _ in range(epochs):
        for x in dp:
            d = x["images"]
            l = x["labels"]
            print(f"tqdm rank: {lr}, label={l.shape}, data={d.shape}")
    
    pbar.close()

def test(
    src,
    *,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    endpoint_url=None,
    epochs=3,
    tens_workers=32,
    torch_workers=8,
    shuffle=False,
    distributed=False,
    backend="gloo",
):
    # supporting torchrun
    global_rank = int(os.getenv("RANK", default=0))
    local_rank = int(os.getenv("LOCAL_RANK", default=0))
    world_size = int(os.getenv("WORLD_SIZE", default=1))
    
    print (f"gr: {global_rank}, lr: {local_rank}, ws: {world_size}")

    ds = open_dataset(
        src=src,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )
    
    loop_read_enterprise(
        ds,
        local_rank,
        shuffle=shuffle,
        epochs=epochs,
        distributed=distributed,
        backend=backend,
    )


# parse arguments
if __name__ == "__main__":
    run(test)
