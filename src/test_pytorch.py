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

from clize import run
import deeplake
from torchvision import transforms
from tqdm import tqdm


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


def loop_read_raw(ds, epochs=3):
    for _ in range(epochs):
        for x in tqdm(ds.tensors["images"]):
            d = x.tobytes()


def loop_read_pytorch(
    ds,
    epochs=3,
    num_workers=8,
    shuffle=False,
):
    dp = ds.pytorch(
        batch_size=512,
        num_workers=num_workers,
        transform={"images": None, "labels": None},
        decode_method={"images": "tobytes"},  # do not decode
        collate_fn=lambda x: x,
        shuffle=shuffle,
    )
    for _ in range(epochs):
        for x in tqdm(dp):
            ...


def loop_read_tensors(
    ds,
    epochs=3,
    num_workers=8,
    shuffle=False,
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
    dp = ds.pytorch(
        batch_size=512,
        num_workers=num_workers,
        transform={"images": tform, "labels": None},
        shuffle=shuffle,
        decode_method={"images": "pil"},
    )
    for _ in range(epochs):
        for x in tqdm(dp):
            d = x["images"]


def loop_read_enterprise(
    ds,
    epochs=3,
    num_workers=8,
    shuffle=False,
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
    dp = (
        ds.dataloader()
        .transform({"images": tform, "labels": None})
        .batch(512)
        .shuffle(shuffle)
        .pytorch(
            num_workers=8,
            decode_method={"images": "pil"},
            prefetch_factor=4,
            num_threads=4,
        )
    )
    for _ in range(epochs):
        for x in tqdm(dp):
            d = x["images"]


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
):
    ds = open_dataset(
        src=src,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )
    loop_read_raw(ds, epochs=epochs)
    loop_read_pytorch(
        ds,
        num_workers=torch_workers,
        shuffle=shuffle,
        epochs=epochs,
    )
    loop_read_tensors(
        ds,
        num_workers=tens_workers,
        shuffle=shuffle,
        epochs=epochs,
    )
    loop_read_enterprise(
        ds,
        num_workers=tens_workers,
        shuffle=shuffle,
        epochs=epochs,
    )


# parse arguments
if __name__ == "__main__":
    run(test)
