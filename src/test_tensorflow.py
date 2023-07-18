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
from tqdm import tqdm
import tensorflow as tf


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


def loop_read_tensorflow(
    ds,
    epochs=3,
    shuffle=False,
):
    bs = 512
    sz = len(ds) // bs
    dt = ds.tensorflow(tobytes=True)
    if shuffle:
        dt = dt.shuffle(8192, reshuffle_each_iteration=True)
    dt = dt.batch(bs).prefetch(tf.data.AUTOTUNE)

    for _ in range(epochs):
        cow = 0
        for x in tqdm(dt, total=sz):
            cow += 1
            if cow == sz:
                break


def loop_read_tensors(
    ds,
    epochs=3,
    shuffle=False,
):
    bs = 512
    sz = len(ds) // bs
    dt = ds.tensorflow(tobytes=False)
    dt = dt.map(lambda x: x["images"])
    dt = dt.map(lambda x: tf.image.resize(x, (300, 300)))
    dt = dt.map(lambda x: tf.image.random_crop(x, (224, 224, 3)))
    dt = dt.map(lambda x: tf.image.random_flip_left_right(x))
    dt = dt.map(lambda x: tf.image.per_image_standardization(x))
    if shuffle:
        dt = dt.shuffle(8192, reshuffle_each_iteration=True)
    dt = dt.batch(bs).prefetch(tf.data.AUTOTUNE)
    for _ in range(epochs):
        cow = 0
        for x in tqdm(dt, total=sz):
            cow += 1
            if cow == sz:
                break


def test(
    src,
    *,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    endpoint_url=None,
    epochs=3,
    shuffle=False,
):
    ds = open_dataset(
        src=src,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )
    loop_read_raw(ds, epochs=epochs)
    loop_read_tensorflow(
        ds,
        shuffle=shuffle,
        epochs=epochs,
    )
    loop_read_tensors(
        ds,
        shuffle=shuffle,
        epochs=epochs,
    )


# parse arguments
if __name__ == "__main__":
    run(test)
