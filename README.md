# Benchmarking DeepLake

This repository contains some simple Python functions to measure the
performance of the DeepLake dataloader in PyTorch and Tensorflow.

## Building and running the Docker container

### PyTorch

The example files can be easily executed within a Docker container:
```bash
$ docker build -t deeplake-test:pytorch -f Dockerfile.pytorch .
$ docker run --rm -it --cap-add=sys_admin deeplake-test:pytorch
```

### Tensorflow

Similarly:
```bash
$ docker build -t deeplake-test:pytorch -f Dockerfile.tensorflow .
$ docker run --rm -it --cap-add=sys_admin deeplake-test:tensorflow
```


## Preparing the dataset

### Locally in the filesystem

To generate a DeepLake dataset in `/data/deeplake-imagenet/train/`
from the ImageNet dataset stored in `/data/imagenet/train/`, go to the
`src/` directory and execute the following command:

```bash
python3 ingest_data.py /data/imagenet/train/ /data/deeplake-imagenet/train/
```

### Ingesting ImageNet dataset in MinIO

If you have a MinIO server listening at
`http://your_minio_server:9000/`, you can store the DeepLake dataset
there by executing the following command:

```bash
python3 ingest_data.py /data/imagenet/train/ s3://imagenet/train \
  --endpoint-url=http://your_minio_server:9000/ \
  --aws-access-key-id=user --aws-secret-access-key=password
```

## Performance testing with PyTorch

### Tests

We have implemented three simple stress tests that try to read as many
images as possible from the DeepLake dataset.

1. [`loop_read_raw`](src/test_pytorch.py:L39) Reads the raw dataset,
   without decoding the images, straight from the DeepLake dataset
   variable.
2. [`loop_read_pytorch`](src/test_pytorch.py:L45) Reads the raw
   dataset, without decoding the images, via the PyTorch Dataloader,
   with a configurable number of workers.
3. [`loop_read_tensors`](src/test_pytorch.py:L64) Reads the dataset
   via the PyTorch Dataloader, with a configurable number of workers,
   decodes the images and applies some standard preprocessing steps
   for the ImageNet dataset (i.e., resize, crop, flip and normalize).

### From the local filesystem

Run the tests using the following command:
```bash
python3 test_pytorch.py /data/deeplake-imagenet/train/ \
  --epochs=3 --shuffle=True --torch-workers=16 --tens-workers=32
```

The `--torch-workers` and `--tens-workers` parameters adjust the
number of workers for the `loop_read_pytorch` and `loop_read_tensors`
functions, respectively.

### From MinIO

Run the tests using the following command:
```bash
python3 test_pytorch.py s3://imagenet/train \
  --endpoint-url=http://your_minio_server:9000/ \
  --aws-access-key-id=user --aws-secret-access-key=password \
  --epochs=3 --shuffle=True --torch-workers=16 --tens-workers=32
```

## Performance testing with Tensorflow

### Tests

We have implemented two simple stress tests that try to read as many
images as possible from the DeepLake dataset.

1. [`loop_read_raw`](src/test_tensorflow.py:L39) Reads the raw dataset,
   without decoding the images, straight from the DeepLake dataset
   variable (same as the PyTorch test).
2. [`loop_read_tensorflow`](src/test_tensorflow.py:L45) Reads the raw
   dataset, without decoding the images, via the Tensorflow Dataloader.
3. [`loop_read_tensors`](src/test_tensorflow.py:L60) Reads the dataset
   via the Tensorflow Dataloader, decodes the images and applies some
   standard preprocessing steps for the ImageNet dataset (i.e., random
   crop, flip and normalize).

### From the local filesystem

Run the tests using the following command:
```bash
python3 test_tensorflow.py /data/deeplake-imagenet/train/ \
  --epochs=3 --shuffle=True 
```

### From MinIO

Run the tests using the following command:
```bash
python3 test_pytorch.py s3://imagenet/train \
  --endpoint-url=http://your_minio_server:9000/ \
  --aws-access-key-id=user --aws-secret-access-key=password \
  --epochs=3 --shuffle=True
```
