# torchrun --nproc_per_node=4 test_pytorch.py hub://activeloop/imagenet-train \
#         --epochs=3 --shuffle=False --distributed=True

python3 test_pytorch.py hub://activeloop/imagenet-train \
        --epochs=3 --shuffle=False --distributed=False
