#!/bin/bash
rsync -avz --delete --include='*/' --include='events.out.tfevents.*' --exclude='*' karolina:/mnt/proj1/eu-25-92/occrae_log/ ./tb_logs_karolina/
tensorboard --logdir ./tb_logs_karolina/ --port 6008