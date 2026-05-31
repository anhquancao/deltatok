#!/bin/bash
rsync -avz --delete --include='*/' --include='events.out.tfevents.*' --exclude='*' bsc:/gpfs/scratch/ehpc558/quan/occrae_output/ ./tb_logs/
tensorboard --logdir ./tb_logs/ --port 6007