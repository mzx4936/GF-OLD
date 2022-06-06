# GF-OLD
## Introduction
Detecting Oﬀensive Language based on Graph Attention Networks and Fusion Features
## Start
python train_joint.py -bs=64 -lr_other=5e-5 -lr_gat=1e-2 -ep=20  -dr=0.5 -ad=0.1 -hs=768 --model=joint --clip --cuda=0
