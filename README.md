# CS-GY 6953 / ECE-GY 7123 Final Project

## Detecting Offensive Language based on Graph Attention Networks and Fusion Features

Transformer-based models have come to dominate natural language processing tasks including difficult problems in computational linguistics such as semantic and sentiment analysis. One application for these models is automated content moderation for social media platforms where they can be deployed to flag and remove abusive or harmful content. In the context of offensive language, recent research demonstrates that the performance of such models can be improved by incorporating information about the communities where that content is produced. For our project, we experiment with a novel method for incorporating community structure features and text features to classify offensive language via attention mechanisms and positional encoding. In comparison to existing published solutions, we incorporate a "corrected" procedure for computing attention on graph-structured data to produce our user embeddings and a better performing pre-trained language model for our text embeddings. Our model achieves a mean F1 score of 89.36\% outperforming our baseline model across multiple iterations.

You can find notebooks that show how to train, test and analyze experiments using this repository under `examples/`. The
notebook `model_training_colab.ipynb` can be used to reproduce our final results and the
notebook `plots.ipynb` can be used to generate the plots.

The scripts have been built off of the code from [Miao 2022](https://github.com/mzx4936/GF-OLD)


## Instructions

The scripts can be run from the command line. The minimum arguments for each can be seen below. Whether you are running the script locally or in a hosted Jupyter notebook, all you need to do is clone the repository and run the script:

```bash
git clone https://github.com/guptaviha/GF-OLD.git
cd cautious-fiesta
python train_joint.py \
  -bs=32 \
  -lr_other=1e-5 \
  -lr_gat=1e-2 \
  -ep=20 \
  -dr=0.5 \
  -ad=0.1 \
  -hs=768 \
  --model=jointv2_twitter_roberta \
  --clip \
  --cuda=1 \
  --num-trials=1 \
  --log-path=/content/drive/MyDrive/dl-project/logs/final
```

The various different models available to evaluate are: ```gat```, ```gatv2```,```bert```, ```roberta```, ```twitter_roberta```, ```joint```, ```joint_roberta```, ```joint_twitter_roberta```, ```jointv2```,```jointv2_roberta```, ```jointv2_twitter_roberta```

## Outputs

The scripts log information to standard output.

`train_joint.py` outputs a `.json` file containing the evaluation metrics `{trial_num: {train_loss, test_loss, train_acc, test_acc, train_recall, test_recall, train_precision, test_precision, train_f1, test_f1, best_train_f1, best_test_f1}}`
