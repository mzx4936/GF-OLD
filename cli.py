import argparse


def get_args():
    parser = argparse.ArgumentParser(description='TRADE Multi-Domain DST')

    # Training hyper-parameters
    parser.add_argument('-ntrials', '--num-trials', help='Number of trials', type=int, required=False, default=1)
    parser.add_argument('-bs', '--batch-size', help='Batch size', type=int, required=True)
    parser.add_argument('-lr_other', '--learning-rate-other', help='Learning rate of others', type=float, required=True,
                        default=5e-5)
    parser.add_argument('-lr_gat', '--learning-rate-gat', help='Learning rate of GAT', type=float, required=True,
                        default=1e-2)  # only gat 1e-5
    parser.add_argument('-wd', '--weight-decay', help='Weight decay', type=float, required=False, default=0)
    parser.add_argument('-ep', '--epochs', help='Number of epochs', type=int, required=True)
    parser.add_argument('-trun', '--truncate', help='Truncate the sequence length to', type=int, required=False,
                        default=512)
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    parser.add_argument('-mo', '--model', help='Which model to use', type=str, required=True)
    parser.add_argument('-ms', '--model-size', help='Which size of model to use', type=str, required=False,
                        default='base')
    parser.add_argument('-cl', '--clip', help='Use clip to gradients', action='store_true')
    parser.add_argument('-se', '--seed', help='Random seed', type=int, required=False, default=19951126)
    parser.add_argument('-tr', '--test-ratio', help='Ratio of test set', type=float, required=False,
                        default=0.3)
    parser.add_argument('-pa', '--patience', help='Patience of early stopping', type=int, required=False,
                        default=5)

    parser.add_argument('--ckpt', type=str, required=False, default='')

    parser.add_argument('-ad', '--attention-dropout', help='transformer attention dropout', type=float, required=False,
                        default=0.1)
    parser.add_argument('-hd', '--hidden-dropout', help='transformer hidden dropout', type=float, required=False,
                        default=0.1)
    parser.add_argument('-dr', '--dropout', help='dropout', type=float, required=False, default=0.1)
    parser.add_argument('-hs', '--hidden-size', help='hidden vector size', type=int, required=False, default=300)

    parser.add_argument('-fm', '--feat-model', help='Which feature to use in social graph', type=str, required=False,
                        default='soft')
    parser.add_argument('-fi', '--feat-init', help='Which feature initialization to use in social graph', type=str,
                        required=False,
                        default='non_off')
    # Data path
    parser.add_argument('-tp', '--tweet-path', help='Which tweet path to use', type=str, required=False,
                        default='data/tweets.csv')
    parser.add_argument('-up', '--user-path', help='Which user path to use', type=str, required=False,
                        default='data/users.csv')
    parser.add_argument('-rp', '--relationship-path', help='Which relationship path to use', type=str, required=False,
                        default='data/relationships.csv')
    parser.add_argument('-log', '--log-path', help='Path for logging metrics and saving models', type=str,
                        required=False, default='')

    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = get_args()
