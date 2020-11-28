"""
File to compress model using pruning or quantization techniques
"""
import logging
import argparse
import finetune
import preprocess

def get_args():

    parser = argparse.ArgumentParser(description='CheXNet Model compression for low edged devices')

    parser.add_argument('--model', default='nih', choices=['kaggle', 'nih', 'pc', 'chex'],
                        help='model name for compression')
    parser.add_argument('--compress-type', default='prune', choices=['prune', 'quantize'],
                        help='type of compression technique to use')
    parser.add_argument('--compress-method', default='dynamic', choices=['static', 'dynamic'],
                        help='type of compression method to use')
    parser.add_argument('--pretrained', default='true', choices=['true', 'false'],
                        help='if to use a pretrained model or train a chexnet model')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight-decay', type=int, default=0.00001, metavar='N',
                        help='weight decay for train (default: 0.00001)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between logging training status')
    parser.add_argument('--log', type=str, default='log.txt',
                        help='log file name')
    parser.add_argument('--sensitivity', type=float, default=2,
                        help="sensitivity value that is multiplied to layer's std in order to get threshold value")


    args = parser.parse_args()

    return args


def main():
    args = get_args()
    if args.pretrained:

        if args.compress_method=='dynamic':
            logging.warning("dynamic compression unavailable for pretrained model using static instead")

        finetune.finetune_pretrained_model(args.model, args.compress_type, args.batch_size, args.log_interval, 0.3)

    else:

        preprocess.preprocess_model(args.compress_type, args.batch_size, args.seed, args.lr, args.weight_decay, args.epochs, 0.2, args.log_interval)


if __name__ == "__main__":
    main()