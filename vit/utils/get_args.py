import argparse

def GetArgs():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--data_path', default='...', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'mnist', 'imagenet', 'tiny-imagenet'])
    parser.add_argument('--model', default='deit-small', help='model architecture', choices=['deit-tiny', 'deit-small'])
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=1024, type=int)
    parser.add_argument('--seed', default=0, type=int, help='dist sampler')
    parser.add_argument('--opt', type=str, default='adamw', help='choose from (sgd, adagrad)')
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--rho', default=0.01, type=float)
    parser.add_argument('--s', type=float, default=0.5, help="sparsity for bandit")
    parser.add_argument('--num_samples', type=int, default=1024, help="Number of samples to compute fisher information. Only for `ssam-f`.")
    parser.add_argument('--update_freq', type=int, default=5, help="Update frequency (epoch) of sparse SAM.")

    return parser.parse_args()
