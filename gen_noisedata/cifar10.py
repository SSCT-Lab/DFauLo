import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=list, default=['alllabel','ranlabel'], help='Number of epochs to train.')

args = parser.parse_args()
print(args.sparse)
print(args.seed)
print(args.epochs)
