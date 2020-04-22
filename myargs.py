import argparse

parser = argparse.ArgumentParser()

# ======= Model parameters =======

parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='momentum for sgd, beta1 for adam')

parser.add_argument('--num_epochs', default=100, type=int,
                    help='epochs to train for')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='epoch to start training. useful if continue from a checkpoint')

parser.add_argument('--batch_size', default=16, type=int,
                    help='input batch size')
parser.add_argument('--seq_len', default=32, type=int,
                    help='length of the sequences for an individual data point')

parser.add_argument('--embedding_size', default=300, type=int,
                    help='size of word embeddings')
parser.add_argument('--RNN_size', default=256, type=int,
                    help='size of output of RNN model')
parser.add_argument('--LSTM_size', default=256, type=int,
                    help='size of output of LSTM model')

parser.add_argument('--RNN_layers', default=2, type=int,
                    help='number of layers of RNN model')
parser.add_argument('--LSTM_layers', default=2, type=int,
                    help='number of layers of LSTM model')

parser.add_argument('--dropout', default=0.1, type=int,
                    help='dropout layer')
args = parser.parse_args()


