import torch
from torch import nn
from myargs import args


class RNN_Model(nn.Module):
    def __init__(self, n_vocab):
        super(RNN_Model, self).__init__()

        self.n_vocab = n_vocab
        self.embedding_size = args.embedding_size
        self.RNN_size = args.RNN_size

        self.embedding = nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.embedding_size)
        self.RNN = nn.RNN(
            input_size=self.embedding_size,
            hidden_size=self.RNN_size,
            num_layers=args.RNN_layers,
            dropout=args.dropout
        )
        self.dense = nn.Linear(in_features=self.RNN_size, out_features=self.n_vocab)

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return torch.zeros(args.RNN_layers, batch_size, self.RNN_size).cuda()
        else:
            return torch.zeros(args.RNN_layers, batch_size, self.RNN_size)

    def forward(self, x, state):
        embedded_vals = self.embedding(x)
        rnn_out, new_state = self.RNN(embedded_vals, state)
        vocab_logits = self.dense(rnn_out)

        return vocab_logits, new_state


class LSTM_Model(nn.Module):
    def __init__(self, n_vocab):
        super(LSTM_Model, self).__init__()

        self.n_vocab = n_vocab
        self.embedding_size = args.embedding_size
        self.LSTM_size = args.LSTM_size

        self.embedding = nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.embedding_size)
        self.LSTM = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.LSTM_size,
            num_layers=args.LSTM_layers,
            dropout=args.dropout
        )
        self.dense = nn.Linear(in_features=self.LSTM_size, out_features=self.n_vocab)

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(args.LSTM_layers, batch_size, self.LSTM_size).cuda(),
               torch.zeros(args.LSTM_layers, batch_size, self.LSTM_size).cuda())
        else:
            return (torch.zeros(args.LSTM_layers, batch_size, self.LSTM_size),
                   torch.zeros(args.LSTM_layers, batch_size, self.LSTM_size))

    def forward(self, x, state):
        embedded_vals = self.embedding(x)
        lstm_out, new_state = self.LSTM(embedded_vals, state)
        vocab_logits = self.dense(lstm_out)

        return vocab_logits, new_state
