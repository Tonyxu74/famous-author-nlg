from utils.dataset import GenerateIterator
from utils.model import LSTM_Model, RNN_Model
from torch import optim
from torch import nn
import torch
import tqdm
import numpy as np
from myargs import args


def train(datapath, txtcode, model_name, continue_train=False):
    """
    Trains a RNN or LSTM model on the given datapath
    :param datapath: path to data
    :param txtcode: a code describing which author to train with
    :param model_name: the model name to train
    :param continue_train: whether the model will continue training from a certain point or not
    :return: no returns, saves a model for each epoch
    """

    print(f'Training on {txtcode} || Model is {model_name}')

    # create iterator
    train_iter = GenerateIterator(datapath, txtcode)

    # get model
    if model_name == 'RNN':
        model = RNN_Model(train_iter.dataset.vocab_len)
    elif model_name == 'LSTM':
        model = LSTM_Model(train_iter.dataset.vocab_len)
    else:
        raise Exception(f'Text code not supported! "{model_name}" given, "RNN", "LSTM" expected.')

    # get optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    lossfn = nn.CrossEntropyLoss()

    start_epoch = args.start_epoch

    # if training model from previous saved weights
    if continue_train:
        pretrained_dict = torch.load('./models/{}/{}_model_{}.pt'.format(
            txtcode,
            model_name,
            args.pretrain_epoch
        ))['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    for epoch in range(start_epoch, args.num_epochs):

        # progress bar to view progression of model
        pbar = tqdm.tqdm(train_iter)

        # used to check accuracy to gauge model progression
        losses_sum = 0
        n_total = 1
        pred_classes = []
        ground_truths = []

        for text, label in pbar:

            # remove from batch created by GenerateIterator
            text = text[0]
            label = label[0]

            # intialize states
            hidden_state = model.init_hidden(batch_size=args.batch_size)

            # move to GPU
            if torch.cuda.is_available():
                text = text.cuda()
                label = label.cuda()

            # new state is used if the batches are related sequentially, but they aren't
            prediction, new_state = model(text, hidden_state)

            # predictions to check for model progression
            pred_class = torch.softmax(prediction, dim=-1)
            pred_class = torch.argmax(pred_class, dim=-1)
            pred_classes.append(pred_class.cpu().data.numpy().tolist())
            ground_truths.append(label.cpu().data.numpy().tolist())

            # because of the format of the inputs, we need to shape this to compute loss properly
            prediction = prediction.view(-1, model.n_vocab)
            label = label.view(-1)

            loss = lossfn(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_sum += loss
            pbar.set_description('Epoch: {} || Loss: {:.5f} '.format(epoch, losses_sum / n_total))
            n_total += 1

        pred_classes = np.asarray(pred_classes)
        ground_truths = np.asarray(ground_truths)

        accuracy = np.mean((pred_classes == ground_truths)).astype(np.float)

        print('Epoch: {} || Accuracy: {} || Loss: {} '.format(
            epoch, accuracy, losses_sum / n_total
        ))

        # change modulo number to save every "number" epochs
        if epoch % 1 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './models/{}/{}_model_{}.pt'.format(txtcode, model_name, epoch))


if __name__ == '__main__':
    train('./data', 'homer', 'RNN', continue_train=False)


