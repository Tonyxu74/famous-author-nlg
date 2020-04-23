from utils.dataset import GenerateIterator
from utils.model import LSTM_Model, RNN_Model
import torch
import random


def generate(datapath, txtcode, model_name, model_epoch, seed=None, seedlen=5, output_len=100):
    """
    Generate a sequence of text based on a pretrained model and an author
    :param datapath: path to data
    :param txtcode: a code describing which author to train with
    :param model_name: name of the model to load
    :param model_epoch: the epoch of the loaded model
    :param seed: a seed to start generating words from
    :param seedlen: length of the seed in words
    :param output_len: length of the output text in words
    :return: none, prints out the generated sequence
    """

    # create iterator
    iter = GenerateIterator(datapath, txtcode)

    # create a random seed with some length, note given seed must be in integer form already
    if seed is None:
        rand_int = random.randint(0, len(iter.dataset.int_text) - seedlen)
        seed = iter.dataset.int_text[rand_int: rand_int + seedlen]

    # get model
    if model_name == 'RNN':
        model = RNN_Model(iter.dataset.vocab_len)
    elif model_name == 'LSTM':
        model = LSTM_Model(iter.dataset.vocab_len)
    else:
        raise Exception(f'Text code not supported! "{model_name}" given, "RNN", "LSTM" expected.')

    # load model weights
    pretrained_dict = torch.load('./models/{}/{}_model_{}.pt'.format(txtcode, model_name, model_epoch))['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # put model in GPU
    if torch.cuda.is_available():
        model = model.cuda()

    # no grad and model is only being evaluated
    with torch.no_grad():
        model = model.eval()

        # the batch size is one here for each item
        hidden_state = model.init_hidden(batch_size=1)
        output = []

        # run the model on the seed for every item
        for word in seed:

            # append the actual words of the seed to the output
            output.append(iter.dataset.int_to_word[word])

            if torch.cuda.is_available():
                word = torch.tensor([[word]]).long().cuda()
            else:
                word = torch.tensor([[word]]).long()

            # now that each word is run through the model separately, we will retain the hidden state
            prediction, hidden_state = model(word, hidden_state)

        # k is arbitrarily 5, obtain topk of word guesses
        values, indices = torch.topk(prediction, k=5)

        # take the first item in list twice, as this item added 2 dimensions for sequence and batch
        indices = indices.tolist()[0][0]
        chosen_int = random.choice(indices)
        output.append(iter.dataset.int_to_word[chosen_int])

        # now generate a sequence of words, similar to seed except the previously generated word becomes the new input
        for i in range(output_len - seedlen):
            if torch.cuda.is_available():
                word = torch.tensor([[chosen_int]]).long().cuda()
            else:
                word = torch.tensor([[chosen_int]]).long()

            prediction, hidden_state = model(word, hidden_state)

            values, indices = torch.topk(prediction, k=5)
            indices = indices.tolist()[0][0]
            chosen_int = random.choice(indices)

            output.append(iter.dataset.int_to_word[chosen_int])

    # print the output as a space-separated string
    print(' '.join(output))


if __name__ == "__main__":
    generate('./data', 'poe', model_name='LSTM', model_epoch=3)

"""
Poe RNN epoch 1:
spoke--the teeth glancing back, with an immense distance over it, we had some little difficulty of people to take the 
same condition of the main building and three circular Their face are exceedingly lofty, and thought the world which had 
been the victim for a long and very brilliant teeth. The other people of a better portion of the huge walls to its full 
length, although they left the machine over as a vessel and in each other the hole by the sea, or an instant afterward 
a man was more of a brilliant and ghastly fangs were upon their exertions,
"""