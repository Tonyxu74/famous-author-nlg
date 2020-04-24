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
        values, indices = torch.topk(prediction, k=10)

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

            values, indices = torch.topk(prediction, k=10)
            indices = indices.tolist()[0][0]
            chosen_int = random.choice(indices)

            output.append(iter.dataset.int_to_word[chosen_int])

    # print the output as a space-separated string
    print(' '.join(output))


if __name__ == "__main__":
    generate('./data', 'poe', model_name='RNN', model_epoch=4)

"""
Poe RNN epoch 4:
just mentioned . To say the question of Mrs would appear but either on our friends . My immediate regard or some common 
limits could found its extreme horror . The question were occupied by Peters of a third . These few and we perceive on 
no particular portion of her body alone without getting on this occasion, and so much obvious to the earth a portion and 
with the left of no farther than ourself for the than thirty hundred feet are the black building which set the rope as 
far open the table which lay before the loss to




Poe LSTM epoch 3:
teares shall fill your eye in the street just one above those thousand hours after nine inches in its surface the 
surface of the water, we saw a few small and fifty persons of a large gale had been thrown into an opposite window that 
our head would the same appearance were about so far a small southern water, of the two men very large as far above his 
own southern hand, is not in any other respect a thousand more fifty miles to be the ordinary appearance which has been 
so easily observed that, as if we have been in

evidence of extensive general reading. In you your mind this, after which he thought he took it until just afterward 
until we came in a single search about one hour, and the third one, in their head, and, to find ourselves thus in a 
strong sound than our usual In their position as most obvious than these purposes appeared to me as I thought proper 
enough would pass down all upon me as to say, from them, I saw an air rather a foot in less utterly less vivid power for 
their absolute general interest apparent than my seven and seventy

"""