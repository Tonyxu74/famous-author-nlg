import random
from utils.dataset import findFile


class MC_Model:
    """A simple bigram Markov Chain model"""

    def __init__(self, datapath, txtcode):
        """
        Initializes the MC model
        :param datapath: path to the data
        :param txtcode: code describing which author the model will operate on
        """

        self.model = None

        # find files for the given textcode, if textcode isn't supported, throw error
        if txtcode not in ['poe', 'homer', 'shakespeare']:
            raise Exception(f'Text code not supported! "{txtcode}" given, "poe", "shakespeare", "homer" expected.')
        filelist = findFile(datapath, txtcode)

        # open all files for given textcode
        txt = ""
        for filepath in sorted(filelist):
            with open(filepath, 'r', encoding='utf8') as txtfile:
                txt += txtfile.read()

        self.txt = txt.split()

    def train(self):
        """
        Trains the MC model
        :return: None
        """

        model = {}

        # since this will use bigrams, the total number of bigrams created will be 2 less than total length of the text
        for i in range(len(self.txt) - 2):
            bigram = (self.txt[i], self.txt[i + 1])
            next_word = self.txt[i + 2]

            if bigram in model:
                model[bigram].append(next_word)

            else:
                model[bigram] = [next_word]

        final_gram = (self.txt[-2], self.txt[-1])

        if final_gram in model:
            model[final_gram].append(None)

        else:
            model[final_gram] = [None]

        self.model = model

    def generate_text(self, seed=None, length=100):
        """
        Generates a sequence of text based on the trained MC model
        :param seed: a seed to use to begin generating the sequence
        :param length: length of the generated sequence
        :return: generated sequence as a string
        """

        # if no seed given, generate a random one from all the bigrams
        if seed is None:
            seed = random.choice(list(self.model.keys()))

        output = list(seed)

        # create the output by chaining bigrams together
        for i in range(2, length):
            curr_bigram = (output[-2], output[-1])

            if curr_bigram in self.model:
                transition = self.model[curr_bigram]
                choice = random.choice(transition)

                # if it was the final item in the text, break
                if choice is None:
                    break

                output.append(choice)

            else:
                break

        # append a period if there isn't one
        if output[-1][-1] not in ['?', '.', '!']:
            output[-1] += '.'

        # capitalize the first word
        output[0] = output[0].capitalize()

        # create a string
        gen_sentence = ' '.join(output)

        return gen_sentence


if __name__ == '__main__':
    MC = MC_Model('./data', 'homer')
    MC.train()

    print(MC.generate_text(seed=('He', 'poised')))
