from torch.utils import data
import os
from collections import Counter
from myargs import args
import numpy as np
import torch


def findFile(root_dir, contains):
    """
    Finds file with given root directory containing keyword "contains"
    :param root_dir: root directory to search in
    :param contains: the keyword that should be contained
    :return: a list of the file paths of found files
    """

    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if contains in file:
                all_files.append(os.path.join(path, file))

    return all_files


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, datapath, txtcode):
        """
        Initializes dataset for given author and datapath
        :param datapath: path to data
        :param txtcode: code that describes the author to load data from
        """

        # find files for the given textcode, if textcode isn't supported, throw error
        if txtcode not in ['poe', 'homer', 'shakespeare']:
            raise Exception(f'Text code not supported! "{txtcode}" given, "poe", "shakespeare", "homer" expected.')

        filelist = findFile(datapath, txtcode)

        # open all files for given textcode
        txt = ""
        for filepath in sorted(filelist):
            with open(filepath, 'r', encoding='utf8') as txtfile:
                txt += txtfile.read()

        punc_text = []

        txt = txt.split()
        for word in txt:
            if "." in word:
                punc_text.append(word.replace('.', ''))
                punc_text.append('.')
            else:
                punc_text.append(word)

        # give the vocabulary of the text from most common to least common word
        unique_words = Counter(punc_text)
        vocab = sorted(unique_words, key=unique_words.get, reverse=True)
        self.vocab_len = len(vocab)
        self.int_to_word = {key: word for (key, word) in enumerate(vocab)}
        self.word_to_int = {word: key for (key, word) in enumerate(vocab)}

        # convert entire text to integers
        int_text = [self.word_to_int[word] for word in punc_text]

        # total length of the text except for an incomplete batch at the end, cutoff text at this
        total_len = len(int_text) - len(int_text) % args.batch_size
        self.int_text = int_text[:total_len]

        # reshape into batches, length of dataset is so that an entire sequence and its label can be generated
        self.text_input = np.reshape(self.int_text, (args.batch_size, -1))
        self.len_dataset = self.text_input.shape[1] - args.seq_len - 1

    def __len__(self):
        """
        Denotes the total number of samples
        :return: length of the dataset
        """

        return self.len_dataset

    def __getitem__(self, index):
        """
        Generates one sample of data
        :param index: index of the data in the datalist
        :return: returns the data in longtensor format
        """

        data = np.asarray([self.text_input[:, index + i] for i in range(args.seq_len)])
        label = np.asarray([self.text_input[:, index + i + 1] for i in range(args.seq_len)])

        data = torch.from_numpy(data).long()
        label = torch.from_numpy(label).long()

        return data, label


def GenerateIterator(datapath, txtcode, shuffle=True):
    """
    Generates a batch iterator for data
    :param datapath: path to data
    :param txtcode: code that describes the author to load data from
    :param shuffle: whether to shuffle the batches around or not
    :return: a iterator combining the data into batches
    """

    params = {
        'batch_size': 1,  # batch size must be 1, as data is already separated into batches
        'shuffle': shuffle,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(datapath=datapath, txtcode=txtcode), **params)
