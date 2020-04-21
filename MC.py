import random
from utils.dataset import findFile


class MC_Model:

    def __init__(self, datapath, txtcode):
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

    print(MC.generate_text())

'''
=== Poe ===
threaded; the distance by the visitation of God.” Having inherited his estate, all went well with me the necessity of 
some Oriental investigations, to consult us, or if we duly consider the innocence, the artlessness, the enthusiasm, and 
the whole extent of a heart almost bursting from the machinery. Between the chief means of which I was quite upon an 
ottoman. “I see,” said he, “the night when I first, by mere accident, made his appearance, the converse of the burning 
stables of the creature bestowed upon me by the way with decision; pausing only for an introduction, I was.



=== Shakespeare ===
Yet, that the bastard boys of ice, and do ’t, and leave us. ARCITE. Till our scale turn the business you have seen more 
days than you; And what impossibility would slay In common worldly things 'tis called ungrateful With dull unwillingness 
to repay a debt to none- yet, more to say, “Do you in some measure satisfy her so That I ask thee what is lost Makes the 
remembrance of a warlike enterprise More venturous or desperate than this. Where is the Duke? DUKE. I am return'd your 
soldier; No more than mortall; So your helpe be, And honour’d.

Them acquainted with your approach; So, humbly take my life, Old fools are as much on her; Go to thy counsel! Then, even 
now, disguis'd? KING. Madam, I will, Or else what lets it but his steward; no meed but he could not put him quite beside 
his part, Or some of you, find out shames and praises be To those of old, I young. GREMIO. And so farewell, Signior 
Lucentio. BAPTISTA. Away with her, she is her question. PANDARUS. That’s Antenor. He has a gentleman loves a cup of 
wine. In this hard world, my ragged prison walls; And,.

=== Homer ===
It. This done, they brought from Arisbe. Hippothous led the people should cross your will with them were also Meriones, 
Aphareus and Deipyrus, and the Trojans, as from some city that is being brought up in presence of him whom noble 
Achilles has cut off the body; but all undaunted answered, "Archer, you who reign in heaven- devise evil for the 
whizzing of the son of Menoetius and his heart was gladdened at the ships." On this Ulysses went at once spoke to the 
counsel of great rain or hail or snowflakes that fly from off his armour out and.

'''