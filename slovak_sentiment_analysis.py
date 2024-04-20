from unidecode import unidecode  # Converts punctuation e.g. Á => A, č => c, ô =>o
from fuzzywuzzy import fuzz  # Fuzzywuzzy uses Levenshtein Distance to calculate similarity of words
from collections import defaultdict  # This is just convenient datastructure for working with empty dictionary


# function to set defaultvalue for defaultdict, its input parameter must be callable
def def_value():
    return []


# pip3 install python-Levenshtein
# pip3 install fuzzywuzzy
# pip3 install unidecode
# pip install -q transformers

# You need to have PyTorch or Tensorflow installed:
# For tensorflow on linux, use this
# pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Tutorial for other systems for PyTorch:
# https://www.gcptutorials.com/post/how-to-install-pytorch-with-pip

# After first successfull anal run of this program, pretrained english model from huggingface is downloaded and can be used

# pip3 install emoji==0.6.0 -> but this is not necessary, no emojis are translated


# WHEN USING THE MODEL finiteautomata/bertweet-base-sentiment-analysis
# NEED TO CITE THIS:
# https://arxiv.org/abs/2106.09462

# Using pipeline class to make predictions from models available in the Hub in an easy way
from transformers import pipeline

# Importing required modules
import tkinter as tk
import tkinter.scrolledtext as st
import tkinter.font as font
import tkinter.messagebox
from tkinter import filedialog as fd, END
from tkinter import Radiobutton
from tkinter import *
import pathlib

# Creating tkinter window
root = tk.Tk()
root.title("Sentiment analysis")
# Title Label
tk.Label(root,
         text="Sentiment analysis of sentences in Slovak language",
         font=("Times New Roman", 20),
         background='darkmagenta',
         foreground="white").grid(column=0,
                                  row=0)

text_area = st.ScrolledText(root,
                            wrap="word",
                            width=100,
                            height=55,
                            font=("Times New Roman",
                                  15),
                            background="whitesmoke")
text_area.tag_config('original', font=("Times New Roman", 20, "bold"))
bold_italic = font.Font(family="Times New Roman", weight="bold", slant="italic", size=15)
text_area.tag_config('raw', font=bold_italic)
text_area.tag_config('translated', font=("Arial", 10))
text_area.tag_config('RED', font=("Times New Roman", 15, "bold"), background="lightcoral", foreground="black")
text_area.tag_config('GREEN', font=("Times New Roman", 15, "bold"), background="lime", foreground="black")
text_area.tag_config('YELLOW', font=("Times New Roman", 15, "bold"), background="moccasin", foreground="black")

text_area.grid(column=0, pady=10, padx=10)

# Creating canvas for "is busy" information - might not be needed
canvas = tk.Canvas(root, width=170, height=50)
canvas.grid(column=1, row=0, pady=10, padx=10)

var = tk.IntVar()
Radiobutton(root, text="STRICT - binary model", variable=var, value=0).place(x=1040, y=380)
Radiobutton(root, text="SOFT - ternary model", variable=var, value=1).place(x=1040, y=400)

# input for file.txt"
entry_text = tk.Entry()
entry_text.place(x=1040, y=450)


class Database:
    def __init__(self):

        '''Holds a refference for each slovak word present in a database to its
        english translation, there might be multiple possible translations...
        reference holds information about part of speach too: (a, n, v, r)
        -> see: https://korpus.juls.savba.sk/WordNet_en.html
        the datastructure looks like this:

        {'koruna':[([royal_crown], 'n'), ('[coin, cash, change], 'n'), ('[tree_crown, tree_top], 'n')}]
        default value == []'''
        self.dictionary = defaultdict(def_value)
        self.slovak_stop_words = []

    '''Loads the database for specific parsing of sk-wn-2013-01-23.txt
    in case of different database, this function might need some adjustment...
    
    line.split("\t")[2] - stands for all Slovak words in synset, separated by ;
    line.split("\t")[1] - stands for part of speach of synset (a, n, v, r)
    line.split(" ␞")[1].split("@")[0].split(' ')[4:]
    -> this splits the input synset line to part with translation from 4th position, where it starts
    from there the english words can be loaded if they are not integers until any
    separator is found ['+','','!', '$', '*', '>', '~'] which stands for explanation, useless for the algorithm
    
    For some reason not all the words in original database have semantically correct
    equivalent because of this inconsistance, search sometimes results to '-' 
    this has to be removed in order not to create a mess in the dictionary.
    Some synsets in SlovakWordDB are present multiple times which contributes to 
    lower quality of the input database, sk-wn-2013-01-23.txt.
    This problem is solved by keeping only unique names and list of translations in dictionary.
    Same name but different translation for words with different meaning are correctly
    present as well.
    
    unidecode.lower ensures similarity
    
    '''

    def load_database(self, database_name="sk-wn-2013-01-23.txt"):
        f = open(database_name, "r", encoding="utf8")
        count = 0
        for line in f.readlines():
            part_of_speech = line.split("\t")[1]
            for new_word in line.split("\t")[2].split(';'):
                if new_word != '-' and (new_word not in self.dictionary.keys()):
                    transaltions = []
                    for part in line.split(" ␞")[1].split("@")[0].split(' ')[4:]:
                        if part in ['+', '', '!', '$', '*', '>', '~', '|', '\\']:
                            break
                        try:
                            int(part)
                        except Exception:
                            # transaltions.append(part.replace("_"," "))
                            transaltions.append(part)
                    self.dictionary[unidecode(new_word).lower()].append((transaltions, part_of_speech))
                    # print(unidecode(new_word).lower())
            count += 1

    def load_stop_words(self):
        # for removal of slovak stopwords // source https://countwordsfree.com/stopwords/slovak
        f_stop_w = open("slovak_stop_words.txt", "r")
        for stop_w in f_stop_w.readlines():
            self.slovak_stop_words.append(unidecode(stop_w.strip()).lower())


"""This loads the database"""
try:

    database = Database()
    database.load_database()
    database.load_stop_words()

except Exception as err:
    tkinter.messagebox.showerror(title=str(err), message="Nepodarilo sa načítať databázu"
                                                         ", reštartujte program a uistite sa"
                                                         " že existuje databáza v tejto zložke"
                                                         " s názvom {} rovnako by sa tu mal"
                                                         " nachádzať súbor {}".format("sk-wn-2013-01-23.txt", "slovak_stop_words.txt"))


class Word:
    def __init__(self, name):
        self.name = name
        # homonymum
        self.chose_slovak_meaning = 0
        # vatiation for translation for the homonymum
        self.chose_english_translation = 0


'''Sentence object holds chronolgical representation of each word that is present in raw_sentence
sentence is a textual input from user or a file which belongs to this sentence. The input is separated
by words in form of a list and punctuations are removed.

Not all words from raw_sentence are necessary mapped to a object Word in representations.
This is because some raw_words are either not found in database or are ommited by choice (stop-words)'''


class Sentence:
    def __init__(self, raw_sentence):
        self.raw_sentence = raw_sentence
        self.parsed_sentence = self.parsed_sentence(raw_sentence)
        '''Datastructure looks like this:
        [['afto', [(Word<"auto">,0.8),(Word<"afro">, 0.8]),(Word.<"autor">,0.6)], 0]...[]...[]]
        list contains words in a sentence in chronological order ancapsulated in another list
        this contains the raw transcript of word, list of possible meanings with Levenshtein Distance rate
        and index of meaning that is applicable for the sentence. This can be changed by user.
        '''
        self.representation = []
        # sentiment value of given sentence
        self.sentiment_value = {}
        self.translated_words = []

        '''Returns a list containing all the words of raw sentence
        without punctuations unidecoded to lowercase'''

    def parsed_sentence(self, raw_sentence):
        punctuations = ";',.!<>?()[]{}:/&%@~`°_-*"
        sentence = unidecode(raw_sentence).lower()
        for x in self.raw_sentence:
            if x in punctuations:
                sentence = sentence.replace(x, "")
        return sentence.split(" ")

    '''Function evaluates each word in parsed_sentence and attaches list of similar words to it
        it defaultly picks the first one (0) index as the chosen transcription
        output is list of evaluations'''

    def analyze_sentence(self):
        for word in self.parsed_sentence:
            new_word = [word, [], 0]
            if word not in database.slovak_stop_words:
                keys = database.dictionary.keys()
                # this is representation of each word found in raw sentence
                # thogether with its ratio and key from dictionary that it maps
                # to with that ratio
                for key in keys:
                    ratio = fuzz.ratio(word, key)
                    if ratio > 70:  # only if ratio is bigger than 70, it is relevant
                        new_word[1].append((Word(key), ratio))
                # sorts the transcripts for each word by their ratios
                new_word[1].sort(key=lambda x: x[1], reverse=True)
            self.representation.append(new_word)

            '''Function transaltes words that were mapped with ratio >70 and returns them
            in a form of list of [slovak_word, translation], where word is original word from sentence'''

    def translation_mapping(self):
        translation = []
        for word in self.representation:
            if (word[1]):  # list of translations is not empty
                # takes relevant translation based on user decision
                # default is 0th index... found on position 2 in representation
                the_word = word[1][word[2]][0]
                translation.append([the_word.name, database.dictionary[the_word.name][the_word.chose_slovak_meaning][0][the_word.chose_english_translation]])
        self.translated_words = translation


class Text:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.representation = self.split_to_sentences(raw_text)
        # sentiment value of the full text
        self.sentiment_value = {}

    """Splits raw_text into raw_sentences. Avoids multiple dots...
    by ommiting empty sentence returns list of objects Sentence(raw_sentence)"""

    def split_to_sentences(self, raw_text):
        # filter removes empty sentences fast and generator creates Sentence object out of remaining
        return [Sentence(x) for x in list((filter(None, raw_text.split("."))))]

    """analyzes the sentiment of the whole text, sentence by sentence
    this can last a little especially if there are more sentences
    After successful run, it should set all the necessary attributes
    correctly"""

    def sentiment_analysis(self):
        # model_type = ["sentiment-analysis","finiteautomata/bertweet-base-sentiment-analysis"]
        # canvas.create_text(60, 20, text="Čakajte prosím...", tags="wait")
        sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis") if var.get() else pipeline("sentiment-analysis")
        full_data = []
        for sentence in self.representation:
            sentence.analyze_sentence()
            sentence.translation_mapping()
            data = []
            # selects just english words in translation list
            for words in sentence.translated_words:
                data.append(words[1])
            full_data += data
            sentence.sentiment_value = sentiment_pipeline(" ".join(data))[0]  # analysis of individual sentence
        self.sentiment_value = sentiment_pipeline(" ".join(full_data))[0]  # analysis of ful text
        # canvas.delete("wait")

    """Prints the result of analysis to the console for convenience"""

    def print_result(self):
        print("original: '{}''".format(self.raw_text))
        for sentence in self.representation:
            print(sentence.raw_sentence)
            print(sentence.translated_words)
            print(sentence.sentiment_value)
            print()

    def get_color(self, sentiment_value):
        return "RED" if sentiment_value["label"] == 'NEG' \
                        or sentiment_value["label"] == 'NEGATIVE' \
            else ('GREEN' if sentiment_value["label"] == 'POS' \
                             or sentiment_value["label"] == 'POSITIVE' else "YELLOW")

    """Outputs the graphical result to sroll GUI
    custom tags are used for different formatting"""

    def graphical_result(self):

        text_area.configure(state='normal')
        #text_area.insert(tk.INSERT, "'{}'\n".format(self.raw_text), "original")
        #text_area.insert(tk.INSERT, "{}\n".format(self.sentiment_value), self.get_color(self.sentiment_value))
        text_area.insert(tk.INSERT, "\n")
        for sentence in self.representation:
            text_area.insert(tk.INSERT, "{} ".format(sentence.raw_sentence), "raw")
            text_area.insert(tk.INSERT, "{}\n".format(sentence.translated_words), "translated")
            text_area.insert(tk.INSERT, "{}\n".format(sentence.sentiment_value), self.get_color(sentence.sentiment_value))
            text_area.insert(tk.INSERT, "\n")
        text_area.configure(state='disabled')


"""Function to load .txt file from computer"""


def read_text():
    try:
        filename = fd.askopenfilename(title='Vyber súbor',
                                      initialdir=pathlib.Path().resolve(),
                                      filetypes=(('text files', '*.txt'), ('All files', '*.*')))
        file = open(filename, "r", encoding="utf8")
        reset_text_area()
    except Exception as err:
        tkinter.messagebox.showerror(title=err)
    else:
        new_text = Text(file.read())
        new_text.sentiment_analysis()
        new_text.graphical_result()


"""Function to input text from user"""


def reset_text_area():
    text_area.configure(state='normal')
    text_area.delete('1.0', tk.END)
    text_area.configure(state='disabled')


def input_own():
    reset_text_area()
    new_text = Text(entry_text.get())
    new_text.sentiment_analysis()
    new_text.graphical_result()


tk.Button(text="Analyzuj text so súboru", bg="khaki", font="Arial 11", command=lambda: read_text(), ).place(x=1040, y=330)
tk.Button(text="Analyzuj vlastný text", bg="papayawhip", font="Arial 11", command=lambda: input_own(), ).place(x=1040, y=480)

root.mainloop()
