import numpy as np
import re
from tempfile import TemporaryFile



f = open('C:\\Users\\QRG_02\\Desktop\\language_detector\\English_3.txt', 'r', encoding='UTF-8')
pdf_contents_English = f.read().split(' ')

g = open('C:\\Users\\QRG_02\\Desktop\\language_detector\\French_3.txt', 'r', encoding='UTF-8')
pdf_contents_French = g.read().split(' ')

h = open('C:\\Users\\QRG_02\\Desktop\\language_detector\\Spanish_3.txt', 'r', encoding='UTF-8')
pdf_contents_Spanish = h.read().split(' ')

x = list()
y = list()
steps = 20
word_vector = np.array(0)
alphabet = ["a","á","à","â","b","c","ç","d","e","é","è","ê","ë","f","g","h","i","í","î","ï","j","k","l","m","n","ñ","o","ó","ô","p","q","r","s","t","u","ù","û","ü","v","w","x","y","z"]
flattened_length_with_2_tags = (steps * len(alphabet)) + 2


def get_word_contents ( word_list_word, which_file, English_or_zero ):
    "This gets quadruplets from words and puts them in a full_vector"
    global full_vector
    global full_vector_stack
    global word_vector
    s = 0
    e = 4
    loops = 0
    word = which_file[word_list_word]
    word = re.sub('[^a-zA-Záàâçéèêëîíïñóôùûü]', '', word, count=0, flags=0)
    word = word.lower()
    full_vector[0] = English_or_zero
    word_length = len(word)
    found = np.where(word_vector == word)
    found_place = found[0]
    if not len(found_place):
        word_vector = np.append(word_vector, word, axis=None)
        if (len(word) > 0):
            if (len(word) == 1):
                singlet = word[0]
                pixel = .25
                if singlet[0] in alphabet:
                    position = (alphabet.index(singlet[0]) + 2)
                    full_vector[position] = pixel
                full_vector[1] = 1
            elif (len(word) == 2):
                doublet = word[0:2]
                pixel = .25
                for letter in range(0, 2):
                    if doublet[letter] in alphabet:
                        position = (alphabet.index(doublet[letter]) + 2)
                        full_vector[position] = pixel
                        pixel = pixel + .25
                full_vector[1] = 1
            elif (len(word) == 3):
                triplet = word[0:3]
                pixel = .25
                for letter in range(0, 3):
                    if triplet[letter] in alphabet:
                        position = (alphabet.index(triplet[letter]) + 2)
                        full_vector[position] = pixel
                        pixel = pixel + .25
                full_vector[1] = 1
            elif (len(word) > 3):
                full_vector[0] = English_or_zero
                word_length = len(word)
                if (word_length % 2 == 0):
                    step = 2
                else:
                    step = 1
                offset = 0
                while (e <= len(word)):
                    quadruplet = (word[s:e])
                    pixel = .25
                    for letter in range(0, 4):
                        if quadruplet[letter] in alphabet:
                            position = ((alphabet.index(quadruplet[letter]) + 2) + offset)
                            full_vector[position] = pixel
                            pixel = pixel + .25
                    s = s + step
                    e = e + step
                    loops = loops + 1
                    offset = offset + len(alphabet)
                full_vector[1] = loops
            full_vector_stack = np.vstack((full_vector, full_vector_stack))
            for f in range(0, len(full_vector)):
                full_vector[f] = 0
        return full_vector_stack;

full_vector = np.zeros(flattened_length_with_2_tags)
full_vector_stack = np.zeros((0, flattened_length_with_2_tags))



for b in range(0, len(pdf_contents_English)):
    get_word_contents(b, pdf_contents_English, 2)
f.close()
    
for b in range(0, len(pdf_contents_French)):
    get_word_contents(b, pdf_contents_French, 1)
f.close()

for b in range(0, len(pdf_contents_Spanish)):
    get_word_contents(b, pdf_contents_Spanish, 0)
f.close()

    
np.random.shuffle(full_vector_stack)
    
FullVectorStack = TemporaryFile()
np.save("FullVectorStack", full_vector_stack)

