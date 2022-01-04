import copy, numpy as np
from graphics import *
import time
import matplotlib.image as img
import random
import re

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

def make_a_column(number_of_rows, x_coord, y_coord, which_layer, x_width, border_width):
    "This makes a single column, which is one layer in the NN"
    xc = list()
    node = 0
    original_x = x_coord
    for row in range(1, number_of_rows+1):
        pt = Point(x_coord, y_coord)
        pt2 = Point(x_coord+15, y_coord+15)
        rect = Rectangle(pt, pt2)
        rect.draw(win)
        rect.setOutline("black")
        rect.setWidth(border_width)
        rect.setFill(color_rgb(int(255*which_layer[node]), int(255*which_layer[node]), int(255*which_layer[node])))
        xc.append(rect)
        node += 1
        x_coord = x_coord + 15
        if row % x_width == 0:
            y_coord = y_coord + 15
            x_coord = original_x
    return xc

def get_stack(word):
    "Puts all of the letters of a word into a stack of vectors that are one-hot"
    global stack
    for let in range(0, len(word)-1):
        input_vector[alphabet.index(word[let])] = 1
        stack = np.vstack((stack, input_vector))
        input_vector[alphabet.index(word[let])] = 0
    return stack

# --------------------------------------------------------------------------------------- #
    




output_dim = 2


    
alphabet = ["a","á","à","â","b","c","ç","d","e","é","è","ê","ë","f","g","h","i","í","î","ï","j","k","l","m","n","ñ","o","ó","ô","p","q","r","s","t","u","ù","ú","û","ü","v","w","x","×","y","z"]
input_vector = np.zeros(len(alphabet))
stack = np.zeros(len(alphabet))
actual_x = np.zeros(output_dim)

alphabet_length = len(alphabet)

alpha = .1
input_dim = alphabet_length
hidden_dim = 600




# open both word files and put all the words into a list called
# A_words, which has language information attached to each word inside of it,
# either 0 for English or 1 for non-English

file = open('C:\\Users\\QRG_02\\Desktop\\original_words.txt', 'r', encoding='latin-1')
file2 = open('C:\\Users\\QRG_02\\Desktop\\other_words.txt', 'r', encoding='latin-1')

E = file.read()
NE = file2.read()

E_words = E.split()
for w in range(0, len(E_words)):
    E_words[w] = E_words[w].lower()
    word = re.sub('[^aáàâbcçdeéèêëfghiíîïjklmnñoóôpqrstuùúûüvwx×yz]', '', E_words[w])
    E_words[w] = word + "0"
NE_words = NE.split()
for w in range(0, len(NE_words)):
    NE_words[w] = NE_words[w].lower()
    word = re.sub('[^aáàâbcçdeéèêëfghiíîïjklmnñoóôpqrstuùúûüvwx×yz]', '', NE_words[w])
    NE_words[w] = word + "1"
A_words = NE_words + E_words
random.shuffle(A_words, random.random)
A_words = [x for x in A_words if x != "0"]
A_words = [x for x in A_words if x != "1"]





# state nodes and output nodes definition
state = np.zeros((hidden_dim))
output = np.random.rand((output_dim))




# Here is our graphics stuff.  Make a plotting window, and 
# fill it with graphics relating to the letter input
# the memory state of the RNN, and the output,
# as well as the desired output
win = GraphWin("Network", 1200,400)

letter = make_a_column(alphabet_length, 10, 10, stack, 30, 1)
stte = make_a_column(hidden_dim, 500, 10, state, 30, 1)
out = make_a_column(output_dim, 1000, 10, output, 30, 1)
desired = make_a_column(output_dim, 1000, 40, actual_x, 30, 1)

for x in range(0, output_dim):
    desired[x].undraw()





# initialize neural network weights and define some sizes
input_thetas = 2*np.random.random((input_dim,hidden_dim)) - 1
hidden_thetas = 2*np.random.random((hidden_dim,hidden_dim)) - 1
output_thetas = 2*np.random.random((hidden_dim,output_dim)) - 1

input_thetas_update = np.zeros_like(input_thetas)
hidden_thetas_update = np.zeros_like(hidden_thetas)
output_thetas_update = np.zeros_like(output_thetas)



sleeptime = 0
overallError = 0




for j in range(1000):
    #if (j > 450):
    #    sleeptime = .2
    for count in range(0, len(A_words)): # A_words is the complete list of all words in question, with language info attached
        
        stack = get_stack(A_words[count]) # stack = the word in question
        length = len(A_words[count])      # get the length of the word in letters
        length = length - 1               # get the position of the final letter in the word
        if A_words[count][length] == "0": # if the final letter is "0", the word is English, so assign an English actual_x
            actual_x = np.array([1, 0])
        else:                             # if the final letter is not "0", the word is not English
            actual_x = np.array([0, 1])
            
        stack = np.delete(stack, 0, 0)    # the first row in the stack is always empty, so let's delete it
            
        message = Text(Point(600, 350), A_words[count])
        message.draw(win)
        
        # Get the network ready analyzing one word
        output_deltas = list()
        state_values = list()
        state_values.append(np.zeros(hidden_dim))
        
        
        for position in range(0,len(stack)):
            X = stack[position]
            y = actual_x
    
            # hidden layer (input ~+ prev_hidden)
            state = sigmoid(np.dot(X,input_thetas) + np.dot(state_values[-1],hidden_thetas))
            
            for x in range(0, alphabet_length):
                letter[x].setFill(color_rgb(int(255*X[x]), int(255*X[x]), int(255*X[x])))
            for x in range(0, hidden_dim):
                stte[x].setFill(color_rgb(int(255*state[x]), int(255*state[x]), int(255*state[x])))
            for x in range(0, output_dim):
                out[x].setFill(color_rgb(int(255*output[x]), int(255*output[x]), int(255*output[x])))
            time.sleep(sleeptime)
    
            # output layer
            output = sigmoid(np.dot(state,output_thetas))
        
            # did we miss?... if so, by how much?
            output_error = (y - output) * 0
            if position == (len(stack)-1):
                output_error = y - output                                                
            output_deltas.append((output_error)*sigmoid_output_to_derivative(output))    
            overallError += np.abs(output_error[0])
            
            # store hidden layer so we can use it in the next timestep
            state_values.append(copy.deepcopy(state))
        
        future_state_delta = np.zeros(hidden_dim)
        
        for position in range(0, len(stack)):
            
            X = stack[position]
            state = state_values[-position-1]
            prev_state = state_values[-position-2]
            
            # error at output layer
            output_delta = output_deltas[-position-1]
            # error at hidden layer
            state_delta = (future_state_delta.dot(hidden_thetas.T) + output_delta.dot(output_thetas.T)) * sigmoid_output_to_derivative(state)
            
            # let's update all our weights so we can try again
            output_thetas_update += np.atleast_2d(state).T.dot(np.atleast_2d(output_delta))
            hidden_thetas_update += np.atleast_2d(prev_state).T.dot(np.atleast_2d(state_delta))
            input_thetas_update += np.atleast_2d(X).T.dot(np.atleast_2d(state_delta))
            
            future_state_delta = state_delta
        
    
        input_thetas += input_thetas_update * alpha
        output_thetas += output_thetas_update * alpha
        hidden_thetas += hidden_thetas_update * alpha    
    
        input_thetas_update *= 0
        output_thetas_update *= 0
        hidden_thetas_update *= 0
        
        # we finished analyzing one word, so empty the stack (that holds the word letters), as well as the memory state
        stack = np.zeros(len(alphabet))
        state = np.zeros((hidden_dim))
        
        
        # compare our actual_x and h_of_x visually
        for x in range(0, output_dim):
            desired[x].setFill(color_rgb(int(255*y[x]), int(255*y[x]), int(255*y[x])))
            out[x].setFill(color_rgb(int(255*output[x]), int(255*output[x]), int(255*output[x])))
            desired[x].draw(win)
        
        time.sleep(0)
        for x in range(0, output_dim):
            desired[x].undraw()
            message.undraw()
        
    print(overallError)
    if (overallError < 500):
        break
    overallError = 0

time.sleep(5)
win.close()

from tempfile import TemporaryFile

ht = TemporaryFile()
ot = TemporaryFile()
np.save("hidden_thetas", hidden_thetas)
np.save("output_thetas", output_thetas)