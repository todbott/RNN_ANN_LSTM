# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:47:32 2018

@author: kikaku03
"""

import numpy as np
import math
from drawnow import drawnow
import matplotlib.pyplot as plt


alphabet = ["a","á","à","â","b","c","ç","d","e","é","è","ê","ë","f","g","h","i","í","î","ï","j","k","l","m","n","ñ","o","ó","ô","p","q","r","s","t","u","ù","û","ü","v","w","x","y","z"," ","?",",","!",".","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","-","1","2","3","4","5","6","7","8","9","0","%","«",";",":","—","Ä","(",")","/","[","]","\n"]
input_vector = np.zeros(len(alphabet))
stack = np.zeros(len(alphabet))



## --------- weights, ubbers and biases for gates -------------

A_Theta = np.random.uniform(-1,1, (len(alphabet)))
I_Theta = np.random.uniform(-1,1, (len(alphabet)))
F_Theta = np.random.uniform(-1,1, (len(alphabet)))
O_Theta = np.random.uniform(-1,1, (len(alphabet)))

A_Ubber = .1
I_Ubber = .1
F_Ubber = .1
O_Ubber = .1

A_Bias = 1
I_Bias= 1
F_Bias = 1
O_Bias = 1



file = open('C:\\Users\\QRG_02\\Desktop\\returns.txt', 'r', encoding='UTF-8')

phrase_to_learn = file.read()
letters = list(phrase_to_learn)

actual_x = np.zeros(len(letters))

for letter in range(0, len(letters)-1):
    input_vector[alphabet.index(letters[letter])] = 1
    stack = np.vstack((stack, input_vector))
    if (input_vector[96] == 1):
        actual_x[letter+1] = 1
    else:
        actual_x[letter+1] = 0
    input_vector[alphabet.index(letters[letter])] = 0
    
""" in the above code, we have created a stack of vectors which are one-hot,
so, if the alpahbet is only "a b c d e f g", the word "cad" will become

[0,0,1,0,0,0,0]
[1,0,0,0,0,0,0]
[0,0,0,1,0,0,0]"""

print("finished")


## ---- A activations, input activations, forget activations and output activations at each time step------------

ALoop = np.zeros(len(letters))
ILoop = np.zeros(len(letters))
FLoop = np.zeros(len(letters)+1)
OLoop = np.zeros(len(letters))

# ------ Activation Gate activations and Input Vector values at each time step ----------

AGLoop = np.zeros(len(letters))

# ---- Memory values, gradient values, and errors at each time step --------

MLoop = np.zeros(len(letters)+1)
GLoop = np.zeros((len(letters), len(alphabet)))
ELoop = np.zeros(len(letters))

# -----  Matrices and vectors used for updating weights, ubbers and biases -------

Theta_Updates = np.zeros((len(alphabet), len(alphabet)))
B_Updates = np.zeros(len(alphabet))
U_Updates = 0

lr = .001
total_error = 0
shirase_interval = 100
x = list()
for w in range(0, len(letters)):
    x.append(w)
y = list()
y2 = list()
r = 0

def sigmoid_squash ( x ):
    "sigmoid squashing function"
    x = 1/(1+(np.exp(-(x))))
    return x

def tanh_squash ( x ):
    "tangent squashing function"
    x = (2/(1+(np.exp(-(2*x)))))-1
    return x
    
def activation_gate ( activation, input_vector, Bias, Theta, Ubber ):
    "..."
    av = np.matmul(input_vector, Theta)
    av2 = activation * Ubber
    av3 = av + av2 + Bias
    x = tanh_squash(av3)
    return x

def input_gate ( activation, input_vector, Bias, Theta, Ubber ):
    "..."
    av = np.matmul(input_vector, Theta)
    av2 = activation * Ubber
    av3 = av + av2 + Bias
    x = sigmoid_squash(av3)
    return x
    
def forget_gate ( activation, input_vector, Bias, Theta, Ubber  ):
    "..."
    av = np.matmul(input_vector, Theta)
    av2 = activation * Ubber
    av3 = av + av2 + Bias
    x = sigmoid_squash(av3)
    return x
    
def output_gate ( activation, input_vector, Bias, Theta, Ubber ):
    "..."
    av = np.matmul(input_vector, Theta)
    av2 = activation * Ubber
    av3 = av + av2 + Bias
    x = sigmoid_squash(av3)
    return x

def alter_memory ( a0, i0, f0, memory):
    "..."
    x = (np.multiply(a0, i0) + np.multiply(f0, memory))
    return x

def make_fig():
    plt.scatter(x, ALoop)
    plt.scatter(x, actual_x)
    plt.draw()

    

for q in range(0, 500000):
    for s in range(0, len(letters)): # forward propagation through time steps in the LSTM
        AGLoop[s] = activation_gate (ALoop[s-1], stack[s], A_Bias, A_Theta, A_Ubber)
        ILoop[s] = input_gate (ALoop[s-1], stack[s], I_Bias, I_Theta, I_Ubber)
        FLoop[s] = forget_gate (ALoop[s-1], stack[s], F_Bias, F_Theta, F_Ubber)
        OLoop[s] = output_gate (ALoop[s-1], stack[s], O_Bias, O_Theta, O_Ubber)
        MLoop[s] = alter_memory(AGLoop[s], ILoop[s], FLoop[s], MLoop[s-1])
        ALoop[s] = np.multiply(tanh_squash(MLoop[s]), OLoop[s])
        
    for s in range(len(letters)-1, -1, -1): # back propagation through time steps in the LSTM to find errors
        this_error = (ALoop[s] - actual_x[s]) + ELoop[s] # this error plus the error from the next timestep forward
        total_error = total_error + abs(this_error)
        old_state = MLoop[s]
        MLoop[s] = np.multiply(this_error, OLoop[s]) * (1-(math.pow(np.tanh(MLoop[s]),2))) + (np.multiply(MLoop[s+1], FLoop[s+1]))
        GLoop[s,0] = np.multiply(MLoop[s], ILoop[s]) * (1-(math.pow(AGLoop[s],2)))
        GLoop[s,1] = np.multiply(MLoop[s], AGLoop[s]) * np.multiply(ILoop[s], (1-(ILoop[s])))
        GLoop[s,2] = np.multiply(MLoop[s], MLoop[s-1]) * np.multiply(FLoop[s], (1-FLoop[s]))
        GLoop[s,3] = np.multiply(this_error, np.tanh(old_state)) * np.multiply(OLoop[s], (1-OLoop[s]))
        ELoop[s-1] = (A_Ubber * GLoop[s, 0]) + (I_Ubber * GLoop[s, 1]) + (F_Ubber * GLoop[s, 2]) + (O_Ubber * GLoop[s, 3])
        

    if (r == shirase_interval):
        drawnow(make_fig)
        print(total_error)
        r = 0
    r = r + 1
    total_error = 0
    
        
        
    for s in range(0, len(letters)): ## update Thetas and Biases
        v = np.outer(stack[s], GLoop[s,]) 
        Theta_Updates = Theta_Updates + v
        b = GLoop[s,]
        B_Updates = B_Updates + b

    
    for s in range(0, (len(letters)-1)): ## update Ubbers
        u = np.outer(GLoop[s+1], ALoop[s])
        U_Updates = U_Updates + u
    
    # update A I F O Thetas
    
    A_Theta = A_Theta - (lr*Theta_Updates[0])
    I_Theta = I_Theta - (lr*Theta_Updates[1])
    F_Theta = F_Theta - (lr*Theta_Updates[2])
    O_Theta = O_Theta - (lr*Theta_Updates[3])
    
    A_Ubber = A_Ubber - (lr*U_Updates[0])
    I_Ubber = I_Ubber - (lr*U_Updates[1])
    F_Ubber = F_Ubber - (lr*U_Updates[2])
    O_Ubber = O_Ubber - (lr*U_Updates[3])
    
    A_Bias = A_Bias - (lr*B_Updates[0])
    I_Bias = I_Bias - (lr*B_Updates[1])
    F_Bias = F_Bias - (lr*B_Updates[2])
    O_Bias = O_Bias - (lr*B_Updates[3])
    
    Theta_Updates = np.zeros((len(alphabet), len(alphabet)))
    B_Updates = np.zeros(len(alphabet))
    U_Updates = 0
    
    
    
   

    
