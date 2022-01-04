import math
import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow
from sklearn import preprocessing
from sklearn.externals import joblib
import re


# This is a nerual network with many layers.  The 1st layer is recurrent, so it can learn patterns in data


#----------- Input Vector, Thetas, Biases and AActivations ------------------------

full_vector_stack = np.load('FullVectorStack.npy')
full_vector = full_vector_stack[0,:]


alphabet = ["a","á","à","â","b","c","ç","d","e","é","è","ê","ë","f","g","h","i","í","î","ï","j","k","l","m","n","ñ","o","ó","ô","p","q","r","s","t","u","ù","û","ü","v","w","x","y","z"]
input_nodes = len(alphabet)
middle_nodes_1 = math.ceil(input_nodes*2)
middle_nodes_2 = math.ceil(input_nodes*4)
middle_nodes_3 = math.ceil(input_nodes*3)
final_nodes = 3

holder = np.zeros((input_nodes))

steps = 20
flattened_length_with_2_tags = (steps * len(alphabet)) + 2
flattened_length_no_tags = steps * len(alphabet)
flattened_length = flattened_length_no_tags + 1

TLoop, TLoop_c = np.random.uniform(-1,1, (steps, input_nodes, input_nodes)), np.random.uniform(-1,1, (steps, input_nodes, input_nodes))
T0, t0c = np.random.uniform(-1,1, (input_nodes, middle_nodes_1)), np.random.uniform(-1,1, (input_nodes, middle_nodes_1))
T1, t1c = np.random.uniform(-1,1, (middle_nodes_1, middle_nodes_2)), np.random.uniform(-1,1, (middle_nodes_1, middle_nodes_2))
T2, t2c = np.random.uniform(-1,1, (middle_nodes_2, middle_nodes_3)), np.random.uniform(-1,1, (middle_nodes_2, middle_nodes_3))
T3, t3c = np.random.uniform(-1,1, (middle_nodes_3, final_nodes)), np.random.uniform(-1,1, (middle_nodes_3, final_nodes))

BLoop = np.random.uniform(-1,1, (steps, input_nodes))
B0 = np.random.uniform(-1,1, middle_nodes_1)
B1 = np.random.uniform(-1,1, middle_nodes_2)
B2 = np.random.uniform(-1,1, middle_nodes_3)
B3 = np.random.uniform(-1,1, final_nodes)

ALoop = np.zeros((steps, input_nodes))
aa0 = np.zeros((middle_nodes_1))
aa1 = np.zeros((middle_nodes_2))
aa2 = np.zeros((middle_nodes_3))
aa3 = np.zeros((final_nodes))

ELoop = np.zeros((steps, input_nodes))
e0 = np.zeros((middle_nodes_1))
e1 = np.zeros((middle_nodes_2))
e2 = np.zeros((middle_nodes_3))
e3 = np.zeros((final_nodes))

bias = 1
lr = .1
total_error = 0
shirase_interval = 500
r = 1

x = list()
y = list()


        
actual_x = np.zeros((final_nodes))


def get_a_activation ( Theta, Bias, input_vector  ):
    "Given a theta matrix and an input vector, this gives the A activations"
    av = np.matmul(input_vector, Theta)
    Bias_temp = Bias * bias
    av = av + Bias_temp
    av = 1/(1+(np.exp(-(av))))
    get_a_activation.a = av
    av = np.zeros(len(av))
   

def get_final_error ( desired_output_vector, activations_vector ):
    "This gets the final error of the network"
    global total_error
    final_error_vector = activations_vector*(1-activations_vector)*(desired_output_vector - activations_vector)
    final_temp = abs(final_error_vector)
    total_error = total_error + np.sum(final_temp, axis=0)
    get_final_error.e = final_error_vector
    get_final_error.t = total_error


def get_intermediate_errors ( Theta, previous_error, activations_vector ):
    "This gets the errors in the hidden layers"
    e_x_t = previous_error * Theta
    errors_in = np.sum(e_x_t, axis=1)
    Act_temp = (1-activations_vector)*activations_vector
    this_error_vector = Act_temp * errors_in
    #print(this_error_vector)
    get_intermediate_errors.e = this_error_vector
    
    
    
    
    

def update_thetas ( Theta, Bias, error_vector, one_back_activations_vector, learning_rate ):
    "This function updates theta values"
    A_temp = one_back_activations_vector.reshape(-1, 1)
    e_x_a = A_temp * error_vector
    New_theta = Theta + (learning_rate*(e_x_a))
    New_bias = Bias + (learning_rate*(bias * (error_vector)))
    #print(New_theta)
    update_thetas.t = New_theta
    update_thetas.b = New_bias
    
def update_loop_thetas ( Theta, Bias, error_vector, one_back_activations_vector, learning_rate ):
    "This function updates theta values"
    A_temp = one_back_activations_vector#.reshape(-1, 1)
    e_x_a = A_temp * error_vector
    New_theta = Theta + (learning_rate*(e_x_a))
    New_bias = Bias + (learning_rate*(bias * (error_vector)))
    #print(New_theta)
    update_loop_thetas.t = New_theta
    update_loop_thetas.b = New_bias

    
    
    
    
    
    

def make_fig():
    plt.scatter(x, y)
    plt.draw()

def test_word( word_list_word, which_file ):
    "This processes a word you input into the console so that it can be analyzed by the NN"
    s = 0
    e = 4
    loops = 0
    word = which_file[word_list_word]
    word = re.sub('/', ' ', word, count=0, flags=0)
    word = re.sub('[^a-zA-Záàâçéèêëîíïñóôùûü]', '', word, count=0, flags=0)
    word = word.lower()
    if (len(word) > 0):
        if (len(word) == 1):
            singlet = word[0]
            pixel = .25
            if singlet[0] in alphabet:
                position = (alphabet.index(singlet[0]) + 1)
                full_vector[position] = pixel
            full_vector[0] = 1
        elif (len(word) == 2):
            doublet = word[0:2]
            pixel = .25
            for letter in range(0, 2):
                if doublet[letter] in alphabet:
                    position = (alphabet.index(doublet[letter]) + 1)
                    full_vector[position] = pixel
                    pixel = pixel + .25
            full_vector[0] = 1
        elif (len(word) == 3):
            triplet = word[0:3]
            pixel = .25
            for letter in range(0, 3):
                if triplet[letter] in alphabet:
                    position = (alphabet.index(triplet[letter]) + 1)
                    full_vector[position] = pixel
                    pixel = pixel + .25
            full_vector[0] = 1
        elif (len(word) > 3):
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
                        position = ((alphabet.index(quadruplet[letter]) + 1) + offset)
                        full_vector[position] = pixel
                        pixel = pixel + .25
                s = s + step
                e = e + step
                loops = loops + 1
                offset = offset + len(alphabet)
            full_vector[0] = loops
        test_word.f = full_vector
        test_word.w = word
                    

vector_stack_with_tags = full_vector_stack #complete vector stack imported from word_tester.  It has language tags at [0] and step tags at [1]
vector_stack_with_NO_tags = np.delete(full_vector_stack, [0,1], axis=1) #trim the language tags and step tags, for training purposes
scaler = preprocessing.StandardScaler(copy=True).fit(vector_stack_with_NO_tags) #set up a scaler, showing it the whole data set 
scaled_vector_stack_with_NO_tags = scaler.transform(vector_stack_with_NO_tags) #scale, before training with it

joblib.dump(scaler, 'scalerForRNN.pkl')

for q in range(0, 80000):
    for i in range(0, len(vector_stack_with_tags)):
        actual_holder = vector_stack_with_tags[i,:]
        word_steps = int(actual_holder[1])
        full_vector = scaled_vector_stack_with_NO_tags[i,:]
        if (actual_holder[0] == 2):
            actual_x = [0,0,1]
        elif (actual_holder[0] == 1):
            actual_x = [0,1,0]
        elif (actual_holder[0] == 0):
            actual_x = [1,0,0]

        
        word_steps = word_steps -1
        
        start = 0
        end = len(alphabet)
        if (word_steps == 0):
            holder = full_vector[start:end]
        elif (word_steps > 0):
            for s in range(0, word_steps):
                t_and_b = s + 1
                get_a_activation(TLoop[t_and_b], BLoop[t_and_b], full_vector[start:end])
                ALoop[t_and_b] = get_a_activation.a
                start = start + len(alphabet)
                end = end + len(alphabet)
                holder = ALoop[s+1] + full_vector[start:end] #+ holder


        
        # we now have LA0, LA1 and LA2 stored in ALoop[one, two and three]
            
        # on to the strictly feed-forward section of the network! (from here below)
        get_a_activation(T0, B0, holder )
        aa0 = get_a_activation.a
       
        get_a_activation(T1, B1, aa0 )
        aa1 = get_a_activation.a
        
        get_a_activation(T2, B2, aa1 )
        aa2 = get_a_activation.a
        
        get_a_activation(T3, B3, aa2)
        aa3 = get_a_activation.a
    
        get_final_error(actual_x, aa3 )
        e3 = get_final_error.e
        total_error = get_final_error.t
        
        
        # do calculation updates of thetas to get middle errors
        
        update_thetas(T3, B3, e3, aa2, 1)
        t3c = update_thetas.t
        get_intermediate_errors(t3c, e3, aa2 )
        e2 = get_intermediate_errors.e
        
        update_thetas(T2, B2, e2, aa1, 1)
        t2c = update_thetas.t
        get_intermediate_errors(t2c, e2, aa1 )
        e1 = get_intermediate_errors.e
        
        update_thetas(T1, B1, e1, aa0, 1)
        t1c = update_thetas.t
        get_intermediate_errors(t1c, e1, aa0)
        e0 = get_intermediate_errors.e
        
        update_thetas(T0, B0, e0, ALoop[word_steps], 1)
        t0c = update_thetas.t
        get_intermediate_errors(t0c, e0, ALoop[word_steps])
        ELoop[word_steps] = get_intermediate_errors.e
        
        for s in range(word_steps, 0, -1):
            a = s - 1
            update_loop_thetas (TLoop[s], BLoop[s], ELoop[s], ALoop[a], 1)
            TLoop_c[s] = update_loop_thetas.t
            get_intermediate_errors(TLoop_c[s], ELoop[s], ALoop[a])
            ELoop[a] = get_intermediate_errors.e
        
        
        # do real updates of thetas
        
        update_thetas(T3, B3, e3, aa2, lr)
        T3 = update_thetas.t
        B3 = update_thetas.b
        update_thetas(T2, B2, e2, aa1, lr)
        T2 = update_thetas.t
        B2 = update_thetas.b
        update_thetas(T1, B1, e1, aa0, lr)
        T1 = update_thetas.t
        B1 = update_thetas.b
        update_thetas(T0, B0, e0, ALoop[word_steps], lr)
        T0 = update_thetas.t
        B0 = update_thetas.b

        for s in range(word_steps, 0, -1):
            a = s - 1
            update_loop_thetas (TLoop[s], BLoop[s], ELoop[s], ALoop[a], lr)
            TLoop[s] = update_loop_thetas.t
            BLoop[s] = update_loop_thetas.b
#        if (r == shirase_interval):
#            print(actual_x)
#            print(aa3)
#            print("")
        
        holder = np.zeros((input_nodes))
        
        
       
            
       
        
    x.append(q)
    y.append(total_error)
    if (r == shirase_interval):
        drawnow(make_fig)
        print(total_error)
        r = 0
    r = r + 1
    if total_error < .00005:
       break
    total_error = 0
    
    if (r == shirase_interval):
        
        full_vector = np.zeros(flattened_length)
    
        h = open('C:\\Users\\QRG_02\\Desktop\\language_detector\\Test.txt','r', encoding='UTF-8')
        test_contents = h.read().split(' ')
    
        for b in range(0, len(test_contents)):
            test_word( b, test_contents )
            word = test_word.w
            if (len(word) > 0):
                full_vector = test_word.f
                
                word_steps = int(full_vector[0])
                word_steps = word_steps - 1
                
                full_vector = np.delete(full_vector, 0, axis=0)

                full_vector = full_vector.reshape(1, -1)
                full_vector = scaler.transform(full_vector)
                full_vector = full_vector.reshape(flattened_length_no_tags,)
              
            start = 0
            end = len(alphabet)
            for s in range(0, word_steps):
                t_and_b = s + 1
                get_a_activation(TLoop[t_and_b], BLoop[t_and_b], full_vector[start:end])
                ALoop[s+1] = get_a_activation.a
                start = start + len(alphabet)
                end = end + len(alphabet)
                full_vector[start:end] = ALoop[s+1] + full_vector[start:end]
            # we now have LA0, LA1 and LA2 stored in ALoop[one, two and three]
                
            # on to the strictly feed-forward section of the network! (from here below)
            
    
            get_a_activation(T0, B0, full_vector[start:end] )
            aa0 = get_a_activation.a
            
            get_a_activation(T1, B1, aa0 )
            aa1 = get_a_activation.a
            
            get_a_activation(T2, B2, aa1 )
            aa2 = get_a_activation.a
            
            get_a_activation(T3, B3, aa2)
            aa3 = get_a_activation.a
    
            print(aa3)
            print(word)
            print("")
                
            full_vector = np.zeros(flattened_length)

    
        h.close()

    
from tempfile import TemporaryFile
ThetaZero = TemporaryFile()
ThetaOne = TemporaryFile()
ThetaTwo = TemporaryFile()
BiasZero = TemporaryFile()
BiasOne = TemporaryFile()
BiasTwo = TemporaryFile()
np.save("ThetaZero", T0)
np.save("ThetaOne", T1)
np.save("ThetaTwo", T2)
np.save("BiasZero", B0)
np.save("BiasOne", B1)
np.save("BiasTwo", B2)

full_vector = np.zeros(flattened_length)

response = ""

while (response != "exit"):
    response = input("Change the file contents, or type exit")

    h = open('C:\\Users\\QRG_02\\Desktop\\language_detector\\Verification.txt','r', encoding='UTF-8')
    test_contents = h.read().split(' ')

    for b in range(0, len(test_contents)):
        test_word( b, test_contents )
        word = test_word.w
        if (len(word) > 0):
            full_vector = test_word.f
                
            word_steps = int(full_vector[0])
            word_steps = word_steps - 1
                
            full_vector = np.delete(full_vector, 0, axis=0)
            
            full_vector = full_vector.reshape(1, -1)
            full_vector = scaler.transform(full_vector)
            full_vector = full_vector.reshape(flattened_length_no_tags,)
              
        start = 0
        end = len(alphabet)
        for s in range(0, word_steps):
            t_and_b = s + 1
            get_a_activation(TLoop[t_and_b], BLoop[t_and_b], full_vector[start:end])
            ALoop[s+1] = get_a_activation.a
            start = start + len(alphabet)
            end = end + len(alphabet)
            full_vector[start:end] = ALoop[s+1] + full_vector[start:end]
        # we now have LA0, LA1 and LA2 stored in ALoop[one, two and three]
                
        # on to the strictly feed-forward section of the network! (from here below)
            
    
        get_a_activation(T0, B0, full_vector[start:end] )
        aa0 = get_a_activation.a
            
        get_a_activation(T1, B1, aa0 )
        aa1 = get_a_activation.a
            
        get_a_activation(T2, B2, aa1 )
        aa2 = get_a_activation.a
            
        get_a_activation(T3, B3, aa2)
        aa3 = get_a_activation.a
    
        print(aa3)
        print(word)
        print("")
    
                
        full_vector = np.zeros(flattened_length)
    
    h.close()
