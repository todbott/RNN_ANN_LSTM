import numpy as np
from graphics import *
import matplotlib.image as img
import io
import matplotlib.pyplot as plt

tag_list = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

def populate_HofX(tags, pic):
    zeros_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    zeros_list[tag_list.index(tags[pic].split(",")[0])] = 1
    return zeros_list

def return_test_result(result):
    return tag_list[np.where(result == np.amax(result))[0][0]]
    











x_length = 784  # dimension of the image array flattened
x_steps = 50000     # how many images
x_testing_steps = 10000 # how many images for testing
x_width = 28    # for display purposes, what is the width of the picture?

layer1_length = 300   # dimension of the middle (hidden) layer
layer2_length = 128

actual_x_length = 10   # dimension of the output

h_of_x = np.zeros((actual_x_length))
x_array = np.zeros(x_length)
x = np.zeros((x_length))

#win = GraphWin("Network", 600,800)


#def make_a_column(number_of_rows, x_coord, y_coord, which_layer, x_width, border_width):
#    "This makes a single column, which is one layer in the NN"
#    xc = list()
#    node = 0
#    original_x = x_coord
#    for row in range(1, number_of_rows+1):
#        pt = Point(x_coord, y_coord)
#        pt2 = Point(x_coord+5, y_coord+5)
#        rect = Rectangle(pt, pt2)
#        rect.draw(win)
#        rect.setOutline("black")
#        rect.setWidth(border_width)
#        rect.setFill(color_rgb(int(255*which_layer[node]), 0, 0))
#        xc.append(rect)
#        node += 1
#        x_coord = x_coord + 5
#        if row % x_width == 0:
#            y_coord = y_coord + 5
#            x_coord = original_x
#    return xc


""" put the images and tags into their respective arrays """
training_array = []
h_of_x_array = []

testing_array = []
testing_h_of_x_array = []

tags = io.open("C://Users//Hotaru LSG//Desktop//python//MINST fashion//index.csv", "r")
tags = tags.readlines()
for pic in range(0, x_steps):
    image = img.imread("C://Users//Hotaru LSG//Desktop//python//MINST fashion//img//fashion%d.png" % pic)
    training_array.append(image.flatten())
    h_of_x_array.append(populate_HofX(tags, pic))
    
for pic in range(x_steps, x_steps + x_testing_steps):
    image = img.imread("C://Users//Hotaru LSG//Desktop//python//MINST fashion//img//fashion%d.png" % pic)
    testing_array.append(image.flatten())
    testing_h_of_x_array.append(populate_HofX(tags, pic)) 
    
training_array = np.vstack(training_array)
h_of_x_array = np.vstack(h_of_x_array)

testing_array = np.vstack(testing_array)
testing_h_of_x_array = np.vstack(testing_h_of_x_array)




t1 = np.random.uniform(-1,1,(x_length, layer1_length))
t1_update = np.zeros((x_length, layer1_length))

state1 = np.zeros((layer1_length))

t2 = np.random.uniform(-1,1,(layer1_length, layer2_length))
t2_update = np.zeros((layer1_length, layer2_length))

t2_temp = np.zeros((layer1_length, layer2_length))

state2 = np.zeros((layer2_length))

t3 = np.random.uniform(-1,1,(layer2_length, actual_x_length))
t3_update = np.zeros((layer2_length, actual_x_length))

state3 = np.zeros((actual_x_length))

error1 = np.zeros((layer1_length))
error2 = np.zeros((layer2_length))
error3 = np.zeros((actual_x_length))




#RGB_squares = make_a_column(x_length, 10, 10, x, x_width, 0)
#desired_squares = make_a_column(actual_x_length, 10, 170, h_of_x, x_width, 1)
#
#Ra1_squares = make_a_column(layer1_length, 10, 500, state1, x_width, 1)
#Ra2_squares = make_a_column(actual_x_length, 10, 650, state2, x_width, 1)






training_cycles = 1000
absolute_error = 0
lr = 0.05


trainingX = []
trainingY = []
testingX = []
testingY = []

for j in range(training_cycles):
    
    total_error = 0
    
    for step in range(0, x_steps):
        
        h_of_x = h_of_x_array[step]
        
        
        # put a full-color picture of the current training example at the top of the screen, and put
        # the desired output at the bottom
        #for sq in range(0, x_length):
        #    RGB_squares[sq].setFill(color_rgb(int(255*training_array[step][sq]), int(255*training_array[step][sq]), int(255*training_array[step][sq])))
        #for sq in range(0, len(h_of_x)):
        #    desired_squares[sq].setFill(color_rgb(int(255*h_of_x[sq]), int(255*h_of_x[sq]), int(255*h_of_x[sq])))
 
        x = training_array[step]
        
        # forward pass 
        state1 = sigmoid(np.dot(x, t1))
        state2 = sigmoid(np.dot(state1, t2))
        state3 = sigmoid(np.dot(state2, t3))
        
    
        # errors and backward pass
        error3 = (h_of_x - state3) * sigmoid_output_to_derivative(state3)
        t3_temp = t3 + (state2.reshape(-1,1)*(error3))
        
        error2 = (np.dot(error3,t3_temp.T)) * sigmoid_output_to_derivative(state2)
        t2_temp = t2 + (state1.reshape(-1,1)*(error2))
        
        error1 = (np.dot(error2,t2_temp.T)) * sigmoid_output_to_derivative(state1)
        
        t3_update += (state2.reshape(-1,1)*(error3)) * lr
        t2_update += (state1.reshape(-1,1)*(error2)) * lr
        t1_update += (x.reshape(-1,1)*(error1)) * lr

        # theta updates
        t1 += t1_update 
        t2 += t2_update  
        t3 += t3_update
    
        t1_update *= 0
        t2_update *= 0
        t3_update *= 0

        total_error = abs(np.sum(error3) + np.sum(error2) + np.sum(error1))
        absolute_error = total_error + absolute_error
        
        # put state information into the hidden layer images, and state information into the final layer images
        #for sq in range(0, layer1_length):
        #    Ra1_squares[sq].setFill(color_rgb(int(255*state1[sq]), int(255*state1[sq]), int(255*state1[sq])))
        #for sq in range(0, actual_x_length):    
        #    Ra2_squares[sq].setFill(color_rgb(int(255*state2[sq]), int(255*state2[sq]), int(255*state2[sq])))
        
    trainingX.append(j)
    trainingY.append(absolute_error)
    
    print(absolute_error)

    absolute_error = 0
    
    total_error = 0
    
    for step in range(0, x_testing_steps):
        
        h_of_x = testing_h_of_x_array[step]
        
        x = testing_array[step]
        
        state1 = sigmoid(np.dot(x, t1))
        state2 = sigmoid(np.dot(state1, t2))
        state3 = sigmoid(np.dot(state2, t3))
        
        error3 = (h_of_x - state3) #* sigmoid_output_to_derivative(state3)
        
        total_error = abs(np.sum(error3))
        absolute_error = total_error + absolute_error
        
    testingX.append(j)
    testingY.append(absolute_error)
    
    print(absolute_error)
    
    absolute_error = 0
    
    total_error = 0
  
#win.close()





plt.subplot(2, 1, 1)
plt.plot(trainingX, trainingY, 'o-')
plt.title('Training and testing error')
plt.ylabel('training_error')

plt.subplot(2, 1, 2)
plt.plot(testingX, testingY, '.-')
plt.xlabel('time (s)')
plt.ylabel('testing_error')

plt.show()

val = None
while val != "e":
    testing_array = []
    val = input("Enter the picture number: ") 
    image = img.imread("C://Users//Hotaru LSG//Desktop//python//MINST fashion//img//fashion%d.png" % int(val))
    testing_array.append(image.flatten())
    
    x = testing_array[0]
    
    # forward pass 
    state1 = sigmoid(np.dot(x, t1))
    state2 = sigmoid(np.dot(state1, t2))
    state3 = sigmoid(np.dot(state2, t3))
    
    print(return_test_result(state3));

