import numpy as np
from graphics import *
import time
import matplotlib.image as img

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)




x_length = 625  # dimension of the image array flattened
x_steps = 2     # how many images
x_width = 25    # for display purposes, what is the width of the picture?

layer1_length = 50   # dimension of the middle (hidden) layer

actual_x_length = 3   # dimension of the output

x_array = np.zeros(x_length)
x = np.zeros((x_length))




""" read the 1st image, then loop through the other images, putting them into into a RGB Array"""

image = img.imread("C://Users//Hotaru LSG//Desktop//1.jpg")
Rimage = image[:,:,0]
Gimage = image[:,:,1]
Bimage = image[:,:,2]
RGB_array = np.vstack((Rimage.flatten(), Gimage.flatten(), Bimage.flatten()))

for pic in range(2, 3):
    image = img.imread("C://Users//Hotaru LSG//Desktop//%d.jpg" % pic)
    Rimage = image[:,:,0]
    Gimage = image[:,:,1]
    Bimage = image[:,:,2]
    RGB_array = np.vstack((RGB_array, Rimage.flatten(), Gimage.flatten(), Bimage.flatten()))


t1 = list()
t1_update = list()
state1 = list()
t2 = list()
t2_update = list()
t2_temp = list()
state2 = list()
error1 = list()
error2 = list()
for w in range(0, 3):
    t1.append(np.random.uniform(-1,1,(x_length, layer1_length)))
    t1_update.append(np.zeros((x_length, layer1_length)))
    state1.append(np.zeros((layer1_length)))
    t2.append(np.random.uniform(-1,1,(layer1_length, actual_x_length)))
    t2_update.append(np.zeros((layer1_length, actual_x_length)))
    t2_temp.append(np.zeros((layer1_length, actual_x_length)))
    state2.append(np.zeros((actual_x_length)))
    error1.append(np.zeros((layer1_length)))
    error2.append(np.zeros((actual_x_length)))





h_of_x_array = np.array([[1,0,0],[0,0,1]])
h_of_x = np.zeros((actual_x_length))

win = GraphWin("Network", 600,800)

def make_a_column(number_of_rows, x_coord, y_coord, which_layer, x_width, border_width):
    "This makes a single column, which is one layer in the NN"
    xc = list()
    node = 0
    original_x = x_coord
    for row in range(1, number_of_rows+1):
        pt = Point(x_coord, y_coord)
        pt2 = Point(x_coord+5, y_coord+5)
        rect = Rectangle(pt, pt2)
        rect.draw(win)
        rect.setOutline("black")
        rect.setWidth(border_width)
        rect.setFill(color_rgb(int(255*which_layer[node]), 0, 0))
        xc.append(rect)
        node += 1
        x_coord = x_coord + 5
        if row % x_width == 0:
            y_coord = y_coord + 5
            x_coord = original_x
    return xc


RGB_squares = make_a_column(x_length, 10, 10, x, x_width, 0)
desired_squares = make_a_column(actual_x_length, 10, 170, h_of_x, x_width, 1)

R_squares = make_a_column(x_length, 10, 200, x, x_width, 0)
G_squares = make_a_column(x_length, 200, 200, x, x_width, 0)
B_squares = make_a_column(x_length, 400, 200, x, x_width, 0)


Ra1_squares = make_a_column(layer1_length, 10, 500, state1[0], x_width, 1)
Ra2_squares = make_a_column(actual_x_length, 10, 650, state2[0], x_width, 1)
Ga1_squares = make_a_column(layer1_length, 200, 500, state1[0], x_width, 1)
Ga2_squares = make_a_column(actual_x_length, 200, 650, state2[0], x_width, 1)
Ba1_squares = make_a_column(layer1_length, 400, 500, state1[0], x_width, 1)
Ba2_squares = make_a_column(actual_x_length, 400, 650, state2[0], x_width, 1)





total_error = 0
start = 0
col = 0

for j in range(200):
    
    R = 0
    G = 1
    B = 2
    
    for step in range(0, x_steps):
        
        h_of_x = h_of_x_array[step]
        
        
        # put a full-color picture of the current training example at the top of the screen, and put
        # the desired output at the bottom
        for sq in range(0, x_length):
            RGB_squares[sq].setFill(color_rgb(RGB_array[R][sq], RGB_array[G][sq], RGB_array[B][sq]))
            R_squares[sq].setFill(color_rgb(RGB_array[R][sq], RGB_array[R][sq], RGB_array[R][sq]))
            G_squares[sq].setFill(color_rgb(RGB_array[G][sq], RGB_array[G][sq], RGB_array[G][sq]))
            B_squares[sq].setFill(color_rgb(RGB_array[B][sq], RGB_array[B][sq], RGB_array[B][sq]))
        for sq in range(0, len(h_of_x)):
            desired_squares[sq].setFill(color_rgb(int(255*h_of_x[sq]), int(255*h_of_x[sq]), int(255*h_of_x[sq])))
            

        
        for color in range(start, start+3):
            x = RGB_array[color]
            
            # forward pass 
            state1[col] = sigmoid(np.dot(x, t1[col]))
            state2[col] = sigmoid(np.dot(state1[col], t2[col]))
            
        
            # errors and backward pass
            error2[col] = (h_of_x - state2[col]) * sigmoid_output_to_derivative(state2[col])
            t2_temp[col] = t2[col] + (state1[col].reshape(-1,1)*(error2[col]))
            error1[col] = (np.dot(error2[col],t2_temp[col].T)) * sigmoid_output_to_derivative(state1[col])
            t2_update[col] += (state1[col].reshape(-1,1)*(error2[col]))
            t1_update[col] += (x.reshape(-1,1)*(error1[col]))
            
    
            # change node colors depending on new values

            
            time.sleep(.5)
            
            # theta updates
            t1[col] += t1_update[col] 
            t2[col] += t2_update[col]  
        
            t1_update[col] *= 0
            t2_update[col] *= 0
            print(state2[col])
            print(h_of_x)
            print("")
            col = col + 1
            
            # put state information into the hidden layer images, and state information into the final layer images
            for sq in range(0, layer1_length):
                Ra1_squares[sq].setFill(color_rgb(int(255*state1[0][sq]), int(255*state1[0][sq]), int(255*state1[0][sq])))
                Ga1_squares[sq].setFill(color_rgb(int(255*state1[1][sq]), int(255*state1[1][sq]), int(255*state1[1][sq])))
                Ba1_squares[sq].setFill(color_rgb(int(255*state1[2][sq]), int(255*state1[2][sq]), int(255*state1[2][sq])))
            for sq in range(0, actual_x_length):    
                Ra2_squares[sq].setFill(color_rgb(int(255*state2[0][sq]), int(255*state2[0][sq]), int(255*state2[0][sq])))
                Ga2_squares[sq].setFill(color_rgb(int(255*state2[1][sq]), int(255*state2[1][sq]), int(255*state2[1][sq])))
                Ba2_squares[sq].setFill(color_rgb(int(255*state2[2][sq]), int(255*state2[2][sq]), int(255*state2[2][sq])))
            
        R = R + 3
        G = G + 3
        B = B + 3
            
        start = start + 3
        col = 0
        time.sleep(1)
    start = 0


time.sleep(5)
win.close()
    
