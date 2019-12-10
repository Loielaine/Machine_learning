import scipy.io as sio
import numpy as np
import copy

bodyfat = sio.loadmat('bodyfat_data.mat')
X = bodyfat['X']
y = bodyfat['y']

x_train = X[:150,]
x_test = X[150:X.shape[0],]
y_train = y[:150,]
y_test = y[150:y.shape[0],]
np.random.seed(0)

# Create the model architecture
input_dim = 2
nodes_hidden_1 = 64
nodes_hidden_2 = 16
output_dim = 1

W = {
    'W_input_h1': np.random.randn(input_dim, nodes_hidden_1),
    'b_input_h1': np.zeros(nodes_hidden_1),

    'W_h1_h2': np.random.randn(nodes_hidden_1, nodes_hidden_2),
    'b_h1_h2': np.zeros(nodes_hidden_2),

    'W_h2_output': np.random.randn(nodes_hidden_2, output_dim),
    'b_h2_output': np.zeros(output_dim)
}

def relu(x):
    return np.maximum(x, 0)

def reluDerivative(x):
    y = copy.deepcopy(x)
    y[y<=0] = 0
    y[y>0] = 1
    return y


def mean_square_loss(t_hat, t):
    return np.mean(np.square(t_hat-t))

def predict(W,X):
    z_h1 = np.dot(X, W['W_input_h1']) + W['b_input_h1']
    a_h1 = relu(z_h1)

    z_h2 = np.dot(a_h1, W['W_h1_h2']) + W['b_h1_h2']
    a_h2 = relu(z_h2)
    prediction = np.dot(a_h2, W['W_h2_output']) + W['b_h2_output']
    return prediction

def early_stopping(n,lst1):
    compare = []
    for i in range(1,n+1):
        compare.append(lst1[(-1)*i]<lst1[(-1*i)-1])
    return compare
    

# Train using Gradient Descent
learning_rate = 1e-07
train_loss_lst = [1e+15,5e+14,1e+14,5e+13,1e+13,5e+12]
test_loss_lst = [1e+15,5e+14,1e+14,5e+13,1e+13,5e+12]
epoch=0


while sum(early_stopping(5,test_loss_lst))>=1 or abs(test_loss_lst[-1]-test_loss_lst[-2])>1e-07:    
    ## forward pass
    z_h1 = np.dot(x_train, W['W_input_h1']) + W['b_input_h1']
    a_h1 = relu(z_h1)

    z_h2 = np.dot(a_h1, W['W_h1_h2']) + W['b_h1_h2']
    a_h2 = relu(z_h2)
    forward_output = np.dot(a_h2, W['W_h2_output']) + W['b_h2_output']

    # Backward Pass
    ## the third layer
    delta_y = 2*(forward_output - y_train)
    print(delta_y.shape)
    
    N = len(delta_y)
    #print(N)
    W_h2_output_grad = np.dot(a_h2.T,delta_y).reshape((nodes_hidden_2, output_dim))
    b_h2_output_grad = np.sum(delta_y)

    ## the second layer
    delta_h2 = np.dot(delta_y,W['W_h2_output'].T)*reluDerivative(z_h2)
    print(delta_h2.shape)
    b_h1_h2_grad = np.sum(delta_h2, axis=0)
    
    W_h1_h2_grad = np.dot(a_h1.T,delta_h2)
    
    ## the first layer
    delta_h1 = np.dot(delta_h2,W['W_h1_h2'].T)*reluDerivative(z_h1)
    print(delta_h1.shape)
    b_input_h1_grad = np.sum(delta_h1, axis=0)
    
    W_input_h1_grad = np.dot(x_train.T,delta_h1)

    # Gradient Descent
    W['W_h2_output'] -= (learning_rate * W_h2_output_grad)/N
    W['b_h2_output'] -= (learning_rate * b_h2_output_grad)/N

    W['W_h1_h2'] -= (learning_rate * W_h1_h2_grad)/N
    W['b_h1_h2'] -= (learning_rate * b_h1_h2_grad)/N

    W['W_input_h1'] -= (learning_rate * W_input_h1_grad)/N
    W['b_input_h1'] -= (learning_rate * b_input_h1_grad)/N

    y_predict = predict(W,x_test)
    train_loss = mean_square_loss(forward_output, y_train)
    test_loss = mean_square_loss(y_predict, y_test)
    train_loss_lst.append(train_loss)
    test_loss_lst.append(test_loss)
    
    epoch = epoch+1
    #if epoch % 100 == 0:
        #print("Train loss after epoch ", epoch, " : ", train_loss,"\nTest loss after epoch ", epoch, " : ", test_loss,)


train_error = train_loss_lst[-1]
print ('the mean squared error on the train inputs is ', train_error)

test_error = test_loss_lst[-1]
print ('the mean squared error on the test inputs is ', test_error)
