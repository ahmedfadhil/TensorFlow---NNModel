'''
input > weight > hidden layer1 (activation function) > weight > hidden layer 2 (activation function) > weight
  > output 
  
  compare the output to the cost function
   then use optimization function (optimizer) > minimize the cost (AdamOptimizer, SGD, AdaGrad)
   
   backpropagation
   feed forward +backprop = epoch
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0]
2 = [0,0,1,0,0,0,0]
3 = [0,0,0,1,0,0,0]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# Setting up the computation graph
def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal(n_nodes_hl1))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal(n_nodes_hl1))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal(n_nodes_hl1))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal(n_classes))}

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases'])

    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases'])

    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases'])

    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output['biases']

    return output



# Training the neural network model











