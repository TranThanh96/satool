import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import tensorflow as tf

def progress(x, total):
    sys.stdout.write("\r  done batch {} in total {}".format(x, total))
    sys.stdout.flush()
    return

def isEmpty(s):
    return not bool(s and s.strip())

def make_image_from_batch(X):
    '''
    this is document
    '''
    batch_size, h, w, c = X.shape
    no_col = int(np.ceil(np.sqrt(batch_size)))
    no_row = int(np.ceil(batch_size/no_col))
    output = np.zeros((int(no_row*h), int(no_col*w), c))
    for row in range(no_row):
        for col in range(no_col):
            if (row*no_col + col) == batch_size:
                break
            output[row*h:(row+1)*h,col*w:(col+1)*w] = X[row*no_col + col]
            
        if (row*no_col + col) == batch_size:
                break
    return np.squeeze(output)

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def one_hot(x, depth): 
    # define convert to one hot
    data = np.array(x)
    return np.eye(depth)[data.astype(np.int16)]

def convert_nguoc(labels):
    '''
    convert nguoc onehot tro laij kieu binh thuong
    '''
    no_classes = labels.shape[1]
    new_labels = np.zeros((labels.shape[0]))
    new_labels = np.array(np.where(labels[:,:]==1)[1])
    return new_labels

def block(inputs, filters, num_layers=2, kernel_size=[3,3], istraining=True, name=None):
    net=inputs
    with tf.variable_scope(name):
        for i in range(num_layers):
            with tf.variable_scope('{}_{}'.format(name, i+1)):
                net = tf.layers.conv2d(
                    net,
                    filters,
                    kernel_size,
                    padding='same',
                    activation=None,
                )
                net = tf.layers.batch_normalization(net, renorm=True, training=istraining)
                net = tf.nn.relu(net)
        with tf.variable_scope('{}_maxpool'.format(name)):
            net = tf.layers.max_pooling2d(net, [2,2], (2,2))
    return net

def load_graph(frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="model")
        return graph