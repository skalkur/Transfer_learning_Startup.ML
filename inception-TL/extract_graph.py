from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np

model='../inception/classify_image_graph_def.pb'

def create_graph():
    
    '''
    Function to extract GraphDef of Inception model.
    
    Returns: Extracted GraphDef
    
    '''    
    with tf.Session() as sess:
        with gfile.FastGFile(model,'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _=tf.import_graph_def(graph_def,name='')
            
    return sess.graph

def batch_pool3_features(sess,X_input):
    
    '''
    Function to extract features for a given batch of images by
    passing it through Inception model until pool_3 layer to get bottlenecks
    
    Args: Current Session, Batch of Images of size:batch_sizex96x96x3
    Returns: Array of 2048 features extracted for every image by Inception
    '''
    n_train=X_input.shape[0]
    pool3=sess.graph.get_tensor_by_name('pool_3:0')
    x_pool3=[]
    for i in range(n_train):
        print ("Iteration: "+str(i))
        features=sess.run(pool3,{'DecodeJpeg:0':X_input[i,:]})
        x_pool3.append(np.squeeze(features))
    return np.array(x_pool3)
    
def iterate_batches(X_input, Y_input, batch_size):
    
    '''
    Function to parse a set batch of images and labels to the training fn.
    
    Args: Images, Labels, batch_size
    Returns: batch_size number of images and labels
    '''
    n_train=X_input.shape[0]
    for i in range(0,n_train, batch_size):
        yield X_input[i:min(i+batch_size,n_train)], \
                      Y_input[i:min(i+batch_size,n_train)]
                      
                 