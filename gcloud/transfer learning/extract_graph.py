import tensorflow as tf
import numpy as np
from get_data import load_stl_data

tf.reset_default_graph()

model_meta='../output/saved-autoencoder-model-0.meta'
model='../output/saved-autoencoder-model-0'

X_train, Y_train, X_test, Y_test=load_stl_data(one_hot=True)

X_train=np.reshape(X_train,[-1,96*96*3])
X_test=np.reshape(X_test,[-1,96*96*3])

def create_graph(sess):
    
    '''
    Function to extract Graph and model from the trained CAE.
    
    '''
    
    saver=tf.train.import_meta_graph(model_meta)
    saver.restore(sess, model)
        
        
def extract_features(sess, X_input):
    
    '''
    Function to extract features for a given batch of images by
    passing it through CAE model until the layer 3 of ReLu of encoder to get bottlenecks
    
    Args: Current Session, Images
    Returns: Array of 4608 features extracted for every image by Inception
    '''
    
    n_train=X_input.shape[0]
    encoder_relu=sess.graph.get_tensor_by_name('encoder/layer_3/h:0')
    x_encoder=[]
    for i in range(n_train):
        print ("Iteration: "+str(i))
        features=sess.run(encoder_relu,{'X:0':np.reshape(X_input[i,:],[1,-1])})
        x_encoder.append(np.squeeze(features))
    return np.array(x_encoder)
 
def save_serialized(filename, X):
    
    '''
    Function to save the bottlenecked features as a Numpy array
    Args: Desired filename to save the bottlenecks, Bottlenecks
    '''
    features=extract_features(sess, X)
    np.save(filename, features)
    print(filename+' generated')
    
def generate_bottlenecks():
    
    save_serialized('./X_train_bottleneck.npy', X_train)
    save_serialized('./X_test_bottleneck.npy', X_test)
    np.save('./Y_train_bottleneck.npy', Y_train)
    np.save('./Y_test_bottleneck.npy', Y_test)
  
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

def extract_encoder_features(sess, X_input):
    
    '''
    Function to extract features from every encoder layer
    '''
    features=[]
    for i in range(4):
        encoder_relu=sess.graph.get_tensor_by_name('encoder/layer_'+str(i)+'/h:0')
        features.append(sess.run(encoder_relu, feed_dict={'X:0':X_input}))
    return features

    
# Uncomment and run to generate bottlenecks

#sess=tf.InteractiveSession()
#create_graph(sess)
#generate_bottlenecks()


