import tensorflow as tf
import numpy as np
from get_data import load_stl_data
from extract_graph import create_graph, batch_pool3_features


BOTTLENECK_TENSOR_NAME='pool_3'
BOTTLENECK_TENSOR_SIZE=2048

def bottleneck_pool3(sess,X,filename):
    '''
    Function to generate bottlenecks images and saving the serialized images
    
    Args: TensorFlow session, Image array, Filename to save the serialized images
    '''

    x_pool3=batch_pool3_features(sess=sess,X_input=X)
    np.save(filename, x_pool3)
    print (filename+" generated")
    
def bottleneck_data(sess):
    
    '''
    Function to load STL data and process them to bottleneck them
    
    Args: TensorFlow session
    '''
    X_train, Y_train, X_test, Y_test=load_stl_data(one_hot=True)
    bottleneck_pool3(sess,X_train, './X_train.npy')
    bottleneck_pool3(sess,X_test, './X_test.npy')
    np.save('./Y_train.npy',Y_train)
    np.save('./Y_test.npy',Y_test)

def ensure_port(tensor_name):
    
    '''
    Function to add a port number to the tensor if it is missing.
    This is required to reference a layer in the Inception graph
    
    Arg: Tensor name to add the port to
    
    Returns: Tensor name with the port attached
    '''
    if ':' not in tensor_name:
        tensor_name_port=tensor_name+':0'
    else:
        tensor_name_port=tensor_name
    
    return tensor_name_port
    
    
def add_final_training_layer(class_count, final_tensor_name,\
                             ground_truth_tensor_name, learning_rate=1e-3):
    
    '''
    Function to define the FC, Softmax classifier model to Classify the serialized
    images. Has Gradient Descent Optimizer.
    Includes Dropout layers and a 2048-1024-512-10 network
    
    Args: No. of classes, final tensor name of the FC network, 
    Ground Truth Tensor name, Learning rate for Optimizer
    
    Returns: Train Op and Cost of the model
    '''
    layers=[1024, 512, 10]
    keep_prob=0.75
    bottleneck_input=tf.placeholder(tf.float32,\
                                    shape=[None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInput')
    currentInput=bottleneck_input
    n_input=BOTTLENECK_TENSOR_SIZE
    for layer, output_size in enumerate(layers):
        with tf.variable_scope('fc/layer{}'.format(layer)):
            W=tf.get_variable(name='W', shape=[n_input, output_size], \
                              initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01))
            b=tf.get_variable(name='b',shape=[output_size],\
                              initializer=tf.constant_initializer([0]))
            h=tf.matmul(currentInput,W)+b
            n_input=output_size
            if output_size!=layers[2]:
                h=tf.nn.tanh(h,name='h')
            else:
                final_tensor=tf.nn.softmax(h, name=final_tensor_name)
            h=tf.nn.dropout(h,keep_prob)
            currentInput=h

    Y=tf.placeholder(tf.float32, shape=[None,class_count],\
                     name=ground_truth_tensor_name)
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=Y)
    cost=tf.reduce_mean(cross_entropy)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    return train_step, cost
    
def evaluation_step(graph, final_tensor_name, ground_truth_tensor_name):
    
    '''
    Function to evaluate the performance of the model by calculating the 
    accuracy of prediction
    
    Args: Final Tensor and Ground Truth Tensor Name, TensorFlow Graph
    
    Return: Evaluation Tensor
    '''
    result_tensor=graph.get_tensor_by_name(ensure_port(final_tensor_name))
    Y_tensor=graph.get_tensor_by_name(ensure_port(ground_truth_tensor_name))
    correct_pred=tf.equal(tf.argmax(result_tensor,1),tf.argmax(Y_tensor,1))
    
    eval_step=tf.reduce_mean(tf.cast(correct_pred,'float'))
    return eval_step

    
# Uncomment this section to create Inception Graph and 
# serialize the images
     
'''    
sess=tf.InteractiveSession()
g=create_graph()
bottleneck_data(sess)

'''