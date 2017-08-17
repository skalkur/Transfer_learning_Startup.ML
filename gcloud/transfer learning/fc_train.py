import tensorflow as tf
import numpy as np
from extract_graph import create_graph, iterate_batches
from retrain import add_final_training_layer, ensure_port, evaluation_step

tf.reset_default_graph()


def load_bottleneck_data():
    
    '''
    Load saved numpy arrays of images which are serialized after bottlenecked
    by passing it through Inception model
    
    Returns: Numpy arrays of the images and labels
    '''
    X_test_file='X_test_bottleneck.npy'
    Y_test_file='Y_test_bottleneck.npy'
    X_train_file='X_train_bottleneck.npy'
    Y_train_file='Y_train_bottleneck.npy'
  
    return np.load(X_train_file),np.load(Y_train_file),\
                                np.load(X_test_file),np.load(Y_test_file)
         
     
     
def trainer(sess, X_input, Y_input, X_test, Y_test):
    
    '''
    Function to train the FC model with a Softmax activation for output layer
    
    Args: TensorFlow Session, Bottlenecked Images for training and testing
    and corresponding labels
    '''
    ground_truth_tensor_name='ground_truth'
    
    # Define Batch size
    mini_batch_size=250
    n_train=X_input.shape[0]

    create_graph(sess)
    graph=tf.get_default_graph()
    
    # Get the train op and loss function
    train_step,cross_entropy=add_final_training_layer\
    (n_classes, final_tensor_name, ground_truth_tensor_name, learning_rate)
    # Intiliaze all variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # Get evaluation tensor
    eval_step=evaluation_step(graph, \
    'fc/layer2/'+final_tensor_name, ground_truth_tensor_name)

    # Get tensors for Input and Output    
    bottleneck_input=graph.get_tensor_by_name(ensure_port('BottleneckInput'))
    Y=graph.get_tensor_by_name(ensure_port(ground_truth_tensor_name))
    
    # Define number of epochs
    epochs=600
    
    # Perform training for number of epochs defined
    for epoch in range(epochs):
        
        # Shuffle the examples
        shuffle=np.random.permutation(n_train)
        
        shuffle_X=X_input[shuffle,:]
        shuffle_Y=Y_input[shuffle]
        
        # Perform batch training
        for Xi, Yi in iterate_batches(shuffle_X, shuffle_Y, mini_batch_size):
            sess.run(train_step, feed_dict={bottleneck_input:Xi, Y:Yi})
        
        # Print out model's performance after every epoch
        train_accuracy, train_cross_entropy=\
        sess.run([eval_step,cross_entropy], \
                 feed_dict={bottleneck_input:X_input, Y:Y_input})
        print ("Epoch %d: Train accuracy:%0.2f, Cross Entropy:%0.2f"\
               %(epoch,train_accuracy*100,train_cross_entropy))
                
    # Get the test accuracy after training is complete        
    test_accuracy=sess.run(eval_step, \
                           feed_dict={bottleneck_input:X_test, Y:Y_test})
    print('Final Test Accuracy for CAE Transfer Learning:%0.2f' %(test_accuracy*100))   
    
    
n_classes=10

X_train, Y_train, X_test, Y_test= load_bottleneck_data()
X_train=np.reshape(X_train,[-1,6*6*64])
X_test=np.reshape(X_test,[-1,6*6*64])
final_tensor_name='final_result'
learning_rate=0.001

# Create TensorFlow session and train model 
sess=tf.InteractiveSession()
trainer(sess, X_train, Y_train, X_test, Y_test)