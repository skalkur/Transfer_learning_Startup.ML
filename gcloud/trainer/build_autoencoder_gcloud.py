'''
Python script which is the trainer task to train the CAE on Google Cloud 
Platform.

Requires Google Cloud Platform account, Training data and scripts are to be 
placed inside Cloud Storage Bucket
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from datetime import datetime
import logging
import argparse, os
from StringIO import StringIO


tf.reset_default_graph()

# Batch size to be inputted
batch_size=500
# Filter window size for every layer
filter_size=[4,4,4,4]


def iterate_batches(x_in, batch_size):
    '''
    Function to randomly shuffle and yield batches for training
    
    Args: Unlabeled images and batch size
    Returns: Batch of images, shuffled
    
    '''
    new_perm=np.random.permutation(range(len(x_in)))
    epoch_images=x_in[new_perm, ...]
   
    current_batch_id=0
    while current_batch_id < len(x_in):
        end=min(current_batch_id+batch_size,len(x_in))
        batch_images={'images': epoch_images[current_batch_id:end]}
        current_batch_id+=batch_size
        yield batch_images['images']
    
                                 
def train_model(train_file='../Unlabeled_X.npy', job_dir='./tmp/autoencoder', \
                output_dir='../output/', learning_rate=0.001, n_epochs=300, **args):
    
    '''
    Function to train the CAE by taking in batches of images. Requires
    arguments to be passed while initiating the job on GCP. Saves the model in 
    the Bucket every 10 epochs
    
    Args: Location of Training data (Cloud Storage Bucket), job-directory to 
    output logs of the job, learning rate and number of iterations for training
    
    
    '''
    logs_path=job_dir+'/logs/'+datetime.now().isoformat()
    output_file=os.path.join(output_dir,'saved-autoencoder-model')
    logging.info('_____________________')
    logging.info('Using Train File located at {}'.format(train_file))
    logging.info('Using Logs_path located at {}'.format(logs_path))
    logging.info('_____________________')
    file_string=StringIO(file_io.read_file_to_string(train_file))
    with tf.Graph().as_default():
        sess=tf.InteractiveSession()
        X_input=np.load(file_string)
        idx=range(len(X_input))
        
        # Shuffle Data
        rand_idxs=np.random.permutation(idx)
        X_input=X_input[rand_idxs,...]


        logging.info('Unlabeled Dataset loaded')
        
        features=X_input.shape[1]

        # Number of filters for every layer
        n_filters=[64,64,64,64]
    
        # Create placeholder for image tensor
        X=tf.placeholder(tf.float32, shape=[None, features], name='X')
        X_image_tensor=tf.reshape(X, [-1, 96, 96, 3])
    
        currentInput=X_image_tensor
        n_input=currentInput.get_shape().as_list()[3]
        Ws=[]
        shapes=[]
        
        # Build a 4-layer convolutional encoder model by appending weights
        # dimensions for decoder
        for layer, output_size in enumerate(n_filters):
            with tf.variable_scope("encoder/layer_{}".format(layer)):
                shapes.append(currentInput.get_shape().as_list())
                W=tf.get_variable(name='W', shape=[filter_size[layer],\
                                                   filter_size[layer],\
                                                    n_input, output_size],\
                                                    initializer=\
                                                    tf.random_normal_initializer(mean=0.0,stddev=0.01))
                b=tf.get_variable(name='b', shape=[output_size], initializer=\
                                  tf.constant_initializer([0]))
                h=(tf.add(tf.nn.conv2d(currentInput, W, strides=[1,2,2,1],\
                               padding='SAME'),b))
                h=tf.nn.relu(h,name='h')
                currentInput=h
                Ws.append(W)
                n_input=output_size
        
        # Reverse weights matrix and shape matrix for decoder
        Ws.reverse()
        shapes.reverse()
        n_filters.reverse()
        n_filters=n_filters[1:]+[3]
        
        # Decoder for reconstruction of images
        for layer, output_size in enumerate(shapes):
            with tf.variable_scope('decoder/layer_{}'.format(layer)):
                W=Ws[layer]
                b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
                output_shape=tf.stack([tf.shape(X)[0], \
                                       output_size[1],output_size[2],output_size[3]])
                h=(tf.add(tf.nn.conv2d_transpose(currentInput, W, output_shape=output_shape, \
                                         strides=[1,2,2,1],padding='SAME'),b))
                h=tf.nn.relu(h,name='h')
                currentInput=h
                
        # Final Placeholder        
        Y=currentInput
        Y=tf.reshape(Y,[-1,96*96*3])
        
        cost=tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X,Y),1))
        optimizer=tf.train.AdamOptimizer(float(learning_rate)).minimize(cost)
        
        # Initiate Saver Instance
        saver=tf.train.Saver()
        
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        
        # Start training
        for i in range(int(n_epochs)):
            for batch_img in iterate_batches(X_input, batch_size=batch_size):
                sess.run(optimizer,feed_dict={X:batch_img})
            # Every 10 epochs, report performance and save model graph and weights
            if i%10==0:    
                logging.info('Epoch:{0}, Cost={1}'.format(i, \
                             sess.run(cost, feed_dict={X: batch_img})))
                saver.save(sess, output_file, global_step=0)
                logging.info('Model Saved')

                
                
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-file', help='GCS or local paths to train data',\
                        required=True)
    parser.add_argument('--job-dir', help='GCS location to write \
    checkpoints and export models', required=True)
    parser.add_argument('--output_dir', help='GCS location \
    to write model', required=True)
    parser.add_argument('--learning-rate', help='Learning Rate', required=True)
    parser.add_argument('--n-epochs', help='Number of epochs', required=True)
    
    args=parser.parse_args()
    arguments=args.__dict__
    
    train_model(**arguments)
