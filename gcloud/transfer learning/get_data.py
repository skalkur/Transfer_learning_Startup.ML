import numpy as np


def load_stl_10_img(filename):
    with open(filename, 'rb') as f:
        X=np.fromfile(f, dtype=np.uint8)
        X=X.reshape(-1,3,96,96).transpose(0,2,3,1)        
        return X
        
def load_stl_10_label(filename):
    with open(filename, 'rb') as f:
        Y=np.fromfile(f, dtype=np.uint8)
        return Y


def load_stl_unlabeled(filename):
    with open(filename, 'rb') as f:
        return np.fromfile(f, dtype=np.uint8)

           
def load_stl_data(one_hot=False):
    
    X_train=load_stl_10_img('../../dataset/train_X.bin')
    X_test=load_stl_10_img('../../dataset/test_X.bin')
    Y_train=load_stl_10_label('../../dataset/train_y.bin')
    Y_test=load_stl_10_label('../../dataset/test_y.bin')
    
    
    if one_hot:
        Y_train=np.eye(10)[Y_train-1]
        Y_test=np.eye(10)[Y_test-1]
    return X_train, Y_train, X_test, Y_test
    
#    
#X=load_stl_unlabeled('../../dataset/unlabeled_X.bin')
#
#X=X.reshape(-1,3*96*96).transpose(0,1)
#np.save('Unlabeled_X.npy',X)
#print ('File saved with shape:{0}'.format(X.shape))
#    
