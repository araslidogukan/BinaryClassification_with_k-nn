#Dogukan Arasli 2517522
import numpy as np
def my_min_max_scaler(train,test):
    num_of_attr = train.shape[1]

    #Storing max and min of the train data
    maxs = np.max(train,axis=0).reshape([1,num_of_attr])
    mins = np.min(train,axis=0).reshape([1,num_of_attr])

    #Rescaling
    train = (train-mins) / (maxs-mins)
    test = (test-mins) / (maxs-mins)

    return train,test