import numpy as np
def my_knn(train_data_x,train_data_y,test_data_x,test_data_y,k):
    #Storing necessary values
    num_of_train_data = train_data_x.shape[0]
    num_of_test_data = test_data_x.shape[0]
    num_of_attr = train_data_x.shape[1]
    
    #creating prediction vars
    predictions_train = np.zeros(num_of_train_data)
    predictions_test = np.zeros(num_of_test_data)

    #For broadcasting
    train_data_reshaped = train_data_x.T.reshape([1,num_of_attr,num_of_train_data])
    
    #Distances
    distance_train = np.linalg.norm(train_data_x.reshape([num_of_train_data,num_of_attr,1]) - train_data_reshaped, axis=1)
    distance_test = np.linalg.norm(test_data_x.reshape([num_of_test_data,num_of_attr,1]) - train_data_reshaped, axis=1)

    #Choosing k nearest indices
    knn_indices_train = distance_train.argsort(axis=-1)[:,:k]
    knn_indices_test = distance_test.argsort(axis=-1)[:,:k]

    #getting k-nn labels
    k_labels_train = train_data_y[knn_indices_train]
    k_labels_test = train_data_y[knn_indices_test]

    #Counting label counts of 1
    predictions_train_mask = np.sum(k_labels_train==1,axis=1)
    predictions_test_mask = np.sum(k_labels_test==1,axis=1)

    #Assigning predictions
    predictions_train[predictions_train_mask>=k/2] = 1
    predictions_test[predictions_test_mask>=k/2] = 1

    #Calculating Error Rates
    acc_train = float(np.sum(predictions_train!=train_data_y)/len(predictions_train))
    acc_test = float(np.sum(predictions_test!=test_data_y)/len(predictions_test))

    return acc_train,acc_test
