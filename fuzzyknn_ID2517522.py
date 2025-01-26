#Dogukan Arasli 2517522
import numpy as np
def my_fuzzyknn(train_data_x,train_data_y,test_data_x,test_data_y,k,m):
    epsilon = 1e-10
    
    num_of_train_data = train_data_x.shape[0]
    num_of_test_data = test_data_x.shape[0]
    num_of_attr = train_data_x.shape[1]
    
    predictions_train = np.zeros(num_of_train_data)
    predictions_test = np.zeros(num_of_test_data)

    train_data_reshaped = train_data_x.T.reshape([1,num_of_attr,num_of_train_data])

    distance_train = np.linalg.norm(train_data_x.reshape([num_of_train_data,num_of_attr,1]) - train_data_reshaped, axis=1)
    distance_test = np.linalg.norm(test_data_x.reshape([num_of_test_data,num_of_attr,1]) - train_data_reshaped, axis=1)

    knn_indices_train = distance_train.argsort(axis=-1)[:,:k]
    knn_indices_test = distance_test.argsort(axis=-1)[:,:k]

    k_labels_train = train_data_y[knn_indices_train]
    k_labels_test = train_data_y[knn_indices_test]

    knn_indices_train += np.arange(num_of_train_data).reshape([num_of_train_data,1]) * num_of_train_data
    knn_indices_test += np.arange(num_of_test_data).reshape([num_of_test_data,1]) * num_of_train_data

    weights_train = distance_train.flat[knn_indices_train]
    weights_test = distance_test.flat[knn_indices_test]

    weights_train[weights_train == 0] = epsilon
    weights_test[weights_test == 0] = epsilon

    train_label1_score = np.sum(((k_labels_train==1)*(1/weights_train**(2/(m-1)))),axis=1) / np.sum(1/weights_train**(2/(m-1)),axis = 1)
    train_label0_score = np.sum(((k_labels_train==0)*(1/weights_train**(2/(m-1)))),axis=1) / np.sum(1/weights_train**(2/(m-1)),axis = 1)

    test_label1_score = np.sum(((k_labels_test==1)*(1/weights_test**(2/(m-1)))),axis=1) / np.sum(1/weights_test**(2/(m-1)),axis = 1)
    test_label0_score = np.sum(((k_labels_test==0)*(1/weights_test**(2/(m-1)))),axis=1) / np.sum(1/weights_test**(2/(m-1)),axis = 1)

    predictions_train[train_label1_score >= train_label0_score] = 1
    predictions_test[test_label1_score >= test_label0_score] = 1

    acc_train = float(np.sum(predictions_train!=train_data_y)/len(predictions_train))
    acc_test = float(np.sum(predictions_test!=test_data_y)/len(predictions_test))

    return acc_train,acc_test