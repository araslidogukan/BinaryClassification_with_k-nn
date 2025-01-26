import numpy as np
def my_modified_rnn(train_data_x,train_data_y,test_data_x,test_data_y,r):
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

    for i in range(num_of_train_data):
        if not np.any(distance_train[i,:] < r):
            predictions_train[i] = 0
            continue
        distance_to_neighbors = distance_train[i,distance_train[i,:] < r].copy()
        labels_neighbors = train_data_y[distance_train[i,:] < r].copy()

        distance_to_neighbors = distance_to_neighbors[distance_to_neighbors.argsort()]
        labels_neighbors = labels_neighbors[distance_to_neighbors.argsort()]

        weights = np.empty_like(distance_to_neighbors)
        weights[1:] = (distance_to_neighbors[-1] - distance_to_neighbors[1:]) / (distance_to_neighbors[-1] - distance_to_neighbors[0])
        weights[0] = 1

        train_label1_score = np.sum(weights * (labels_neighbors == 1))
        train_label0_score = np.sum(weights * (labels_neighbors == 0))
        
        if train_label1_score >= train_label0_score:
            predictions_train[i] = 1

    for i in range(num_of_test_data):
        if not np.any(distance_test[i,:] < r):
            predictions_test[i] = 0
            continue
        distance_to_neighbors = distance_test[i,distance_test[i,:] < r].copy()
        labels_neighbors = train_data_y[distance_test[i,:] < r].copy()

        neigbour_idx = distance_to_neighbors.argsort()
        distance_to_neighbors = distance_to_neighbors[neigbour_idx]
        labels_neighbors = labels_neighbors[neigbour_idx]

        weights = np.empty_like(distance_to_neighbors)
        weights[1:] = (distance_to_neighbors[-1] - distance_to_neighbors[1:]) / (distance_to_neighbors[-1] - distance_to_neighbors[0])
        weights[0] = 1

        test_label1_score = np.sum(weights * (labels_neighbors == 1))
        test_label0_score = np.sum(weights * (labels_neighbors == 0))

        if test_label1_score >= test_label0_score:
            predictions_test[i] = 1

    acc_train = float(np.sum(predictions_train!=train_data_y)/len(predictions_train))
    acc_test = float(np.sum(predictions_test!=test_data_y)/len(predictions_test))

    return acc_train,acc_test
