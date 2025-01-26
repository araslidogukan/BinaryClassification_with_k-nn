#Dogukan Arasli 2517522
import numpy as np
def my_train_test_split(data_x,data_y,train_to_test_ratio = 0.8,number_of_folds = None):
    #This function both includes train/test split and folding scheme
    num_of_data = data_x.shape[0]

    for i in range(num_of_data-1):
        # Fisher - Yates Shuffles (modern variant)
        swap_idx = np.random.randint(low=i,high=num_of_data)
        placer_x = data_x[i,:].copy()
        placer_y = data_y[i].copy()
        data_x[i,:] = data_x[swap_idx,:].copy()
        data_y[i] = data_y[swap_idx].copy()
        data_x[swap_idx,:] = placer_x
        data_y[swap_idx] = placer_y

    train_size = round(num_of_data * train_to_test_ratio)

    if number_of_folds is None: #If no folding
        train_x = data_x[:train_size,:]
        train_y = data_y[:train_size]

        test_x = data_x[train_size:,:]
        test_y = data_y[train_size:]

        return train_x, train_y, test_x, test_y
    
    else : #If folding
        folds_x = []
        folds_y = []
        fold_size = round(num_of_data / number_of_folds)

        for i in range(number_of_folds):
            folds_x.append(data_x[i*fold_size:(i+1)*fold_size,:])
            folds_y.append(data_y[i*fold_size:(i+1)*fold_size])

        return folds_x, folds_y