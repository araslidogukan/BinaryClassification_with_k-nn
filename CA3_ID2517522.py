#Dogukan Arasli 2517522
import pandas as pd
import numpy as np
from mknn_ID2517522 import my_mknn
from partition_ID2517522 import my_train_test_split
from MinMaxScaler_ID2517522 import my_min_max_scaler
from fuzzyknn_ID2517522 import my_fuzzyknn
from rnn_ID2517522 import my_rnn
from fuzzyknn_cont_ID2517522 import my_cont_fuzzyknn
from rnn_modified_ID2517522 import my_modified_rnn

df = pd.read_excel('CA3_ID2517522/Wisconsin Diagnostic Breast Cancer.xlsx')

#Getting labels (with assumption that labels are on the last axis)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#Preprocessing
train_x, train_y, test_x, test_y = my_train_test_split(x,y,0.8,None)
train_x_scaled, test_x_scaled = my_min_max_scaler(train_x,test_x)

#Storing results for part b
train_accs_a = []
test_accs_a = []
train_accs_b = []
test_accs_b = []
train_accs_c = []
test_accs_c = []
train_accs_d = []
test_accs_d = []
train_accs_e = []
test_accs_e = []
train_accs_f = []
test_accs_f = []

for k in range(1,11):
    train_acc_k, test_acc_k = my_mknn(train_x_scaled,train_y,test_x_scaled,test_y,k)
    train_accs_a.append(train_acc_k)
    test_accs_a.append(test_acc_k)
    train_acc_k, test_acc_k = my_fuzzyknn(train_x_scaled,train_y,test_x_scaled,test_y,k,2)
    train_accs_b.append(train_acc_k)
    test_accs_b.append(test_acc_k)

best_k = test_accs_b.index(min(test_accs_b)) + 1

for m in range(2,9):
    train_acc_k, test_acc_k = my_fuzzyknn(train_x_scaled,train_y,test_x_scaled,test_y,best_k,m)
    train_accs_c.append(train_acc_k)
    test_accs_c.append(test_acc_k)

for k in range(1,11):
    train_acc_k, test_acc_k = my_cont_fuzzyknn(train_x_scaled,train_y,test_x_scaled,test_y,k,2)
    train_accs_d.append(train_acc_k)
    test_accs_d.append(test_acc_k)


for r in np.linspace(0,1**(1/10),10):
    train_acc_k, test_acc_k = my_rnn(train_x_scaled,train_y,test_x_scaled,test_y,r)
    train_accs_e.append(train_acc_k)
    test_accs_e.append(test_acc_k)

for r in np.linspace(0,1**(1/10),10):
    train_acc_k, test_acc_k = my_modified_rnn(train_x_scaled,train_y,test_x_scaled,test_y,r)
    train_accs_f.append(train_acc_k)
    test_accs_f.append(test_acc_k)

part2a_mknn = []
part2a_fuzzy = []
part2a_contfuzzy = []
part2a_rnn = []
part2a_mrnn = []
folds_x, folds_y = my_train_test_split(x,y,0,5) #partitioned only once
for i in range(5): #num of folds
    test_x = folds_x[i]
    test_y = folds_y[i]
    train_x = np.concatenate([fold for n,fold in enumerate(folds_x) if n!=i],axis=0)
    train_y = np.concatenate([fold for n,fold in enumerate(folds_y) if n!=i],axis=0)
    train_x_scaled, test_x_scaled = my_min_max_scaler(train_x,test_x) #Scale everytime as train set changes
    #Storing fold results
    test_accs_mk = []
    test_accs_fuzzy = []
    test_accs_contfuzzy = []
    test_accs_rnn = []
    test_accs_mrnn = []
    for k in range(1,11):
        _, test_acc_mk = my_mknn(train_x_scaled,train_y,test_x_scaled,test_y,k)
        _, test_acc_fuzzy = my_fuzzyknn(train_x_scaled,train_y,test_x_scaled,test_y,k,2)
        _, test_acc_contfuzzy = my_cont_fuzzyknn(train_x_scaled,train_y,test_x_scaled,test_y,k,2)
        test_accs_mk.append(test_acc_mk)
        test_accs_fuzzy.append(test_acc_fuzzy)
        test_accs_contfuzzy.append(test_acc_contfuzzy)
    for r in np.linspace(0,1**(1/10),10):
        _, test_acc_rnn = my_rnn(train_x_scaled,train_y,test_x_scaled,test_y,r)
        _, test_acc_mrnn = my_modified_rnn(train_x_scaled,train_y,test_x_scaled,test_y,r)
        test_accs_rnn.append(test_acc_rnn)
        test_accs_mrnn.append(test_acc_mrnn)

    part2a_mknn.append(min(test_accs_mk))
    part2a_fuzzy.append(min(test_accs_fuzzy))
    part2a_contfuzzy.append(min(test_accs_contfuzzy))
    part2a_rnn.append(min(test_accs_rnn))
    part2a_mrnn.append(min(test_accs_mrnn))

part2a_mknn = [min(part2a_mknn),max(part2a_mknn),sum(part2a_mknn)/len(part2a_mknn)]
part2a_fuzzy = [min(part2a_fuzzy),max(part2a_fuzzy),sum(part2a_fuzzy)/len(part2a_fuzzy)]
part2a_contfuzzy = [min(part2a_contfuzzy),max(part2a_contfuzzy),sum(part2a_contfuzzy)/len(part2a_contfuzzy)]
part2a_rnn = [min(part2a_rnn),max(part2a_rnn),sum(part2a_rnn)/len(part2a_rnn)]
part2a_mrnn = [min(part2a_mrnn),max(part2a_mrnn),sum(part2a_mrnn)/len(part2a_mrnn)]

part2b_mknn = []
part2b_fuzzy = []
part2b_contfuzzy = []
part2b_rnn = []
part2b_mrnn = []
for j in range(5):
    folds_x, folds_y = my_train_test_split(x,y,0,5) #partitioned only once
    part2b1_mknn = []
    part2b1_fuzzy = []
    part2b1_contfuzzy = []
    part2b1_rnn = []
    part2b1_mrnn = []
    for i in range(5): #num of folds
        test_x = folds_x[i]
        test_y = folds_y[i]
        train_x = np.concatenate([fold for n,fold in enumerate(folds_x) if n!=i],axis=0)
        train_y = np.concatenate([fold for n,fold in enumerate(folds_y) if n!=i],axis=0)
        train_x_scaled, test_x_scaled = my_min_max_scaler(train_x,test_x) #Scale everytime as train set changes
        #Storing fold results
        test_accs_mk = []
        test_accs_fuzzy = []
        test_accs_contfuzzy = []
        test_accs_rnn = []
        test_accs_mrnn = []
        for k in range(1,11):
            _, test_acc_mk = my_mknn(train_x_scaled,train_y,test_x_scaled,test_y,k)
            _, test_acc_fuzzy = my_fuzzyknn(train_x_scaled,train_y,test_x_scaled,test_y,k,2)
            _, test_acc_contfuzzy = my_cont_fuzzyknn(train_x_scaled,train_y,test_x_scaled,test_y,k,2)
            test_accs_mk.append(test_acc_mk)
            test_accs_fuzzy.append(test_acc_fuzzy)
            test_accs_contfuzzy.append(test_acc_contfuzzy)
        for r in np.linspace(0,1**(1/10),10):
            _, test_acc_rnn = my_rnn(train_x_scaled,train_y,test_x_scaled,test_y,r)
            _, test_acc_mrnn = my_modified_rnn(train_x_scaled,train_y,test_x_scaled,test_y,r)
            test_accs_rnn.append(test_acc_rnn)
            test_accs_mrnn.append(test_acc_mrnn)

        part2b1_mknn.append(min(test_accs_mk))
        part2b1_fuzzy.append(min(test_accs_fuzzy))
        part2b1_contfuzzy.append(min(test_accs_contfuzzy))
        part2b1_rnn.append(min(test_accs_rnn))
        part2b1_mrnn.append(min(test_accs_mrnn))

    part2b_mknn.append([min(part2b1_mknn),max(part2b1_mknn),sum(part2b1_mknn)/len(part2b1_mknn)])
    part2b_fuzzy.append([min(part2b1_fuzzy),max(part2b1_fuzzy),sum(part2b1_fuzzy)/len(part2b1_fuzzy)])
    part2b_contfuzzy.append([min(part2b1_contfuzzy),max(part2b1_contfuzzy),sum(part2b1_contfuzzy)/len(part2b1_contfuzzy)])
    part2b_rnn.append([min(part2b1_rnn),max(part2b1_rnn),sum(part2b1_rnn)/len(part2b1_rnn)])
    part2b_mrnn.append([min(part2b1_mrnn),max(part2b1_mrnn),sum(part2b1_mrnn)/len(part2b1_mrnn)])


#Write the results into excel (TAKE INTO COMMENT IF ERROR)
with pd.ExcelWriter('CA3_ID2517522/CA3results_ID2517522.xlsx',engine='openpyxl',mode='a') as writer:
    pd.DataFrame(train_accs_a).to_excel(writer,sheet_name='Part a Training Error',index=False,header=False)
    pd.DataFrame(test_accs_a).to_excel(writer,sheet_name='Part a Test Error',index=False,header=False)
    pd.DataFrame(train_accs_b).to_excel(writer,sheet_name='Part b Training Error',index=False,header=False)
    pd.DataFrame(test_accs_b).to_excel(writer,sheet_name='Part b Test Error',index=False,header=False)
    pd.DataFrame(train_accs_c).to_excel(writer,sheet_name='Part c Training Error',index=False,header=False)
    pd.DataFrame(test_accs_c).to_excel(writer,sheet_name='Part c Test Error',index=False,header=False)
    pd.DataFrame(train_accs_d).to_excel(writer,sheet_name='Part d Training Error',index=False,header=False)
    pd.DataFrame(test_accs_d).to_excel(writer,sheet_name='Part d Test Error',index=False,header=False)
    pd.DataFrame(train_accs_e).to_excel(writer,sheet_name='Part e Training Error',index=False,header=False)
    pd.DataFrame(test_accs_e).to_excel(writer,sheet_name='Part e Test Error',index=False,header=False)
    pd.DataFrame(train_accs_f).to_excel(writer,sheet_name='Part f Training Error',index=False,header=False)
    pd.DataFrame(test_accs_f).to_excel(writer,sheet_name='Part f Test Error',index=False,header=False)
    pd.DataFrame(part2a_mknn).to_excel(writer,sheet_name='Part 2a mknn Results',index=False,header=False)
    pd.DataFrame(part2a_fuzzy).to_excel(writer,sheet_name='Part 2a fuzzy Results',index=False,header=False)
    pd.DataFrame(part2a_contfuzzy).to_excel(writer,sheet_name='Part 2a contfuzzy Results',index=False,header=False)
    pd.DataFrame(part2a_rnn).to_excel(writer,sheet_name='Part 2a rnn Results',index=False,header=False)
    pd.DataFrame(part2a_mrnn).to_excel(writer,sheet_name='Part 2a mrnn Results',index=False,header=False)
    pd.DataFrame(part2b_mknn).to_excel(writer,sheet_name='Part 2b mknn Results',index=False,header=False)
    pd.DataFrame(part2b_fuzzy).to_excel(writer,sheet_name='Part 2b fuzzy Results',index=False,header=False)
    pd.DataFrame(part2b_contfuzzy).to_excel(writer,sheet_name='Part 2b contfuzzy Results',index=False,header=False)
    pd.DataFrame(part2b_rnn).to_excel(writer,sheet_name='Part 2b rnn Results',index=False,header=False)
    pd.DataFrame(part2b_mrnn).to_excel(writer,sheet_name='Part 2b mrnn Results',index=False,header=False)