import pandas as pd
import numpy as np
from utils.partition import partition_data
import random
import math

def setting2_dirch_val(train_full_dataset,test_full_dataset, num_users):
    np.random.seed(42)  # Set the seed for reproducibility
    dict_users, dict_users_test, dict_users_val = {}, {},{}
    for i in range(num_users):
        dict_users[i]=[]
        dict_users_test[i]=[]
        dict_users_val[i]=[]
    
    df=pd.DataFrame(list(train_full_dataset), columns=['images', 'labels'])
    df_test=pd.DataFrame(list(test_full_dataset), columns=['images', 'labels'])
    num_of_classes=len(df['labels'].unique())

    dict_classwise={}
    dict_classwise_test={}
    
    total_train_samples_per_client = 500
    total_test_samples_per_client = 1000
    total_val_samples_per_client = 250
 
    for i in range(num_of_classes):
      dict_classwise[i] = df[df['labels']==i].index.values.astype(int)

    for i in range(num_of_classes):
      dict_classwise_test[i] = df_test[df_test['labels']==i].index.values.astype(int)
      
    for i in range(num_users):
        dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes))
        print("dirichlet_dist",dirichlet_dist)
        num_samples_train = np.round(dirichlet_dist * total_train_samples_per_client).astype(int)
        print("num_samples_train",num_samples_train)
        num_samples_test = np.round(dirichlet_dist * total_test_samples_per_client).astype(int)
        print("num_samples_test",num_samples_test)
        num_samples_val = np.round(dirichlet_dist * total_val_samples_per_client).astype(int)
        print("num_samples_train",num_samples_val)

        for j in range(num_of_classes):
            #class_df_train = df[df['labels'] == j]
            #class_df_test = df_test[df_test['labels'] == j]
            population_size_train = len(dict_classwise[j])
            population_size_test = len(dict_classwise_test[j])

            if num_samples_train[j] > population_size_train:
                num_samples_train[j] = population_size_train
            if num_samples_test[j] > population_size_test:
                num_samples_test[j] = population_size_test
                
            temp=list(np.random.choice(dict_classwise[j], num_samples_train[j], replace = False))
            dict_users[i].extend(temp)
            dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))

            temp_test=list(np.random.choice(dict_classwise_test[j], num_samples_test[j], replace = True))
            dict_users_test[i].extend(temp_test)
            dict_classwise_test[j] = list(set(dict_classwise_test[j]) -set( temp_test))
            
            
            population_size_val = len(dict_classwise[j])
            if num_samples_val[j] > population_size_val:
                num_samples_val[j] = population_size_val
            
            temp_val = list(np.random.choice(dict_classwise[j], num_samples_val[j], replace = False))
            dict_users_val[i].extend(temp_val)
            dict_classwise[j] = list(set(dict_classwise[j]) -set(temp_val))
            
    return dict_users , dict_users_test, dict_users_val



def setting2(train_full_dataset,test_full_dataset, num_users):
    dict_users, dict_users_test = {}, {}
    for i in range(num_users):
        dict_users[i]=[]
        dict_users_test[i]=[]
    
    df=pd.DataFrame(list(train_full_dataset), columns=['images', 'labels'])
    df_test=pd.DataFrame(list(test_full_dataset), columns=['images', 'labels'])
    num_of_classes=len(df['labels'].unique())

    dict_classwise={}
    dict_classwise_test={}
    
 
    for i in range(num_of_classes):
      dict_classwise[i] = df[df['labels']==i].index.values.astype(int)

    for i in range(num_of_classes):
      dict_classwise_test[i] = df_test[df_test['labels']==i].index.values.astype(int)

    for i in range(num_users):
        
        for j in range(num_of_classes):
            if(i==j or (i+1)%10==j):
                temp=list(np.random.choice(dict_classwise[j], 225, replace = False))
                dict_users[i].extend(temp)
                dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))
            
            elif((i+2)%10==j or (i+3)%10==j):
                temp=list(np.random.choice(dict_classwise[j], 7, replace = False))
                dict_users[i].extend(temp)
                dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))

            else:
                temp=list(np.random.choice(dict_classwise[j],6, replace = False))
                dict_users[i].extend(temp)
                dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))
   
    for i in range(num_users):
        
        for j in range(num_of_classes):
            if(i==j or (i+1)%10==j):
                temp=list(np.random.choice(dict_classwise_test[j],450 , replace = False))
                dict_users_test[i].extend(temp)
                dict_classwise_test[j] = list(set(dict_classwise_test[j]) -set( temp))
            elif((i+2)%10==j or (i+3)%10==j or (i+4)%10==j or (i+5)%10==j):
                temp=list(np.random.choice(dict_classwise_test[j], 13, replace = False))
                dict_users_test[i].extend(temp)
                dict_classwise_test[j] = list(set(dict_classwise_test[j]) -set( temp))
            else:
                temp=list(np.random.choice(dict_classwise_test[j], 12, replace = False))
                dict_users_test[i].extend(temp)
                dict_classwise_test[j] = list(set(dict_classwise_test[j]) -set( temp))
   
   
    return dict_users , dict_users_test     


def setting1_val(dataset, num_users, datapoints):
    
    dict_users, dict_users_val = {},{}
    
    for i in range(num_users):
        dict_users[i]=[]
        dict_users_val[i]=[]
        
    df=pd.DataFrame(list(dataset), columns=['images', 'labels'])
    num_of_classes=len(df['labels'].unique())
    
    per_class_client=int(datapoints/num_of_classes)
    per_class_client_val = int(0.5 * per_class_client)
    per_class_total=per_class_client*num_users + per_class_client_val*num_users
    #per_class_total_val=per_class_client_val*num_users
    
    dict_classwise={}
 
    for i in range(num_of_classes):
      dict_classwise[i] = df[df['labels']==i].index.values.astype(int)[:per_class_total]

    for i in range(num_users):
        
        for j in range(num_of_classes):
          temp=list(np.random.choice(dict_classwise[j], per_class_client, replace = False))
          dict_users[i].extend(temp)
          dict_classwise[j] = list(set(dict_classwise[j]) -set(temp))
          
          temp_val=list(np.random.choice(dict_classwise[j], per_class_client_val, replace = False))
          dict_users_val[i].extend(temp_val)
          dict_classwise[j] = list(set(dict_classwise[j]) -set(temp_val))
   
    return dict_users, dict_users_val      


def setting1(dataset, num_users, datapoints):
    
    dict_users = {}
    
    for i in range(num_users):
        dict_users[i]=[]
    df=pd.DataFrame(list(dataset), columns=['images', 'labels'])
    num_of_classes=len(df['labels'].unique())
    
    per_class_client=int(datapoints/num_of_classes)
    per_class_total=per_class_client*num_users

    dict_classwise={}
 
    for i in range(num_of_classes):
      dict_classwise[i] = df[df['labels']==i].index.values.astype(int)[:per_class_total]

    for i in range(num_users):
        
        for j in range(num_of_classes):
          temp=list(np.random.choice(dict_classwise[j], per_class_client, replace = False))
          dict_users[i].extend(temp)
          dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))
   
    return dict_users  

def create_test_data_cross_device(clients, number_of_clients, similar_percentage, client_ids, test_dict, num_of_classes, test_dataset, client_idxs):
    
    selected_client_ids=random.sample(client_ids, int(number_of_clients*similar_percentage))
    remaining_client_ids=list(set(client_ids) - set(selected_client_ids))

    df_test=pd.DataFrame(list(test_dataset), columns=['images', 'labels'])
    dict_classwise_original={}
    dict_classwise={}
    for i in range(num_of_classes):
      dict_classwise_original[i] = df_test[df_test['labels']==i].index.values.astype(int)
      
    dict_classwise=dict_classwise_original.copy()
    
    dict_users_test={}
    for i in range(number_of_clients):
        dict_users_test[i]=[]

    for client_id in selected_client_ids:
        df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])  
        total_len=len(list(clients[client_id].train_dataset))
        df=df['labels'].value_counts().sort_index()
        
        df_list=df.index.tolist()
        

        for i in range(10):
            
            if (i) in df_list:
                test_data_per_class_len=math.floor((df[i]/total_len)*100)
                if(test_data_per_class_len/df[i]>0.2):
                    test_data_per_class_len=math.floor(0.2*df[i])
            else:
                test_data_per_class_len=0
           
            
            temp=list(np.random.choice(dict_classwise[i], test_data_per_class_len, replace = False))
            
            dict_users_test[client_idxs[client_id]].extend(temp)
            dict_classwise[i] = list(set(dict_classwise[i]) -set( temp))
        
    dict_test_used={}
    
    for i in range(num_of_classes):
        dict_test_used[i]=list(set(dict_classwise_original[i])-set(dict_classwise[i]))
        
    for client_id in remaining_client_ids:
        dict_users2=partition_data(test_dataset , 10, "noniid-labeldir", len(remaining_client_ids), 0.5, 1234, dict_test_used)
    
    idx=0
    for client_id in remaining_client_ids:
        dict_users_test[client_idxs[client_id]]=dict_users2[idx]
        print("test dict length finally: ", len(dict_users_test[client_idxs[client_id]]))
        idx+=1
    return dict_users_test, selected_client_ids
    

def get_test_dict(dataset, num_users):
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, 2000, replace=False))
    return dict_users            


def get_dicts(train_full_dataset, test_full_dataset,  num_users, setting, datapoints):
    
    if setting == 'setting2':
        #dict_users, dict_users_test=setting2_dirch(train_full_dataset, test_full_dataset, num_users)
        dict_users, dict_users_test, dict_users_val = setting2_dirch_val(train_full_dataset, test_full_dataset, num_users)
        

    elif setting == 'setting1':
        dict_users,dict_users_val = setting1_val(train_full_dataset, num_users, datapoints)
        dict_users_test=get_test_dict(test_full_dataset, num_users)
        
    
    return dict_users, dict_users_test, dict_users_val
    
