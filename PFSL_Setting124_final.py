import os
import random
import string
import socket
import requests
import sys
import statistics
import threading
import time
import torch
from math import ceil
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils.split_dataset import split_dataset, split_dataset_cifar10tl_exp
from utils.client_simulation import generate_random_clients
from utils.connections import send_object
from utils.arg_parser import parse_arguments
import matplotlib.pyplot as plt
import time
import multiprocessing
from sklearn.metrics import classification_report
import torch.optim as optim 
import copy
from datetime import datetime
from scipy.interpolate import make_interp_spline
import numpy as np
from ConnectedClient import ConnectedClient
import importlib
from utils.merge import merge_grads, merge_weights
import wandb
import pandas as pd
import time 
from utils import dataset_settings, datasets
import torch.nn.functional as F


#To load train and test data for each client for setting 1 and setting 2
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

#######################################################################################
#To intialize every client with their train and test data for setting 4
def initialize_client(client, dataset, batch_size, test_batch_size, tranform):
    
    client.load_data(dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(batch_size, test_batch_size)
######################################################################################
# To check the class distribution of each client
def check_class_distribution(dict_users, full_dataset):
    class_distribution = {}

    # Convert full_dataset to a list
    full_dataset = list(full_dataset)

    # Iterate over the clients
    for client_id, samples in dict_users.items():
        # Count the occurrences of each class label in the samples
        class_counts = {}
        for sample in samples:
            label = full_dataset[sample][1]  # Assuming the label is at index 1
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Calculate the class distribution for the client
        #total_samples = len(samples)
        #class_distribution[client_id] = {label: count / total_samples for label, count in class_counts.items()}
        class_distribution[client_id] = class_counts

    return class_distribution
######################################################################################


#Plots class distribution of train data available to each client
def plot_class_distribution(clients, dataset, batch_size, epochs, opt, client_ids):
    class_distribution=dict()
    number_of_clients=len(client_ids)
    if(len(clients)<=20):
        plot_for_clients=client_ids
    else:
        plot_for_clients=random.sample(client_ids, 20)
    
    fig, ax = plt.subplots(nrows=(int(ceil(len(client_ids)/5))), ncols=5, figsize=(15, 10))
    j=0
    i=0

    #plot histogram
    for client_id in plot_for_clients:
        df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
        class_distribution[client_id]=df['labels'].value_counts().sort_index()
        df['labels'].value_counts().sort_index().plot(ax = ax[i,j], kind = 'bar', ylabel = 'frequency', xlabel=client_id)
        j+=1
        if(j==5 or j==10 or j==15):
            i+=1
            j=0
    fig.tight_layout()
    plt.show()
    wandb.log({"Histogram": wandb.Image(plt)})
    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_histogram.png')  
    plt.savefig('plot_setting3_exp.png')

    max_len=0
    #plot line graphs
    for client_id in plot_for_clients:
        df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
        df['labels'].value_counts().sort_index().plot(kind = 'line', ylabel = 'frequency', label=client_id)
        max_len=max(max_len, list(df['labels'].value_counts(sort=False)[df.labels.mode()])[0])
    plt.xticks(np.arange(0,10))
    plt.ylim(0, max_len)
    plt.legend()
    plt.show()
    wandb.log({"Line graph": wandb.Image(plt)})
    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_line_graph.png')
    
    return class_distribution

############################################################################################################

if __name__ == "__main__":    

    
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Arguments provided", args)

    #setup for wandb

    mode = "online"
    if args.disable_wandb:
        mode = "disabled"
        
    wandb.init(entity="iitbhilai", project="Split_learning_exps", mode = mode)
    wandb.run.name = args.opt_iden

    config = wandb.config          
    config.batch_size = args.batch_size    
    config.test_batch_size = args.test_batch_size        
    config.epochs = args.epochs             
    config.lr = args.lr       
    config.dataset = args.dataset
    config.model = args.model
    config.seed = args.seed
    config.opt = args.opt_iden   
                        

    max_acc = 0
    best_val_acc = 0
    patience = 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    overall_val_acc = []
    overall_test_acc = []
    overall_train_acc = []
    overall_macro_f1_score = []
    overall_weighted_f1_score = [] 

    print('Generating random clients...', end='')
    clients = generate_random_clients(args.number_of_clients)
    client_ids = list(clients.keys())    
    print('Done')

    train_dataset_size, input_channels = split_dataset(args.dataset, client_ids, pretrained=args.pretrained)

    print(f'Random client ids:{str(client_ids)}')
    transform=None
    max_epoch=0
    max_f1=0

    #Assigning train and test data to each client depending for each client
    print('Initializing clients...')
    
    if(args.setting=="setting4"):
        for _, client in clients.items():
            (initialize_client(client, args.dataset, args.batch_size, args.test_batch_size, transform))
    else:
        train_full_dataset, test_full_dataset, input_channels = datasets.load_full_dataset(args.dataset, "data", args.number_of_clients, args.datapoints, args.pretrained)
        #----------------------------------------------------------------
        dict_users , dict_users2, dict_users3 = dataset_settings.get_dicts(train_full_dataset, test_full_dataset, args.number_of_clients, args.setting, args.datapoints)

        dict_users_test_equal=dataset_settings.get_test_dict(test_full_dataset, args.number_of_clients)

        client_idx=0
        dict_user_train=dict()
        dict_user_test=dict()
        dict_user_val=dict()
        client_idxs=dict()

        for _, client in clients.items():
            dict_user_train[_]=dict_users[client_idx]
            dict_user_test[_]=dict_users2[client_idx]
            dict_user_val[_]=dict_users3[client_idx]
            client_idxs[_]=client_idx
            client_idx+=1
        for _, client in clients.items():
            client.train_dataset=DatasetSplit(train_full_dataset, dict_user_train[_])
            client.test_dataset=DatasetSplit(test_full_dataset, dict_user_test[_])
            client.val_dataset=DatasetSplit(train_full_dataset, dict_user_val[_])
            client.create_DataLoader(args.batch_size, args.test_batch_size)
            print("client: ", _)
            print("test dataset: ", len(client.test_dataset))
            print("val dataset: ", len(client.val_dataset))
            print("train dataset: ", len(client.train_dataset))
            
         # Train and test data intialisation complete
        #######################################################################
        train_class_distribution = check_class_distribution(dict_users,train_full_dataset)
        test_class_distribution = check_class_distribution(dict_users2,test_full_dataset)
        val_class_distribution = check_class_distribution(dict_users3,train_full_dataset)
        # Print the class distribution for each client
        for client_id, class_dist in train_class_distribution.items():
            print(f"Client {client_id} training class distribution:")
            for label, proportion in class_dist.items():
                print(f"Class {label}: {proportion}")
            print()
            
        for client_id, class_dist in val_class_distribution.items():
            print(f"Client {client_id} validation class distribution:")
            for label, proportion in class_dist.items():
                print(f"Class {label}: {proportion}")
            print()

        for client_id, class_dist in test_class_distribution.items():
            print(f"Client {client_id} testing class distribution:")
            for label, proportion in class_dist.items():
                print(f"Class {label}: {proportion}")
            print()
        ##########################################################################
    print('Client Intialization complete.')    
    # Train and test data intialisation complete 
    #class_distribution=plot_class_distribution(clients, args.dataset, args.batch_size, args.epochs, args.opt_iden, client_ids)


    #Assigning front, center and back models and their optimizers for all the clients
    model = importlib.import_module(f'models.{args.model}')
  
    for _, client in clients.items():
        client.front_model = model.front(input_channels, pretrained=args.pretrained)
        client.back_model = model.back(pretrained=args.pretrained)
    print('Done')
  
    for _, client in clients.items():
        # client.front_optimizer = optim.SGD(client.front_model.parameters(), lr=args.lr, momentum=0.9)
        # client.back_optimizer = optim.SGD(client.back_model.parameters(), lr=args.lr, momentum=0.9)
        client.front_optimizer = optim.Adam(client.front_model.parameters(), lr=args.lr)
        client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)

    first_client = clients[client_ids[0]]
    num_iterations = ceil(len(first_client.train_DataLoader.dataset)/args.batch_size)
    num_val_iterations = ceil(len(first_client.val_DataLoader.dataset)/args.batch_size)
    num_test_iterations= ceil(len(first_client.test_DataLoader.dataset)/args.test_batch_size)
    sc_clients = {} #server copy clients

    for iden in client_ids:
        sc_clients[iden] = ConnectedClient(iden, None)

    for _,s_client in sc_clients.items():
        s_client.center_model = model.center(pretrained=args.pretrained)
        s_client.center_model.to(device)
        # s_client.center_optimizer = optim.SGD(s_client.center_model.parameters(), lr=args.lr, momentum=0.9)
        s_client.center_optimizer = optim.Adam(s_client.center_model.parameters(), args.lr)

    st = time.time()

    macro_avg_f1_classes=[]
    weighted_avg_f1_classes=[]

    criterion=F.cross_entropy
    flag = {}
    per_conv ={}
    #logging the gradients of the models of all the three parts to wandb
    for _, client in clients.items(): 
        wandb.watch(client.front_model, criterion, log="all",log_freq=2) 
        wandb.watch(client.back_model, criterion, log="all", log_freq=2)
    for _, s_client in sc_clients.items():
        wandb.watch(s_client.center_model, criterion, log="all", log_freq=2)
    print("................. Generalisation Training Phase ...................")
    #Starting the training process 
    #for epoch in range(args.epochs):
    epoch = 1
    check = 0
    for client_id, client in clients.items():
        flag[client_id] = 0
        per_conv[client_id] = 0
    while epoch <= args.epochs: 
        print("............................................Epoch.....................................", epoch)
        if(args.setting == 'setting2' and epoch == args.checkpoint): # When starting epoch of the perosnalisation is reached, freeze all the layers of the center model 
            print(".....................Generalisation Phase Done.....................")
            print(".....................Personalisation Phase Started..........................")
            for _, s_client in sc_clients.items():
                s_client.center_model.freeze(epoch, pretrained=True)
                ########

        overall_train_acc.append(0)


        for _, client in clients.items():
            if  flag[_] > 0 :
                continue
            else: 
                client.train_acc.append(0)
                client.iterator = iter(client.train_DataLoader)
            
        #For every batch in the current epoch
        for iteration in range(num_iterations):
            print(f'\rEpoch: {epoch}, Iteration: {iteration+1}/{num_iterations}', end='')

            
            for _, client in clients.items():
                if  flag[_] > 0 :
                    continue
                else:
                    client.forward_front()

            for client_id, client in sc_clients.items():
                if  flag[client_id] > 0 :
                    continue
                else:
                    client.remote_activations1 = clients[client_id].remote_activations1
                    client.forward_center()

            for client_id, client in clients.items():
                if  flag[client_id] > 0 :
                    continue
                else:
                    client.remote_activations2 = sc_clients[client_id].remote_activations2
                    client.forward_back()

            for _, client in clients.items():
                if  flag[_] > 0 :
                    continue
                else:
                    client.calculate_loss()

            for _, client in clients.items():
                if  flag[_] > 0 :
                    continue
                else:
                    client.backward_back()

            for client_id, client in sc_clients.items():
                if  flag[client_id] > 0 :
                    continue
                else:
                    client.remote_activations2 = clients[client_id].remote_activations2
                    client.backward_center()

            for _, client in clients.items():
                if  flag[_] > 0 :
                    continue
                else:
                    client.step_back()
                    client.zero_grad_back()

            #merge grads uncomment below

            # if epoch%2 == 0:
            #     params = []
            #     normalized_data_sizes = []
            #     for iden, client in clients.items():
            #         params.append(sc_clients[iden].center_model.parameters())
            #         normalized_data_sizes.append(len(client.train_dataset) / train_dataset_size)
            #     merge_grads(normalized_data_sizes, params)

            for _, client in sc_clients.items():
                if  flag[_] > 0 :
                    continue
                else:
                    client.center_optimizer.step()
                    client.center_optimizer.zero_grad()

            for _, client in clients.items():
                if  flag[_] > 0 :
                    continue
                else:
                    client.train_acc[-1] += client.calculate_train_acc() #train accuracy of every client in the current epoch in the current batch

        for c_id, client in clients.items():
            if  flag[c_id] > 0 :
                    continue
            else:
                client.train_acc[-1] /= num_iterations # train accuracy of every client of all the batches in the current epoch
                
        for c_id, client in clients.items():
            if  flag[c_id] > 0 :
                overall_train_acc[-1] += client.train_acc[-6]
            else:
                overall_train_acc[-1] += client.train_acc[-1] 

        overall_train_acc[-1] /= len(clients) #avg train accuracy of all the clients in the current epoch
        if(epoch < args.checkpoint):
            print(f' Generalized Average Train Acc: {overall_train_acc[-1]}')
        else :
            print(f' Personalized Average Train Acc: {overall_train_acc[-1]}')

        # merge weights below uncomment 
        if(epoch < args.checkpoint):
            params = []
            for _, client in sc_clients.items():
                params.append(copy.deepcopy(client.center_model.state_dict()))
            w_glob = merge_weights(params)

            for _, client in sc_clients.items():
                client.center_model.load_state_dict(w_glob)

        params = []

        #In the personalisation phase merging of weights of the back layers is stopped
        if(args.setting == 'setting1' or args.setting == 'setting4' or (args.setting == 'setting2' and epoch < args.checkpoint)):
            for _, client in clients.items():
                params.append(copy.deepcopy(client.back_model.state_dict()))
            w_glob_cb = merge_weights(params)
            del params
    
            for _, client in clients.items():
                client.back_model.load_state_dict(w_glob_cb)
        
        if epoch == 1:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>        Communication Overhead per datapoint       >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #for client_id, client in clients.items():
            print("Client Front to Server : Activations")
            print((clients[client_id].activations1[0]).size())
            print("Total Tesor Size:",(clients[client_id].activations1[0]).element_size())
            print("Total number of elements:",clients[client_id].activations1[0].numel())
            print("Total Size in Bytes",clients[client_id].activations1[0].element_size() * (clients[client_id].activations1[0].numel()))
                #clients[client_id].activations1
            print(".............................................................")
            print("Server to Client Back : Activations")
            print((clients[client_id].remote_activations2[0]).size())
            print("Total Tesor Size:",(clients[client_id].remote_activations2[0]).element_size())
            print("Total number of elements:",clients[client_id].remote_activations2[0].numel())
            print("Total Size in Bytes",clients[client_id].remote_activations2[0].element_size() * (clients[client_id].remote_activations2[0].numel()))
            print(".............................................................")
            print("Client Back to Server : Geadients")
            print((clients[client_id].remote_activations2.grad[0]).size())
            print("Total Tesor Size:",(clients[client_id].remote_activations2.grad[0]).element_size())
            print("Total number of elements:",clients[client_id].remote_activations2.grad[0].numel())
            print("Total Size in Bytes",(clients[client_id].remote_activations2.grad[0]).element_size() * (clients[client_id].remote_activations2.grad[0]).numel())        
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            
        
        
        print("............ Validation Started ............")
        #Testing every epoch
        if (epoch%1 == 0 ):
            if(args.setting == 'setting2' and epoch == args.checkpoint):
                for _, s_client in sc_clients.items():
                    s_client.center_model.freeze(epoch, pretrained=True)
                    
            with torch.no_grad():
                test_acc = 0
                overall_val_acc.append(0)
            
                for _, client in clients.items():
                    if  flag[_] > 0 :
                        continue
                    else:
                        client.val_acc.append(0)
                        client.macro_f1_score.append(0)
                        client.weighted_f1_score.append(0)
                        client.iterator = iter(client.val_DataLoader)
                        client.pred=[]
                        client.y=[]

                #For every batch in the testing phase
                for iteration in range(num_val_iterations):
    
                    for _, client in clients.items():
                        if  flag[_] > 0 :
                            continue
                        else:
                            client.forward_front()

                    for client_id, client in sc_clients.items():
                        if  flag[client_id] > 0 :
                            continue
                        else:
                            client.remote_activations1 = clients[client_id].remote_activations1
                            client.forward_center()

                    for client_id, client in clients.items():
                        if  flag[client_id] > 0 :
                            continue
                        else:
                            client.remote_activations2 = sc_clients[client_id].remote_activations2
                            client.forward_back()

                    for _, client in clients.items():
                        if  flag[_] > 0 :
                            continue
                        else:
                            client.val_acc[-1] += client.calculate_test_acc()


                for _, client in clients.items():
                    if  flag[_] > 0 :
                            continue
                    else:
                        client.val_acc[-1] /= num_val_iterations
                    #overall_val_acc[-1] += client.val_acc[-1]
                    
                for c_id, client in clients.items():
                    if  flag[c_id] > 0 :
                        overall_val_acc[-1] += client.val_acc[-6]
                    else:
                        overall_val_acc[-1] += client.val_acc[-1]
                    #Calculating the F1 scores using the classification report from sklearn metrics
                    #if(args.setting=='setting2'):
                    #    clr=classification_report(np.array(client.y), np.array(client.pred), output_dict=True)
                    #    idx=client_idxs[_]
                    #
                    #    macro_avg_f1_classes.append((clr[str(idx)]['f1-score']+clr[str((idx+1)%10)]['f1-score'])/2) #macro f1 score of the 2 prominent classes in setting2
                if args.setting == 'setting2':
                    for _, client in clients.items():
                        if  flag[_] > 0 :
                            print("**************************************************************************************************** Skip",_)
                            continue
                        else:
                            clr = classification_report(np.array(client.y), np.array(client.pred), output_dict=True)
                            #f1_scores = [clr.get(str(i), {'f1-score': 0})['f1-score'] for i in range(10)]  # Assuming you have 10 classes
                            #macro_avg_f1 = sum(f1_scores) / len(f1_scores)
                            macro_avg_f1 = clr['macro avg']['f1-score']
                            weighted_f1_score = clr['weighted avg']['f1-score']
                            client.macro_f1_score[-1] = macro_avg_f1 
                            client.weighted_f1_score[-1] =  weighted_f1_score
                            macro_avg_f1_classes.append(macro_avg_f1)
                            weighted_avg_f1_classes.append(weighted_f1_score)
                            #print(macro_avg_f1_classes)
                             
                
                overall_val_acc[-1] /= len(clients) #average Val Accuracy of all the clients in the current epoch

                if(args.setting=='setting2'):
                    f1_avg_all_user = sum(macro_avg_f1_classes)/len(macro_avg_f1_classes) #average f1 scores of the clients for the prominent 2 classes in the current epoch
                    weighted_f1_all_user = sum(weighted_avg_f1_classes)/len(weighted_avg_f1_classes)
                    overall_macro_f1_score.append(f1_avg_all_user)
                    overall_weighted_f1_score.append(weighted_f1_all_user)
                    macro_avg_f1_classes = []
                    weighted_avg_f1_classes =[]
                    #Noting the maximum f1 score
                    if(f1_avg_all_user> max_f1):
                        max_f1=f1_avg_all_user
                        max_epoch=epoch
                    #if epoch < args.checkpoint : 
                    print(f' Personalized Average Val Acc: {overall_val_acc[-1]}  Macro f1 score: {f1_avg_all_user} Max f1 Score: {max_f1} Weighted f1 score: {weighted_f1_all_user}')
                    max_acc=max(overall_val_acc)
                    print(f' Maximum Val Acc: {max_acc} Epoch: {overall_val_acc.index(max_acc)+1}')
                    if epoch < args.checkpoint : 
                        if overall_val_acc[-1] > best_val_acc:
                            best_val_acc = overall_val_acc[-1]
                            patience = 0
                            # change-1
                            # for _, client in clients.items():
                            #     torch.save(client.front_model.state_dict(), "gen_best_front_model.pt")
                            #     torch.save(client.back_model.state_dict(), "gen_best_back_model.pt")
                            # for _,s_client in sc_clients.items():
                            #     torch.save(s_client.center_model.state_dict(), "gen_best_center_model.pt")
                            #     #torch.save(s_client.center_front_model.state_dict(), "best_Cfront_model.pt")
                            #     #torch.save(s_client.center_back_model.state_dict(), "best_Cback_model.pt")
                            
                            # Create a directory for saved models if it doesn't exist
                            os.makedirs("saved_models", exist_ok=True)

                            # For each client
                            for client_id, client in clients.items():
                                print("Save client side general model",client_id)
                                # Construct the filenames for the models based on the client_id
                                front_model_filename = f"gen_best_front_model_client.pt"
                                back_model_filename = f"gen_best_back_model_client.pt"

                                # Create the file paths for the models inside the "saved_models" directory
                                front_model_path = os.path.join("saved_models", front_model_filename)
                                back_model_path = os.path.join("saved_models", back_model_filename)

                                # Save the models
                                torch.save(client.front_model.state_dict(), front_model_path)
                                torch.save(client.back_model.state_dict(), back_model_path)

                            # For each sc_client
                            for sc_client_id, s_client in sc_clients.items():
                                print("Save server side general model",sc_client_id)
                                # Construct the filename for the center model based on the sc_client_id
                                center_model_filename = f"gen_best_center_model_sc_client.pt"

                                # Create the file path for the center model inside the "saved_models" directory
                                center_model_path = os.path.join("saved_models", center_model_filename)

                                # Save the center model
                                torch.save(s_client.center_model.state_dict(), center_model_path)

                        else: patience += 1
                    else:
                        for client_id, client in clients.items():
                            if  flag[client_id] > 0 :
                                continue
                            else:
                                if client.val_acc[-1] > best_val_acc_in[client_id] :
                                    best_val_acc_in[client_id] = client.val_acc[-1]
                                    patience_individual[client_id] = 0
                                else:
                                    patience_individual[client_id] += 1
                   
                    print(f"Epoch :{epoch : 3d} | Val Accuracy : {overall_val_acc[-1]:.4f} | Patience {patience: 3d}/{5}")
                    if (epoch < args.checkpoint and patience == 5) or (epoch == args.checkpoint-1 and patience < 5):
                        print("Reached to the convergence point at generalisation phase")
                        gen_conv = epoch - 5
                        conv = epoch
                        print(conv)
                        args.checkpoint = epoch + 1
                        print(epoch)
                        patience_individual={}
                        best_val_acc_in = {}
                        for client_id, client in clients.items():
                            patience_individual[client_id] = 0
                            best_val_acc_in[client_id] = best_val_acc
                        patience = 0
                        best_val_acc = 0
                        # change-2
                        # # For front and back models of clients
                        # for client_id, client in clients.items():
                        #     front_model_filename = f"gen_best_front_model_client_{client_id}.pt"
                        #     back_model_filename = f"gen_best_back_model_client_{client_id}.pt"
                            
                        #     front_model_path = os.path.join(client_id, front_model_filename)
                        #     back_model_path = os.path.join(client_id, back_model_filename)
                            
                        #     os.makedirs(client_id, exist_ok=True)  # Create a directory for each client if it doesn't exist
                            
                        #     torch.save(client.front_model.state_dict(), front_model_path)
                        #     torch.save(client.back_model.state_dict(), back_model_path)

                        # # For center models of sc_clients
                        # for sc_client_id, s_client in sc_clients.items():
                        #     center_model_filename = f"gen_best_center_model_sc_client_{sc_client_id}.pt"
                        #     center_model_path = os.path.join(sc_client_id, center_model_filename)
                            
                        #     os.makedirs(sc_client_id, exist_ok=True)  # Create a directory for each sc_client if it doesn't exist
                            
                        #     torch.save(s_client.center_model.state_dict(), center_model_path)
                        print("Load All the saved models")
                        
                        # For each client
                        for client_id, client in clients.items():
                            print("Load client side general model",client_id)
                            # Construct the filenames for the models based on the client_id
                            front_model_filename = f"gen_best_front_model_client.pt"
                            back_model_filename = f"gen_best_back_model_client.pt"

                            # Create the file paths for the models inside the "saved_models" directory
                            front_model_path = os.path.join("saved_models", front_model_filename)
                            back_model_path = os.path.join("saved_models", back_model_filename)

                            # Load the models
                            client.front_model.load_state_dict(torch.load(front_model_path))
                            client.back_model.load_state_dict(torch.load(back_model_path))

                        # For each sc_client
                        for sc_client_id, s_client in sc_clients.items():
                            print("Load server side general model",sc_client_id)
                            # Construct the filename for the center model based on the sc_client_id
                            center_model_filename = f"gen_best_center_model_sc_client.pt"

                            # Create the file path for the center model inside the "saved_models" directory
                            center_model_path = os.path.join("saved_models", center_model_filename)

                            # Load the center model
                            s_client.center_model.load_state_dict(torch.load(center_model_path))


                            
                    if epoch >= args.checkpoint:
                        for client_id, client in clients.items():
                            if  flag[client_id] > 0 :
                                continue
                            else:
                                if patience_individual[client_id] == 5: 
                                    flag[client_id] += 1
                                    print("Reached to the convergence point at personalisation phase",client_id)
                                    print("Epoch",epoch)
                                    per_conv[client_id] = epoch - 5
                                    print(per_conv)
                                    print(flag)
                                    client.back_model.freeze(epoch, pretrained=True)
                                    #change-3
                                    # front_model_filename = f"per_best_front_model_client_{client_id}.pt"
                                    # back_model_filename = f"per_best_back_model_client_{client_id}.pt"
                                    # front_model_path = os.path.join(client_id, front_model_filename)
                                    # back_model_path = os.path.join(client_id, back_model_filename)
                                    
                                    # os.makedirs(client_id, exist_ok=True)  # Create a directory for each client if it doesn't exist
                                    
                                    # torch.save(client.front_model.state_dict(), front_model_path)
                                    # torch.save(client.back_model.state_dict(), back_model_path)
                                    print("Save client back side personal model",client_id)

                                    # Create a directory for saved models if it doesn't exist
                                    os.makedirs("saved_models", exist_ok=True)
 
                                    # Create a directory for each client based on the client_id
                                    #client_dir = os.path.join("saved_models", client_id)
                                    #os.makedirs(client_dir, exist_ok=True)

                                    # Construct the filenames for the models based on the client_id
                                    #front_model_filename = f"per_best_front_model_client_{client_id}.pt"
                                    back_model_filename = f"per_best_back_model_client_{client_id}.pt"

                                    # Create the file paths for the models inside the client's directory
                                    #front_model_path = os.path.join(client_dir, front_model_filename)
                                    back_model_path = os.path.join("saved_models", back_model_filename)

                                    # Save the models
                                    #torch.save(client.front_model.state_dict(), front_model_path)
                                    torch.save(client.back_model.state_dict(), back_model_path)
                       
                        check  = 0 
                        for client_id, client in clients.items():
                            if flag[client_id] > 0 : 
                                check +=1
                                
                        if check == 10:
                            print("*************************All reached to the convergence**************************************")
                            # change-4
                            # For center models of sc_clients
                            # for sc_client_id, s_client in sc_clients.items():
                            #     center_model_filename = f"per_best_center_model_sc_client_{sc_client_id}.pt"
                            #     center_model_path = os.path.join(sc_client_id, center_model_filename)
                                
                            #     os.makedirs(sc_client_id, exist_ok=True)  # Create a directory for each sc_client if it doesn't exist
                                
                            #     torch.save(s_client.center_model.state_dict(), center_model_path)
                            # For each sc_client
                            # for sc_client_id, s_client in sc_clients.items():
                            #     # Create a directory for each sc_client based on the sc_client_id
                            #     sc_client_dir = os.path.join("saved_models", sc_client_id)
                            #     os.makedirs(sc_client_dir, exist_ok=True)

                            #     # Construct the filename for the center model based on the sc_client_id
                            #     center_model_filename = f"per_best_center_model_sc_client_{sc_client_id}.pt"

                            #     # Create the file path for the center model inside the sc_client's directory
                            #     center_model_path = os.path.join(sc_client_dir, center_model_filename)

                            #     # Save the center model
                            #    torch.save(s_client.center_model.state_dict(), center_model_path)
                            break
                else:
                    print(f' Generalisation Average Val Acc: {overall_val_acc[-1]}   ')
                    max_acc=max(max_acc, overall_val_acc[-1])
                    print("Maximum Val Acc: ", max_acc)
                    if overall_val_acc[-1] < best_val_acc:
                        best_val_acc = overall_val_acc[-1]
                        patience = 0
                        # change -5
                        # for client_id, client in clients.items():
                        #     front_model_filename = f"gen_best_front_model_client_{client_id}.pt"
                        #     back_model_filename = f"gen_best_back_model_client_{client_id}.pt"
                            
                        #     front_model_path = os.path.join(client_id, front_model_filename)
                        #     back_model_path = os.path.join(client_id, back_model_filename)
                            
                        #     os.makedirs(client_id, exist_ok=True)  # Create a directory for each client if it doesn't exist
                            
                        #     torch.save(client.front_model.state_dict(), front_model_path)
                        #     torch.save(client.back_model.state_dict(), back_model_path)

                        # # For center models of sc_clients
                        # for sc_client_id, s_client in sc_clients.items():
                        #     center_model_filename = f"gen_best_center_model_sc_client_{sc_client_id}.pt"
                        #     center_model_path = os.path.join(sc_client_id, center_model_filename)
                            
                        #     os.makedirs(sc_client_id, exist_ok=True)  # Create a directory for each sc_client if it doesn't exist
                            
                        #     torch.save(s_client.center_model.state_dict(), center_model_path)
                       

                        os.makedirs("saved_models", exist_ok=True)

                            # For each client
                        for client_id, client in clients.items():
                                print("Save client side general model",client_id)
                                # Construct the filenames for the models based on the client_id
                                front_model_filename = f"gen_best_front_model_client.pt"
                                back_model_filename = f"gen_best_back_model_client.pt"

                                # Create the file paths for the models inside the "saved_models" directory
                                front_model_path = os.path.join("saved_models", front_model_filename)
                                back_model_path = os.path.join("saved_models", back_model_filename)

                                # Save the models
                                torch.save(client.front_model.state_dict(), front_model_path)
                                torch.save(client.back_model.state_dict(), back_model_path)

                            # For each sc_client
                        for sc_client_id, s_client in sc_clients.items():
                                print("Save server side general model",sc_client_id)
                                # Construct the filename for the center model based on the sc_client_id
                                center_model_filename = f"gen_best_center_model_sc_client.pt"

                                # Create the file path for the center model inside the "saved_models" directory
                                center_model_path = os.path.join("saved_models", center_model_filename)

                                # Save the center model
                                torch.save(s_client.center_model.state_dict(), center_model_path)
 
                    else: patience += 1
                    print(f"Epoch :{epoch : 3d} | Val Accuracy : {overall_val_acc[-1]:.4f} | Patience {patience: 3d}/{5}")
                    if patience == 5:
                        print("Reached to the convergence point at generalisation phase")
                        break  
                    
            wandb.log({
                "Epoch": epoch,
                "Personalized Average Train Accuracy": overall_train_acc[-1],
                "Personalized Average val Accuracy": overall_val_acc[-1],  
            })
        epoch+=1
    ################################################################################################
    if args.setting=='setting2' and epoch > args.epochs: 
        print("Reached to the end without converging at personalisation phase")
        # Assuming `clients` and `sc_clients` are dictionaries
        # For front and back models of clients
        #change -6
        # for client_id, client in clients.items():
        #     if flag[client_id] == 0 : 
        #         print("Save the model",client_id)
        #         front_model_filename = f"per_best_front_model_client_{client_id}.pt"
        #         back_model_filename = f"per_best_back_model_client_{client_id}.pt"
                
        #         front_model_path = os.path.join(client_id, front_model_filename)
        #         back_model_path = os.path.join(client_id, back_model_filename)
                
        #         os.makedirs(client_id, exist_ok=True)  # Create a directory for each client if it doesn't exist
                
        #         torch.save(client.front_model.state_dict(), front_model_path)
        #         torch.save(client.back_model.state_dict(), back_model_path)

        # # For center models of sc_clients
        # for sc_client_id, s_client in sc_clients.items():
        #     center_model_filename = f"per_best_center_model_sc_client_{sc_client_id}.pt"
        #     center_model_path = os.path.join(sc_client_id, center_model_filename)
            
        #     os.makedirs(sc_client_id, exist_ok=True)  # Create a directory for each sc_client if it doesn't exist
            
        #     torch.save(s_client.center_model.state_dict(), center_model_path)
 
        # Create a directory for saved models if it doesn't exist
        os.makedirs("saved_models", exist_ok=True)

        # For each client
        for client_id, client in clients.items():
            if flag[client_id] == 0 : 
                print("Save client back side personal model",client_id)
                per_conv[client_id] = epoch 
                # Create a directory for each client based on the client_id
                #client_dir = os.path.join("saved_models", client_id)
                #os.makedirs(client_dir, exist_ok=True)

                # Construct the filenames for the models based on the client_id
                #front_model_filename = f"per_best_front_model_client_{client_id}.pt"
                back_model_filename = f"per_best_back_model_client_{client_id}.pt"

                # Create the file paths for the models inside the client's directory
                #front_model_path = os.path.join("saved_models", front_model_filename)
                back_model_path = os.path.join("saved_models", back_model_filename)

                # Save the models
                #torch.save(client.front_model.state_dict(), front_model_path)
                torch.save(client.back_model.state_dict(), back_model_path)


        
    ##################################################################################################
    macro_avg_f1_classes=[]
    weighted_avg_f1_classes =[]
    print("..............................Testing Started...............................")
    epoch = 1
    test_epoch = 1
    if args.setting=='setting2':
        test_epoch = 2
    
    while epoch <= test_epoch:  
        #print("epoch",epoch)
        with torch.no_grad():
            test_acc = 0
            overall_test_acc.append(0)  
            if epoch == 1:
                print("Generalisation Started")
                # for client_id, client in clients.items():
                #     front_model_filename = f"gen_best_front_model_client_{client_id}.pt"
                #     back_model_filename = f"gen_best_back_model_client_{client_id}.pt"
                    
                #     front_model_path = os.path.join(client_id, front_model_filename)
                #     back_model_path = os.path.join(client_id, back_model_filename)
                    
                #     front_model_state_dict = torch.load(front_model_path)
                #     back_model_state_dict = torch.load(back_model_path)
                    
                #     client.front_model.load_state_dict(front_model_state_dict)
                #     client.back_model.load_state_dict(back_model_state_dict)
                # # For center models of sc_clients
                # for sc_client_id, s_client in sc_clients.items():
                #     center_model_filename = f"gen_best_center_model_sc_client_{sc_client_id}.pt"
                #     center_model_path = os.path.join(sc_client_id, center_model_filename)
                    
                #     center_model_state_dict = torch.load(center_model_path)
                    
                #     s_client.center_model.load_state_dict(center_model_state_dict)
                
                # For each client
             
                # For each client
                for client_id, client in clients.items():
                    print("Load client side general model",client_id)
                    # Construct the filenames for the models based on the client_id
                    front_model_filename = f"gen_best_front_model_client.pt"
                    back_model_filename = f"gen_best_back_model_client.pt"

                    # Create the file paths for the models inside the "saved_models" directory
                    front_model_path = os.path.join("saved_models", front_model_filename)
                    back_model_path = os.path.join("saved_models", back_model_filename)

                    # Load the models
                    client.front_model.load_state_dict(torch.load(front_model_path))
                    client.back_model.load_state_dict(torch.load(back_model_path))

                # For each sc_client
                for sc_client_id, s_client in sc_clients.items():
                    print("Load server side general model",sc_client_id)
                    # Construct the filename for the center model based on the sc_client_id
                    center_model_filename = f"gen_best_center_model_sc_client.pt"

                    # Create the file path for the center model inside the "saved_models" directory
                    center_model_path = os.path.join("saved_models", center_model_filename)

                    # Load the center model
                    s_client.center_model.load_state_dict(torch.load(center_model_path))

            if epoch == 2 and args.setting=='setting2': 
                print("Personalisation Started")
                # Assuming `clients` and `sc_clients` are dictionaries
                # For front and back models of clients
                # for client_id, client in clients.items():
                #     front_model_filename = f"per_best_front_model_client_{client_id}.pt"
                #     back_model_filename = f"per_best_back_model_client_{client_id}.pt"
                    
                #     front_model_path = os.path.join(client_id, front_model_filename)
                #     back_model_path = os.path.join(client_id, back_model_filename)
                    
                #     front_model_state_dict = torch.load(front_model_path)
                #     back_model_state_dict = torch.load(back_model_path)
                    
                #     client.front_model.load_state_dict(front_model_state_dict)
                #     client.back_model.load_state_dict(back_model_state_dict)
                # # For center models of sc_clients
                # for sc_client_id, s_client in sc_clients.items():
                #     center_model_filename = f"per_best_center_model_sc_client_{sc_client_id}.pt"
                #     center_model_path = os.path.join(sc_client_id, center_model_filename)
                    
                #     center_model_state_dict = torch.load(center_model_path)
                    
                #     s_client.center_model.load_state_dict(center_model_state_dict)
                
                # For each client
                for client_id, client in clients.items():
                    print("Load client back side personal model",client_id)
                    # Create the directory path for the client
                    #client_dir = os.path.join("saved_models", client_id)

                    # Construct the filenames for the models based on the client_id
                    #front_model_filename = f"per_best_front_model_client_{client_id}.pt"
                    back_model_filename = f"per_best_back_model_client_{client_id}.pt"

                    # Create the file paths for the models inside the client's directory
                    #front_model_path = os.path.join(client_dir, front_model_filename)
                    back_model_path = os.path.join("saved_models", back_model_filename)

                    # Load the models
                    #client.front_model.load_state_dict(torch.load(front_model_path))
                    client.back_model.load_state_dict(torch.load(back_model_path))

            for _, client in clients.items():
                client.test_acc.append(0)
                client.iterator = iter(client.test_DataLoader)
                client.pred=[]
                client.y=[]
            #For every batch in the testing phase
            for iteration in range(num_test_iterations):
                for _, client in clients.items():
                    client.forward_front()

                for client_id, client in sc_clients.items():
                    client.remote_activations1 = clients[client_id].remote_activations1
                    client.forward_center()

                for client_id, client in clients.items():
                    client.remote_activations2 = sc_clients[client_id].remote_activations2
                    client.forward_back()

                for _, client in clients.items():
                    client.test_acc[-1] += client.calculate_test_acc()


            for _, client in clients.items():
                    client.test_acc[-1] /= num_test_iterations
                    overall_test_acc[-1] += client.test_acc[-1]
                    if args.setting == 'setting2':
                        clr = classification_report(np.array(client.y), np.array(client.pred), output_dict=True)
                        #f1_scores = [clr.get(str(i), {'f1-score': 0})['f1-score'] for i in range(10)]  # Assuming you have 10 classes
                        macro_avg_f1 = clr['macro avg']['f1-score']
                        weighted_f1_score = clr['weighted avg']['f1-score']
                        #client.macro_f1_score[-1] = macro_avg_f1 
                        #client.weighted_f1_score[-1] =  weighted_f1_score
                        macro_avg_f1_classes.append(macro_avg_f1)
                        weighted_avg_f1_classes.append(weighted_f1_score)
                        
                        
                        #macro_avg_f1 = sum(f1_scores) / len(f1_scores)
                        #macro_avg_f1_classes.append(macro_avg_f1) 
                        
            overall_test_acc[-1] /= len(clients) #average Val Accuracy of all the clients in the current epoch

            if(args.setting=='setting2'):
                    f1_avg_all_user = sum(macro_avg_f1_classes)/len(macro_avg_f1_classes) #average f1 scores of the clients for the prominent 2 classes in the current epoch
                    weighted_f1_all_user = sum(weighted_avg_f1_classes)/len(weighted_avg_f1_classes)
                    macro_avg_f1_classes = []
                    weighted_avg_f1_classes = []
                    print(f' Generalized Average Test Acc: {overall_test_acc[-1]} Macro f1 score: {f1_avg_all_user} Weighted f1 score: {weighted_f1_all_user}')
                    
            else:
                    print(f' Personalized Average Test Acc: {overall_test_acc[-1]}   ')
        epoch += 1
            
    ###################################################################################################
       
        

    timestamp = int(datetime.now().timestamp())
    plot_config = f'''dataset: {args.dataset},
                    model: {args.model},
                    batch_size: {args.batch_size}, lr: {args.lr},
                    '''

    et = time.time()
    print("............................................................................................................")
    print(f"Time taken for this run {(et - st)/60} mins")
    print("............................................................................................................")
    wandb.log({"time taken by program in mins": (et - st)/60})
    print("............................................................................................................")
    print("overall Val Accuracy Generalisation",overall_val_acc[:conv])
    print("............................................................................................................")
    print("overall Val Accuracy Personalisation",overall_val_acc[conv:])
    print("............................................................................................................")
    print("overall Test Accuracy",overall_test_acc)
    print("............................................................................................................")
    print("Individual clients validation accuracies")
    for _,client in clients.items():
        print(client ,client.val_acc)
    print("............................................................................................................")
    print("Individual clients test accuracies")
    for _,client in clients.items():
        print(client ,client.test_acc)
    print("............................................................................................................")
        


    # calculating the train and test standarad deviation and teh confidence intervals 
    X = range(args.epochs)
    all_clients_stacked_train = np.array([client.train_acc[:conv] for _,client in clients.items()])
    all_clients_stacked_test = np.array([client.test_acc[:conv] for _,client in clients.items()])
    all_clients_stacked_val = np.array([client.val_acc[:conv] for _,client in clients.items()])
    epochs_train_std = np.std(all_clients_stacked_train,axis = 0, dtype = np.float64)
    epochs_val_std = np.std(all_clients_stacked_val,axis = 0, dtype = np.float64)
    epochs_test_std = np.std(all_clients_stacked_test,axis = 0, dtype = np.float64)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Validation Standard Deviations  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(epochs_val_std)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Test Standard Deviations  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
    print(epochs_test_std)
    if(args.setting=='setting2'):
            print("Total Runs",len(overall_val_acc))
            print("Total Runs gen",len(overall_val_acc[:conv]))
            print("Total Runs per",len(overall_val_acc[conv:]))
            max_gen = max(overall_val_acc[:conv])
            #max_pers = max(overall_val_acc[conv:])
            max_f1_gen = max(overall_macro_f1_score[:conv])
            max_wf1_gen = max(overall_weighted_f1_score[:conv])
            #max_f1_per = max(overall_macro_f1_score[conv:])
            index_max_gen = overall_val_acc.index(max_gen)
            #index_max_per = overall_val_acc.index(max_pers)
            index_max_f1_gen = overall_macro_f1_score.index(max_f1_gen)
            index_max_wf1_gen = overall_weighted_f1_score.index(max_wf1_gen)
            #index_max_f1_per = overall_macro_f1_score.index(max_f1_per)
            ######################################################################################
            print(f'Epoch of Gen Convergence: {index_max_gen+1} Gen Val Accuracy at that epoch:{overall_val_acc[index_max_gen]} Std at Gen Convergence: {epochs_val_std[index_max_gen]} Gen Train accuracy at that epoch: {overall_train_acc[index_max_gen]} Gen Max Macro F1 Score: {overall_macro_f1_score[index_max_f1_gen]} Epoch for Max f1 Score: {index_max_f1_gen+1} Gen Weighted F1 Score:{overall_weighted_f1_score[index_max_gen]}')
            #print(f'Epoch of Personal Convergence: {index_max_per+1} Per Val Accuracy at that epoch:{overall_val_acc[index_max_per]} Std at Per Convergence: {epochs_val_std[index_max_per]} Personal Train accuracy at that epoch: {overall_train_acc[index_max_per]} Per Max F1 Score: {overall_macro_f1_score[index_max_f1_per]} Epoch for Max f1 Score: {index_max_f1_per+1}')
    else:
            index_max=overall_val_acc.index(max_acc)
            print(f'Epoch of Convergence: {index_max+1} Std at Convergence: {epochs_val_std[index_max]} Train accuracy at that epoch: {overall_train_acc[index_max]}')
    ######################################################################################
    print("###########################################################################################################################")
    print(per_conv)
    acc_test_list = {}
    acc_train_list = {}
    acc_val_list = {}
    macro_f1_list = {}
    weighted_f1_list = {}
    print("At the time of generalisation phase:")
    for _,client in clients.items():
            indx = gen_conv-1
            acc_train_list[_] = client.train_acc[indx]
            acc_val_list[_] = client.val_acc[indx]
            macro_f1_list[_] = client.macro_f1_score[indx]
            weighted_f1_list[_] = client.weighted_f1_score[indx]
            print(f'{client} For Val Acc: Epoch of Gen Convergence: {indx+1}  Val Accuracy at that epoch:{client.val_acc[indx]} Train accuracy at that epoch: {client.train_acc[indx]} Macro F1 Score: {client.macro_f1_score[indx]} Weighted F1 Score: {client.weighted_f1_score[indx]}')
    print("..............................................................")
    if(args.setting=='setting2'): 
        for _,client in clients.items():
            indx = client.macro_f1_score.index(max(client.macro_f1_score[:conv]))
            print(f'{client} For F1 Score: Epoch of Gen Convergence: {indx+1} Gen Val Accuracy at that epoch:{client.val_acc[indx]} Personal Train accuracy at that epoch: {client.train_acc[indx]} F1 Score: {client.macro_f1_score[indx]}')
    
    print("..............................................................")
    print("Mean of Train Accuracies: ", np.mean(list(acc_train_list.values())))
    print("Std train:", statistics.stdev(list(acc_train_list.values())))
    print("Mean of Val Accuracies: ", np.mean(list(acc_val_list.values())))
    print("Std Val: ", statistics.stdev(list(acc_val_list.values())))
    print("Mean of Macro F1-score : ", np.mean(list(macro_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(macro_f1_list.values())))
    print("Mean of Weighted F1-score : ", np.mean(list(weighted_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(weighted_f1_list.values())))
    
    print("..............................................................")
    if(args.setting=='setting2'): 
        print("At the time of personalisation phase:")
        for _,client in clients.items():
            indx = per_conv[_]-1
            acc_train_list[_] = client.train_acc[indx]
            acc_val_list[_] = client.val_acc[indx]
            macro_f1_list[_] = client.macro_f1_score[indx]
            weighted_f1_list[_] = client.weighted_f1_score[indx]
            print(f'{client} For Val Acc: Epoch of Personal Convergence: {indx+1} Per Val Accuracy at that epoch:{client.val_acc[indx]} Personal Train accuracy at that epoch: {client.train_acc[indx]} Macro F1 Score: {client.macro_f1_score[indx]} Weighted F1 Score: {client.weighted_f1_score[indx]}')
    print("..............................................................")
    if(args.setting=='setting2'): 
        for _,client in clients.items():   
            indx = client.macro_f1_score.index(max(client.macro_f1_score[conv:]))
            print(f'{client} For F1 Score: Epoch of Personal Convergence: {indx+1} Per Val Accuracy at that epoch:{client.val_acc[indx]} Personal Train accuracy at that epoch: {client.train_acc[indx]} F1 Score: {client.macro_f1_score[indx]}')
    print("..............................................................")
    print("Mean of Train Accuracies: ", np.mean(list(acc_train_list.values())))
    print("Std train:", statistics.stdev(list(acc_train_list.values())))
    print("Mean of Val Accuracies: ", np.mean(list(acc_val_list.values())))
    print("Std Val: ", statistics.stdev(list(acc_val_list.values())))
    print("Mean of F1-score : ", np.mean(list(macro_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(macro_f1_list.values())))
    print("Mean of Weighted F1-score : ", np.mean(list(weighted_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(weighted_f1_list.values())))
    #print("Mean of Test Accuracies: ", np.mean(list(acc_test_list.values())))
    #print("Std across entropies: ", statistics.stdev(list(acc_test_list.values())))
    #########################################################################################
   
    
    
    
    
    print("#####################################################second######################################################################")
    
    
    acc_test_list = {}
    acc_train_list = {}
    acc_val_list = {}
    macro_f1_list = {}
    print("At the time of generalisation phase:")
    for _,client in clients.items():
            indx = overall_val_acc.index(max(overall_val_acc[:conv]))
            acc_train_list[_] = client.train_acc[indx]
            acc_val_list[_] = client.val_acc[indx]
            macro_f1_list[_] = client.macro_f1_score[indx]
            weighted_f1_list[_] = client.weighted_f1_score[indx]
            print(f'{client} For Val Acc: Epoch of Gen Convergence: {indx+1}  Val Accuracy at that epoch:{client.val_acc[indx]} Train accuracy at that epoch: {client.train_acc[indx]} Macro F1 Score: {client.macro_f1_score[indx]} Weighted F1 Score: {client.weighted_f1_score[indx]}')
    print("..............................................................")
    if(args.setting=='setting2'): 
        for _,client in clients.items():
            indx = client.macro_f1_score.index(max(client.macro_f1_score[:conv]))
            print(f'{client} For F1 Score: Epoch of Gen Convergence: {indx+1} Gen Val Accuracy at that epoch:{client.val_acc[indx]} Personal Train accuracy at that epoch: {client.train_acc[indx]} F1 Score: {client.macro_f1_score[indx]}')
    
    print("..............................................................")
    print("Mean of Train Accuracies: ", np.mean(list(acc_train_list.values())))
    print("Std train:", statistics.stdev(list(acc_train_list.values())))
    print("Mean of Val Accuracies: ", np.mean(list(acc_val_list.values())))
    print("Std Val: ", statistics.stdev(list(acc_val_list.values())))
    print("Mean of F1-score : ", np.mean(list(macro_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(macro_f1_list.values())))
    print("Mean of Weighted F1-score : ", np.mean(list(weighted_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(weighted_f1_list.values())))
    print("..............................................................")
    if(args.setting=='setting2'): 
        print("At the time of personalisation phase:")
        for _,client in clients.items():
            indx = client.val_acc.index(max(client.val_acc[conv:]))
            print(indx)
            #acc_test_list[_] = client.test_acc[indx]
            acc_train_list[_] = client.train_acc[indx]
            acc_val_list[_] = client.val_acc[indx]
            macro_f1_list[_] = client.macro_f1_score[indx]
            weighted_f1_list[_] = client.weighted_f1_score[indx]
            print(f'{client} For Val Acc: Epoch of Personal Convergence: {indx+1} Per Val Accuracy at that epoch:{client.val_acc[indx]} Personal Train accuracy at that epoch: {client.train_acc[indx]} Macro F1 Score: {client.macro_f1_score[indx]} Weighted F1 Score: {client.weighted_f1_score[indx]}')
    print("..............................................................")
    if(args.setting=='setting2'): 
        for _,client in clients.items():   
            indx = client.macro_f1_score.index(max(client.macro_f1_score[conv:]))
            print(f'{client} For F1 Score: Epoch of Personal Convergence: {indx+1} Per Val Accuracy at that epoch:{client.val_acc[indx]} Personal Train accuracy at that epoch: {client.train_acc[indx]} F1 Score: {client.macro_f1_score[indx]}')
    print("..............................................................")
    print("Mean of Train Accuracies: ", np.mean(list(acc_train_list.values())))
    print("Std train:", statistics.stdev(list(acc_train_list.values())))
    print("Mean of Val Accuracies: ", np.mean(list(acc_val_list.values())))
    print("Std Val: ", statistics.stdev(list(acc_val_list.values())))
    print("Mean of F1-score : ", np.mean(list(macro_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(macro_f1_list.values())))
    print("Mean of Weighted F1-score : ", np.mean(list(weighted_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(weighted_f1_list.values())))
    #print("Mean of Test Accuracies: ", np.mean(list(acc_test_list.values())))
    #print("Std across entropies: ", statistics.stdev(list(acc_test_list.values())))
    #########################################################################################
    print("###########################################################################################################################")
    #if(args.setting=='setting2'): 
    #    for client_id,client in clients.items():
    #        print(client_id,flag[client_id])
    if(args.setting=='setting2'): 
        print("At the time of personalisation phase:")
        for _,client in clients.items():
            print("length:",len(client.val_acc))
            indx = client.val_acc.index(client.val_acc[-6])
            #print(indx)
            #acc_test_list[_] = client.test_acc[indx]
            acc_train_list[_] = client.train_acc[indx]
            acc_val_list[_] = client.val_acc[indx]
            macro_f1_list[_] = client.macro_f1_score[indx]
            weighted_f1_list[_] = client.weighted_f1_score[indx]
            print(f'{client} For Val Acc: Epoch of Personal Convergence: {indx+1} Per Val Accuracy at that epoch:{client.val_acc[indx]} Personal Train accuracy at that epoch: {client.train_acc[indx]} Macro F1 Score: {client.macro_f1_score[indx]} Weighted F1 Score: {client.weighted_f1_score[indx]}')
    print("..............................................................")
    
    print("Mean of Train Accuracies: ", np.mean(list(acc_train_list.values())))
    print("Std train:", statistics.stdev(list(acc_train_list.values())))
    print("Mean of Val Accuracies: ", np.mean(list(acc_val_list.values())))
    print("Std Val: ", statistics.stdev(list(acc_val_list.values())))
    print("Mean of F1-score : ", np.mean(list(macro_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(macro_f1_list.values())))
    print("Mean of Weighted F1-score : ", np.mean(list(weighted_f1_list.values())))
    print("Std Val: ", statistics.stdev(list(weighted_f1_list.values())))