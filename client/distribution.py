import numpy as np
import torch, random, os
import matplotlib.pyplot as plt

def data_distribution(config, trainset):
    labels = []
    base_dir = os.getcwd()
    storepath = os.path.join(base_dir, 'Distribution/', config['dataset']+'/')
    for i in range(len(trainset)):
        labels.append(trainset[i][1])
    #print(trainset)
    unique_labels = np.unique(np.array(labels))
    label_index_list = {}
    for key in unique_labels:
        label_index_list[key] = []
    for index, label in enumerate(labels):
        label_index_list[label].append(index)
    
    seed = 10
    random.seed(seed)
    num_users = 150
    num_classes = len(unique_labels)
    K = config['niid']
    #print(K)
    q_step = (1 - (1/num_classes))/(K-1)
    print(q_step)
    for i in range(len(label_index_list)):
        random.shuffle(label_index_list[i])
    for j in range(K):
        dist = np.random.uniform(q_step, (1+j)*q_step, (num_classes, num_users))
        if j != 0:
            data_presence_indicator = np.random.choice([0, 1], (num_classes, num_users), p=[j*q_step, 1-(j*q_step)])
            dist = np.multiply(dist,data_presence_indicator)
        psum = np.sum(dist, axis=1)
        for i in range(dist.shape[0]):
            dist[i] = dist[i]*len(label_index_list[i])/psum[i]
        dist = np.floor(dist).astype(int)
        
        ## If any client does not get any data
        gainers = list(np.where(np.sum(dist, axis=0) != 0))[0]
        if len(gainers) < num_users:
            losers = list(np.where(np.sum(dist, axis=0) == 0))[0]
            donors = np.random.choice(gainers, len(losers))
            for index, donor in enumerate(donors):
                avail_digits = np.where(dist[:,donor] != 0)[0]
                for digit in avail_digits:
                    transfer_frac = np.random.uniform(0.1,0.9)
                    num_transfer = int(dist[digit, donor]*transfer_frac)
                    dist[digit, donor] = dist[digit, donor] - num_transfer
                    dist[digit, losers[index]] = num_transfer
        
        for num in range(num_classes):
            while dist[num].sum() != len(label_index_list[num]):
                index = random.randint(0,num_users-1)
                if dist[num].sum() < len(label_index_list[num]):
                    dist[num][index]+=1
                else:
                    dist[num][index]-=1
        
        split = [[] for i in range(num_classes)]
        for num in range(num_classes):
            start = 0
            for i in range(num_users):
                split[num].append(label_index_list[num][start:start+dist[num][i]])
                start = start+dist[num][i]
        
        datapoints = [[] for i in range(num_users)]
        class_histogram = [[] for i in range(num_users)]
        class_stats= [[] for i in range(num_users)]
        for i in range(num_users):
            for num in range(num_classes):
                datapoints[i] += split[num][i]
                class_histogram[i].append(len(split[num][i]))
                if(len(split[num][i])==0):
                    class_stats[i].append(0)
                else:
                    class_stats[i].append(1)
                
        #file_name = storepath 
        if not os.path.exists(storepath):
            os.makedirs(storepath)
        file_name = 'data_split_niid_'+ str(K)+'.pt'

        #print(storepath+file_name)
        torch.save({'datapoints': datapoints, 'histograms': class_histogram, 'class_statitics': class_stats}, storepath + file_name)
        
    