import os, json
import matplotlib.pyplot as plt


def values(txt_path):
    accuracy = []
    round = []
    with open(txt_path) as f:
        for line in f.readlines():
            #print(line.split(','))
            accuracy.append(float(line.split(',')[1].split(':')[1]))
            round.append(int(line.split(',')[2].split(':')[1].split('}')[0]))
    result = (accuracy, round)
    return result

path = './results'
datasets = os.listdir(path)
dataset = 'FashionMNIST'
algorithms = os.listdir(os.path.join(path, dataset))
print(algorithms)
niids = os.listdir(os.path.join(path, dataset, algorithms[0]))
niids = [int(i) for i in niids]

#media = './media'

dataset = 'FashionMNIST'

fedavg_value  =[]
fedadam_value = []
feddyn_value = []
fedavgm_value = []
fedadagrad_value = []
mimelite_value = []
fedyogi_value = []
mime_value = []
scaffold_value = []
for i in range(len(niids)):
    fedavg_path = sorted(os.listdir(os.path.join(path, dataset, 'fedavg', str(i+1))),reverse= True)[0]
    fedavg_path = os.path.join(path, dataset, 'fedavg', str(i+1), fedavg_path, 'FL_results.txt')
    fedavg_value.append(values(fedavg_path))
    #fedavg_value[i] =
    
    fedadam_path = sorted(os.listdir(os.path.join(path, dataset, 'fedadam', str(i+1))),reverse= True)[0]
    fedadam_path = os.path.join(path, dataset, 'fedadam', str(i+1), fedadam_path, 'FL_results.txt')
    #print(fedavg_path)
    fedadam_value.append(values(fedadam_path))

    feddyn_path = sorted(os.listdir(os.path.join(path, dataset, 'feddyn', str(i+1))),reverse= True)[0]
    feddyn_path = os.path.join(path, dataset, 'feddyn', str(i+1), feddyn_path, 'FL_results.txt')
    #print(fedavg_path)
    feddyn_value.append(values(feddyn_path))

    fedavgm_path = sorted(os.listdir(os.path.join(path, dataset, 'fedavgm', str(i+1))),reverse= True)[0]
    fedavgm_path = os.path.join(path, dataset, 'fedavgm', str(i+1), fedavgm_path, 'FL_results.txt')
    #print(fedavg_path)
    fedavgm_value.append(values(fedavgm_path))

    fedadagrad_path = sorted(os.listdir(os.path.join(path, dataset, 'fedadagrad', str(i+1))),reverse= True)[0]
    fedadagrad_path = os.path.join(path, dataset, 'fedadagrad', str(i+1), fedadagrad_path, 'FL_results.txt')
    #print(fedavg_path)
    fedadagrad_value.append(values(fedadagrad_path))

    #mimelite_path = sorted(os.listdir(os.path.join(path, dataset, 'mimelite', str(i+1))),reverse= True)[0]
    #mimelite_path = os.path.join(path, dataset, 'mimelite', str(i+1), mimelite_path, 'FL_results.txt')
    #print(fedavg_path)
    #mimelite_value.append(values(mimelite_path))

    fedyogi_path = sorted(os.listdir(os.path.join(path, dataset, 'fedyogi', str(i+1))),reverse= True)[0]
    fedyogi_path = os.path.join(path, dataset, 'fedyogi', str(i+1), fedyogi_path, 'FL_results.txt')
    #print(fedavg_path)
    fedyogi_value.append(values(fedyogi_path))

    #mime_path = sorted(os.listdir(os.path.join(path, dataset, 'mime', str(i+1))),reverse= True)[0]
    #mime_path = os.path.join(path, dataset, 'mime', str(i+1), mime_path, 'FL_results.txt')
    #print(mime_path)
    #mime_value.append(values(mime_path))

    #scaffold_path = sorted(os.listdir(os.path.join(path, dataset, 'scaffold', str(i+1))),reverse= True)[0]
    #scaffold_path = os.path.join(path, dataset, 'scaffold', str(i+1), scaffold_path, 'FL_results.txt')
    #print(fedavg_path)
    #scaffold_value.append(values(scaffold_path))


total = [fedavg_value, fedadam_value, feddyn_value, fedavgm_value, 
         fedadagrad_value, fedyogi_value]

#plot non-iid vs accuracy
fedavg_value_1  =[]
fedadam_value_1 = []
feddyn_value_1 = []
fedadagrad_value_1 = []
mimelite_value_1 = []
fedyogi_value_1 = []
mime_value_1 = []
scaffold_value_1 = []

for i in range(len(niids)):
    fedavg_value_1.append(fedavg_value[i][0][-1])
    fedadam_value_1.append(fedadam_value[i][0][-1])
    feddyn_value_1.append(feddyn_value[i][0][-1])
    fedadagrad_value_1.append(fedadagrad_value[i][0][-1])
    #mimelite_value_1.append(mimelite_value[i][0][-1])
    fedyogi_value_1.append(fedyogi_value[i][0][-1])
    #mime_value_1.append(mime_value[i][0][-1])
    #scaffold_value_1.append(scaffold_value[i][0][-1])
##plot non-iid vs accuracy
X = sorted(niids)
plt.plot(X,fedavg_value_1, 'o--', label = "fedavg")
# plotting the line 2 points 
plt.plot(X, fedadam_value_1, 'o--',  label = "fedadam")
plt.plot(X, feddyn_value_1, 'o--',  label = "feddyn")
plt.plot(X, fedadagrad_value_1, 'o--',  label = "fedadagrad") 
#plt.plot(X, mimelite_value_1, 'o--',  label = "mimelite") 
plt.plot(X, fedyogi_value_1, 'o--',  label = "fedyogi") 
#plt.plot(X, mime_value_1, 'o--',  label = "mime") 
#plt.plot(X, scaffold_value_1, 'o--',  label = "scaffold") 

plt.xlabel('niid')
# naming the y axis
plt.ylabel('accuracy')
  
# giving a title to my graph
#plt.title('')
plt.legend()  
# function to show the plot
plt.show()
plt.savefig('./media/Accuracy.png')



#plot round vs accuracy
##plot non-iid vs accuracy
for i in range(len(niids)):
    fig, ax = plt.subplots()
    X = fedavg_value[i][1]
    ax.plot(X, fedavg_value[i][0], label="FedAvg")
    ax.plot(X, fedadam_value[i][0], label="FedAdam")
    ax.plot(X, feddyn_value[i][0],  label="FedDyn")
    ax.plot(X, fedadagrad_value[i][0],  label="FedAdaGrad")
    #ax.plot(X, mimelite_value[i][0],  label="MimeLite")
    ax.plot(X, fedyogi_value[i][0],  label="FedYogi")
    #ax.plot(X, mime_value[i][0],  label="Mime")
    #ax.plot(X, scaffold_value[i][0],  label="Scaffold")

    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.set_title(f"NIID Dataset {i+1}")
    ax.legend()

    plt.tight_layout()
    plt.savefig('./media/niid_'+ str(i+1))
    

    
    
for i in range(len(total)):
    fig, ax = plt.subplots()
    for d in total[i]:
        ax.plot(d[1], d[0], label="K = {}".format(total[i].index(d)+1))

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    #print(algorithms)
    ax.set_title(algorithms[i])
    ax.legend()
    plt.show()
    plt.savefig('./media/Al_' + str(i))