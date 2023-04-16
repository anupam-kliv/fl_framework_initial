import torch
from tqdm import tqdm
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(config):
    trainset, testset = get_data(config)
    testloader = DataLoader(testset, batch_size=config['batch_size'])
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    #print(num_examples)
    print('Data load is done') 
    return testloader, num_examples

### Load different dataset
def get_data(config):
    print("Dataset Name: ",config['dataset'])
    if config['dataset'] == 'MNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.MNIST(root='./MNIST', train=True, download=True, transform=apply_transform)
        testset = datasets.MNIST(root='./MNIST', train=False, download=True, transform=apply_transform)
    if config['dataset'] == 'FashionMNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.FashionMNIST(root='./FashionMNIST', train=True, download=True, transform=apply_transform)
        testset = datasets.FashionMNIST(root='./FashionMNIST', train=False, download=True, transform=apply_transform)

    if config['dataset'] == 'CIFAR10':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=apply_transform)

    if config['dataset'] == 'CIFAR100':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR100(root='./CIFAR100', train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR100(root='./CIFAR100', train=False, download=True, transform=apply_transform)

    return trainset, testset


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)        
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = self.fc1(x)
        x = self.relu(x) 
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x

def get_net(config):
    if config["net"] == 'LeNet':
        net = LeNet()
    if config["net"] == 'resnet18':
        net = models.resnet18()
    if config["net"] == 'resnet50':
        net = models.resnet18()
    if config["net"] == 'vgg16':
        net = models.vgg16()
    if config['net'] == 'AlexNet':
        net = models.AlexNet()
    print('model is loaded')
    return net

def train_model(net, trainloader):
    #print('tranning is started')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    #for images, labels in trainloader:
        #images, labels = images.to(device), labels.to(device)
        #print(images.shape)
    dataiter = iter(trainloader)
    #print(dataiter)
    images, labels = next(dataiter)
    outputs = net(images)
    #print(outputs.shape)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return net

def test_model(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader) :
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def save_intial_model(config):
    testloader, _ = load_data(config)
    #print(config['initial_model_path'])
    net = get_net(config)
    net = train_model(net, testloader)
    torch.save(net.state_dict(), 'server/src/initial_model.pt')
    print('Initial model is saved')

