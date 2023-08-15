# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    #train_transforms = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #   transforms.ToTensor(),
    #    transforms.Normalize(
    #        mean=[0.485, 0.456, 0.406],
    #        std=[0.229, 0.224, 0.225]
    #    )
    #])

    pretrained_weights = models.ResNet50_Weights.IMAGENET1K_V2
    
    train_dataset = datasets.ImageFolder(root="dataset/", transform=pretrained_weights.transforms())
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False,
                              pin_memory=True, num_workers=8)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)
    model = models.resnet50(weights=pretrained_weights)
    #print(model)



    #model = nn.Module()
    embeddings = []
    embedding_size = 2048 # Dummy variable, replace with the actual embedding size once you 
    # pick your model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates. 
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print()
    model.eval() #sets the neural network to evaluation mode
    #print(model)

    for i, (images, _) in enumerate (train_loader):
        batch_size = images.shape[0]
        #print('batch size:', batch_size)
        #print('start batch:',i)
        #print('shape of embeddings is:',embeddings.shape)
        #print('shape of my batch', model(images).detach().numpy().reshape(64, embedding_size).shape)
        embeddings[i*64:(i*64)+batch_size] = model(images).detach().numpy().reshape(batch_size,embedding_size)#gibt mir ein vektor von embeddings von unserem bild welcher horizontal eingefügt wird
        print('finish batch:',i)
   
    np.save('dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []  #here we have a triplets list with triplets
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')
    # TODO: Normalize the embeddings across the dataset
    normalize(embeddings, axis=1, norm='l2')

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:  #wir gehen durch jedes triplet
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]])) #stacks arrays in one array so all our embeddings are now in one array
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(-1)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 8):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super(Net,self).__init__()
        
        
        self.fc1 = nn.Linear(6144, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.Tanh()
        self.relu = nn.Tanh()
        self.dropout = nn.Dropout(p=0.4)
    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        m = nn.Sigmoid()
        x1 = self.fc1(x)
        #print('nach 1.',x)
        x = self.bn1(x1)
        x = self.relu(x)
        x = self.dropout(x)
        #print('nach 1. relu',x)
        x2 = self.fc2(x)
        #print('nach 2.layer',x)
        x = self.bn2(x2)
        x = self.relu(x)
        x = self.dropout(x)
        #print('nach 2. relu',x)
        x3 = self.fc3(x)
        #print('nach 3.layer',x)
        x = self.bn3(x3)
        x = self.relu(x)
        #x = self.dropout(x)
        #print ('nach 3. relu',x)
        x = self.fc4(x)
        x = self.bn4(x)

        x = m(x)
        x = self.fc5(x)
        #x = x + x3
        #x = self.relu(x)
        #x = x + x2
        #x = self.relu(x)
        #x = x + x1
        #x = self.relu(x)
        #print(' am schluss',x)
        return x
def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 8
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.

    #lossyfunc = nn.BCEWithLogitsLoss()
    lossyfunc = nn.CrossEntropyLoss()
    #lossyfunc = nn.TripletMarginLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    running_loss = 0
    print('now we start training')
    for epoch in range(n_epochs):        
        print(epoch)
        for i,[X, y] in enumerate(train_loader):
            #print('type X', type(X))
            #print('type X', type(y))
            optimizer.zero_grad() #makes gradient zero
            #print('batch:',i)
            #print('shape of X:',X.shape)
            #print('shape of mode:', model)
            output = torch.squeeze(model(X)) #feeds the batch through the model
            #print('shape of output of model',output.shape)
            #print('shape of y:',y.shape)
            y = y.float() #makes y to a float
            #print('this is our input from embeddings',X)
            #print('should be value',y)
            #print('predicted value',output)
            loss = lossyfunc(output,y)  
            loss.backward()
            optimizer.step()
            # print statistics

            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    print('Finished Training')
    return model

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0] = 1
            predicted[predicted < 0] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
