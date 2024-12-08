import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

# Objet pour la batch normalisation 
class my_conv(nn.Module):
    """Objet pour la batch normalisation 
    
    Args:
        in_size: Taille de l'input
        out_size: Taille en sortie
    """
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))
    
class bottleneck(nn.Module):
    """Structure en bottleneck à utiliser dans un réseau 
    
    Args:
        in_size: Taille de l'input
        hidden_dim: Taille des couches cachées 
        out_size: Taille en sortie
        kernel_size: Taille de la fenêtre de convolution 
    """
    def __init__(self, in_size=3, hidden_dim=32, out_size=10, kernel_size=3):
        super().__init__()
        #Réduction
        self.conv_a = my_conv(in_size, hidden_dim, kernel_size)
        #Maintient
        self.conv_b = my_conv(hidden_dim, hidden_dim, kernel_size)
        #Augmentation
        self.conv_c = my_conv(hidden_dim, out_size, kernel_size)
        
        #connexion résiduelle ? 
        self.res_true = (in_size == out_size)
    
    def forward(self, x):
        res = 0
        if(self.res_true): 
            res = x
        
        x = self.conv_c(self.conv_b(self.conv_a(x)))
        x = x + res
        return x
    
class bottleneck_stack(nn.Module):
    def __init__(self, in_size=3, hidden_dim=32, out=10, kernel_size=3):
        super().__init__()
        self.bottleneck_a = bottleneck(in_size, hidden_dim, hidden_dim, kernel_size)
        self.bottleneck_b = bottleneck(hidden_dim, hidden_dim, hidden_dim, kernel_size)
        self.bottleneck_c = bottleneck(hidden_dim, hidden_dim, out, kernel_size)
        # maxpool plutôt que stride 2 pour pouvoir effectuer les connexions résiduelles 
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.a = nn.Linear(hidden_dim * 8 * 8, hidden_dim)
        self.b = nn.Linear(hidden_dim, hidden_dim)
        self.c = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out) 

    def forward(self, x):
                    
        x = self.bottleneck_a(x)
        x = self.bottleneck_b(x)
        x = self.bottleneck_c(x)
        
        x = x.view(x.size(0), -1)        

        return x

def accuracy(pred, label):
    correct_predictions = (pred == label).sum().item()
    
    total_predictions = label.size(0)
    return correct_predictions / total_predictions
            
def fit_one_cycle(model, train_dataloader, valid_dataloader, loss_fn, dev):
    """
    Réalise une itération complète (epoch) sur un DataLoader pour l'entraînement et la validation.
    """     
   
    model.train() 
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    learningRate = 0.0001
    opt = torch.optim.Adam(model.parameters(), lr=learningRate)

    #boucle d'entrainement
    for batch, label in train_dataloader:

        #training
        batch = batch.to(dev)
        label = label.to(dev)

        output = model(batch) 
        #print((output.size()))
        #print((label.size()))
        
        loss = loss_fn(output, label) 
        loss.backward()
        #update parameters
        opt.step()
        #reset gradients to 0
        opt.zero_grad()
        
    
    #boucle de validation 
    for batch, label in valid_dataloader:

        batch = batch.to(dev)
        label = label.to(dev)

        output = model(batch) 
        #print((output.size()))
        #print((label.size()))
        
        loss = loss_fn(output, label) 
        loss.backward()
        #update parameters
        opt.step()
        #reset gradients to 0
        opt.zero_grad()

        #stats
        train_loss += loss.item()
        _, preds = torch.max(output, dim=1)
        train_correct += (preds == label).sum().item()
        train_total += label.size(0)

    train_accuracy = train_correct / train_total
    train_loss /= len(valid_dataloader)
    
    

    print(f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    return train_loss, train_accuracy


def train_model(model, train_dataloader, valid_dataloader, loss_fn, num_epochs, dev):
    """
    Entraîne le modèle sur plusieurs epochs en utilisant fit_one_cycle().
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        train_loss, train_accuracy = fit_one_cycle(model, train_dataloader, valid_dataloader, loss_fn, dev)
        
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Chargement du Training Set 
transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root = "./", train = True, download = True, transform=transform)
valid_set = torchvision.datasets.CIFAR10(root = "./", train = False, download = True, transform=transform)



# Dataloader - Permet de charger un dataset dans un objet Pytorch, Batch = dim supp pour traiter plusieurs données simultanément
train_dataloaded = torch.utils.data.DataLoader(train_set, batch_size=128)
valid_dataloaded = torch.utils.data.DataLoader(valid_set, batch_size=128)


# Élément 0 du dataloader 
# Séparation entre batch et label 
batch, label = ((next(iter(train_dataloaded))))
batch_valid, label_valid = ((next(iter(valid_dataloaded))))


# Lancement de la classe "My_class" (plus haut)
modele_multicouche = bottleneck_stack(in_size=3, hidden_dim=32, out=10, kernel_size=3)
'''output = modele_multicouche.forward(batch[0].view(-1,3*32*32))'''

#Envoie du model sur le GPU 
modele_multicouche.to(dev)

# Chargemnet de la fonction de loss (ici cross-entropy)
lossFn = F.cross_entropy

print("test sur une epoch :\n")
fit_one_cycle(modele_multicouche, train_dataloaded, valid_dataloaded, lossFn, dev)
print("\n---------------\n\n")

print("test sur plusieurs epochs")
train_model(modele_multicouche, train_dataloaded, valid_dataloaded, lossFn, num_epochs=50, dev=dev)
print("\n---------------\n\n")


