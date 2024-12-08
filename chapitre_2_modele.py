import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

class my_conv(nn.Module):
    """Objet pour la batch normalisation 
    
    Args:
        in_size: Taille de l'input
        out_size: Taille en sortie
    """
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Définition des couches 
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Application des couches sur les données 
        return self.activation(self.norm(self.conv(x)))
    
class My_class(nn.Module):
    """Deuxième modèle, avec des convolutions 2D & activation 
    """
    def __init__(self, hidden_dim=32, out=10):
        super().__init__()
        self.conv_a = my_conv(3, hidden_dim)
        self.conv_b = my_conv(hidden_dim, hidden_dim)
        self.conv_c = my_conv(hidden_dim, hidden_dim)
        # maxpool plutôt que stride 2 pour pouvoir effectuer les connexions résiduelles 
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.a = nn.Linear(hidden_dim * 8 * 8, hidden_dim)
        self.b = nn.Linear(hidden_dim, hidden_dim)
        self.c = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out) 

    def forward(self, x):
        # Application des couches sur les données 
                
        x = self.conv_a(x)
        x = self.maxpool(x)
        
        x1 = x
        x = self.conv_b(x) + x1
        x = self.maxpool(x)
        
        x2 = x
        x = self.conv_c(x) + x2
        
        x = x.view(x.size(0), -1)
        
        x = self.out(self.c(self.b(self.a(x))))
        return x

def accuracy(pred, label):
    """Calcul des statistiques sur la précisions du modèle

    Args:
        pred: Les prédictions effectuées par le modèle
        label: Les valeurs attendues

    Returns:
        prédictions correctes, prédictions 
    """
    correct_predictions = (pred == label).sum().item()
    
    total_predictions = label.size(0)
    return correct_predictions / total_predictions
            
def fit_one_cycle(model, train_dataloader, valid_dataloader, loss_fn, dev):
    """
    Réalise une itération complète (epoch) sur un DataLoader pour l'entraînement et la validation.
    
    Args:
        model: Le modèle à entraîner
        train_dataloader: DataLoader pour les données d'entraînement
        valid_dataloader: DataLoader pour les données de validation
        loss_fn: Fonction de loss
        dev: Le device (CPU/GPU)
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
        
        
# Main



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
modele_multicouche = My_class()
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


