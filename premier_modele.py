import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class My_class(nn.Module):
    """Premier modèle à base de tranformations linéaires
    
    Args:
        len_input: Taille de l'input 
        len_out_a: Nombre de neurones dans la première couche cachée
        len_out_b: Nombre de neurones dans la deuxième couche cachée
        len_output: Taille de l'output 
    """
    def __init__(self, len_input = 3, len_out_a=32, len_out_b=24, len_output=10):
        super().__init__()
        # Définition des couches 
        self.a = nn.Linear(len_input, len_out_a)
        self.b = nn.Linear(len_out_a, len_out_b)
        self.c = nn.Linear(len_out_b, len_output)
    def forward(self, x):
        # Application des couches sur les données 
        return self.c(self.b(self.a(x)))

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
    return correct_predictions, total_predictions
            
def fit_one_cycle(model, train_dataloader, valid_dataloader, loss_fn):
    """
    Réalise une itération complète (epoch) sur un DataLoader pour l'entraînement et la validation.
    
    Args:
        model: Le modèle à entraîner
        train_dataloader: DataLoader pour les données d'entraînement
        valid_dataloader: DataLoader pour les données de validation
        loss_fn: Fonction de loss
    """     
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    learningRate = 0.001
    opt = torch.optim.SGD(model.parameters(), lr=learningRate)

    # Boucle d'entrainement
    for batch, label in train_dataloader:
        batch = batch.view(-1, 3*32*32)
        output = model(batch) 
        #print((output.size()))
        #print((label.size()))
        
        loss = loss_fn(output, label) 
        loss.backward()
        # Update parameters
        opt.step()
        # Reset gradients to 0
        opt.zero_grad()
        
    
    # Boucle de validation 
    for batch, label in valid_dataloader:
        batch = batch.view(-1, 3*32*32)
        output = model(batch) 
        #print((output.size()))
        #print((label.size()))
        
        loss = loss_fn(output, label) 
        loss.backward()
        # Update parameters
        opt.step()
        # Reset gradients to 0
        opt.zero_grad()

        # Stats
        train_loss += loss.item()
        _, preds = torch.max(output, dim=1)
        train_correct += (preds == label).sum().item()
        train_total += label.size(0)
    # Calculs de précision et de la loss 
    train_accuracy = train_correct / train_total
    train_loss /= len(valid_dataloader)
    print(f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    
    return train_loss, train_accuracy


def train_model(model, train_dataloader, valid_dataloader, loss_fn, num_epochs):
    """Entraîne le modèle sur plusieurs epochs en utilisant fit_one_cycle()
    
    Args:
        model: Le modèle à entraîner
        train_dataloader: DataLoader pour les données d'entraînement
        valid_dataloader: DataLoader pour les données de validation
        loss_fn: Fonction de loss
        num_epochs: Nombre d'epochs
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        fit_one_cycle(model, train_dataloader, valid_dataloader, loss_fn)


# Main 

# Chargement du Training Set 
transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root = "./", train = True, download = True, transform=transform)
valid_set = torchvision.datasets.CIFAR10(root = "./", train = False, download = True, transform=transform)


# Dataloader - Permet de charger un dataset dans un objet Pytorch, Batch = dim supp pour traiter plusieurs données simultanément
train_dataloaded = torch.utils.data.DataLoader(train_set, batch_size=32)
valid_dataloaded = torch.utils.data.DataLoader(valid_set, batch_size=32)


# Élément 0 du dataloader 
# Séparation entre batch et label 
batch, label = ((next(iter(train_dataloaded))))
batch_valid, label_valid = ((next(iter(valid_dataloaded))))

# Lancement de la classe "My_class" (plus haut)
modele_multicouche = My_class(len_input=3*32*32, len_output=10)

# Chargemnet de la fonction de loss (ici cross-entropy)
lossFn = F.cross_entropy

print("test sur une epoch :\n")
fit_one_cycle(modele_multicouche, train_dataloaded, valid_dataloaded, lossFn)
print("\n---------------\n\n")

print("test sur plusieurs epochs")
train_model(modele_multicouche, train_dataloaded, valid_dataloaded, lossFn, num_epochs=50)
print("\n---------------\n\n")



