import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import random
import pandas as pd
from PIL import Image
import torch.optim as optim
import kagglehub
 import matplotlib.pyplot as plt

path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")

data_dir ="/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset"
# Define paths to COVID and Normal image folders
covid_images_path = os.path.join(data_dir, "COVID", "images")
normal_images_path = os.path.join(data_dir, "Normal", "images")

# Get list of images from both classes
covid_images = [os.path.join(covid_images_path, img) for img in os.listdir(covid_images_path) if img.endswith(".png")]
normal_images = [os.path.join(normal_images_path, img) for img in os.listdir(normal_images_path) if img.endswith(".png")]

# Randomly shuffle images
random.seed(123)
random.shuffle(covid_images)
random.shuffle(normal_images)

normal_train_downsampled = random.sample(normal_images, 3600)

covid_train, covid_test = covid_images[:2880], covid_images[2880:]
normal_train, normal_test = normal_train_downsampled[:2880], normal_train_downsampled[2880:]

# Create DataFrames with labels (COVID = 1, Normal = 0)
covid_train_df = pd.DataFrame({"filepaths": covid_train, "labels": 1})
normal_train_df = pd.DataFrame({"filepaths": normal_train, "labels": 0})

covid_test_df = pd.DataFrame({"filepaths": covid_test, "labels": 1})
normal_test_df = pd.DataFrame({"filepaths": normal_test, "labels": 0})

# Combine train and test datasets
train_df = pd.concat([covid_train_df, normal_train_df], ignore_index=True)
test_df = pd.concat([covid_test_df, normal_test_df], ignore_index=True)

# Shuffle training data
train_df = train_df.sample(frac=1, random_state=123).reset_index(drop=True)

# Split training data into 5 clients 
clients = []
for i in range(5):
    client_df = train_df.iloc[i * 576:(i + 1) * 576].reset_index(drop=True)
    clients.append(client_df)

# Custom Dataset Class
class CovidDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepaths']
        label = self.df.iloc[idx]['labels']
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create datasets and dataloaders
train_datasets = [CovidDataset(df, transform=transform) for df in clients]
test_dataset = CovidDataset(test_df, transform=transform)

batch_size = 16
train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in train_datasets]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
client_train_accuracies = [[] for _ in range(len(train_loaders))]
client_test_accuracies = [[] for _ in range(len(train_loaders))]
global_test_accuracies = []
# Define CNN Model
from torchvision.models import resnet50

class ResNet50Gray(nn.Module):
    def __init__(self):
        super(ResNet50Gray, self).__init__()
        self.model = resnet50(pretrained=True)
        # Change input channel from 3 to 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final fully connected layer for 2 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)


# Training Local Model
def train_local_model(model, train_loader, device):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    for epoch in range(7):  
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# Accuracy calculate on Given Dataset
def evaluate_model(model, data_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def average_models_from_state_dicts(global_model, state_dicts):
    global_dict = global_model.state_dict()
    avg_dict = {}

    for key in global_dict.keys():
        stacked_tensors = torch.stack([
            sd[key].float() if sd[key].dtype in [torch.float32, torch.float64]
            else sd[key].to(torch.float32)
            for sd in state_dicts
        ])
        avg = torch.mean(stacked_tensors, dim=0)

        if global_dict[key].dtype in [torch.long, torch.int64]:
            avg = avg.to(torch.long)

        avg_dict[key] = avg

    global_model.load_state_dict(avg_dict)
    return global_model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_rounds = 15
global_model = ResNet50Gray().to(device)

client_model_history_paths = []
global_model_history_paths = []
os.makedirs("client_checkpoints", exist_ok=True)
os.makedirs("global_checkpoints", exist_ok=True)


for round in range(num_rounds):
    print(f"\nRound {round+1}")
    client_model_paths = []

    # Save global model at the start of the round
    global_model_path = f"global_checkpoints/global_model_round_{round}.pth"
    torch.save(global_model.state_dict(), global_model_path)
    global_model_history_paths.append(global_model_path)

    for client_id, train_loader in enumerate(train_loaders):
        local_model = ResNet50Gray().to(device)
        local_model.load_state_dict(global_model.state_dict())
        local_model = train_local_model(local_model, train_loader, device)

        # Save client model
        client_model_path = f"client_checkpoints/round_{round+1}_client_{client_id}.pt"
        torch.save(local_model.state_dict(), client_model_path)
        client_model_paths.append(client_model_path)
        print(f"Round : {round+1} , Client Trained : {client_id}")

    client_model_history_paths.append(client_model_paths)

    # Federated Averaging
    client_models = [torch.load(path, map_location=device, weights_only=True) for path in client_model_paths]
    global_model = average_models_from_state_dicts(global_model, client_models)

## Saving Global State into Disk
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "global_model_state_dict": global_model.state_dict(),
    "client_model_history_paths": client_model_history_paths,
    "global_model_history_paths": global_model_history_paths,
    "num_rounds": num_rounds,
    "num_clients": len(train_loaders),
}, "checkpoints/fed_checkpoint.pth")

# FedEraser
class FedEraser:
    def __init__(self, learning_rate, global_model, checkpoint, device):
        self.learning_rate = learning_rate
        self.global_model = global_model
        self.device = device
        self.checkpoint = checkpoint

    def apply_gradient_ascent_from_round(self, client_id, start_round):
        global_dict = self.global_model.state_dict()
    
        for round_idx in reversed(range(start_round + 1)):
            # Load previous global model
            prev_global_state = torch.load(self.checkpoint["global_model_history_paths"][round_idx], map_location=self.device)
    
            # Load client model for that round
            client_path = self.checkpoint["client_model_history_paths"][round_idx][client_id]
            client_state = torch.load(client_path, map_location=self.device)
    
            # Compute the update
            update = {
                key: client_state[key] - prev_global_state[key]
                for key in global_dict.keys()
                if global_dict[key].dtype.is_floating_point
            }
    
            # Apply gradient ascent
            for key in update:
                global_dict[key] -= self.learning_rate * update[key]
    
        self.global_model.load_state_dict(global_dict)
        print(f"Client {client_id+1} successfully unlearned from round {start_round} to 0.")

## Main function
global_accs = []
unlearned_accs = []
checkpoint = torch.load("checkpoints/fed_checkpoint.pth", map_location=device, weights_only=True)

for round_idx in range(num_rounds):
    # Load global model at round r
    global_model = ResNet50Gray().to(device)
    global_model.load_state_dict(torch.load(global_model_history_paths[round_idx], map_location=device, weights_only=True))

    # Accuracy before unlearning
    acc = evaluate_model(global_model, test_loader, device)
    global_accs.append(acc)

    # Clone model for unlearning
    unlearn_model = ResNet50Gray().to(device)
    unlearn_model.load_state_dict(torch.load(checkpoint["global_model_history_paths"][round_idx], weights_only=True))

    fed_eraser = FedEraser(learning_rate=0.01, global_model=unlearn_model, checkpoint=checkpoint, device=device)
    fed_eraser.apply_gradient_ascent_from_round(client_id=2, start_round=round_idx)

    # Accuracy after unlearning
    acc_unlearned = evaluate_model(unlearn_model, test_loader, device)
    unlearned_accs.append(acc_unlearned)

## Plots
rounds = list(range(num_rounds))  # [0, 1, 2, ..., num_rounds-1]
plt.plot(rounds, global_accs, label="Global Accuracy", color='blue')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Global Accuracy Over Rounds")
plt.legend()
plt.grid()
plt.show()

rounds = list(range(num_rounds))  # e.g., [0, 1, 2, ..., 19]
plt.plot(rounds, global_accs, label="Global Accuracy")
plt.plot(rounds, unlearned_accs, label="Unlearned Accuracy")

plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Global vs Unlearned Accuracy over Rounds")
plt.xticks(rounds)  # Ensures x-axis has whole number ticks
plt.legend()

plt.show()
