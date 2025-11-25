import os
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import confusion_matrix

class MRIDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        self.labels = []
        class_map = {
            "glioma": 0,
            "meningioma": 1,
            "pituitary": 2,
            "notumor": 3
        }
        self.transform = transform

        for cls, lab in class_map.items():
            path = os.path.join(root, cls)
            if os.path.exists(path):
                for f in os.listdir(path):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.paths.append(os.path.join(path, f))
                        self.labels.append(lab)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def train():
    # SETTINGS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    epochs = 15
    batch_size = 16
    learning_rate = 1e-4

    # DATA TRANSFORMS
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.05, 0.05, 0.05, 0.02),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # LOAD DATASETS
    train_ds = MRIDataset("data/train", transform=train_tf)
    val_ds = MRIDataset("data/val", transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # CLASS WEIGHTS FOR IMBALANCE HANDLING
    if len(train_ds.labels) == 0:
        raise RuntimeError("No training images found in data/train/* folders.")

    unique, counts = np.unique(train_ds.labels, return_counts=True)
    total = sum(counts)
    class_weights = [total / (len(counts) * c) for c in counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights.tolist())

    # LOAD RESNET18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

    # FREEZE EARLY LAYERS TO PREVENT OVERFIT
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # OPTIMIZER & LOSS
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_acc = 0.0
    os.makedirs("models", exist_ok=True)

    # TRAINING LOOP
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_dl.dataset)

        # VALIDATION
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss_v = loss_fn(out, yb)
                val_loss += loss_v.item() * xb.size(0)

                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

                y_true.extend(yb.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        val_loss = val_loss / len(val_dl.dataset)
        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{epochs} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | Time={time.time()-t0:.1f}s")

        if epoch % 2 == 0:
            print("Confusion Matrix:")
            print(confusion_matrix(y_true, y_pred))

        # SAVE BEST MODEL
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/model_resnet_best.pt")
            print("Saved BEST MODEL with val_acc=", best_acc)

    print("Training Finished! Best Validation Accuracy =", best_acc)


if __name__ == "__main__":
    train()