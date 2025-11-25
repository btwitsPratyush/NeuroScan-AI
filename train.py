# train.py
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import torchvision.transforms as T
import time

class SimpleMRIDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        self.labels = []
        class_map = {
            'glioma': 0,
            'meningioma': 1,
            'pituitary': 2,
            'notumor': 3
        }
        self.transform = transform
        for cls, lab in class_map.items():
            path = os.path.join(root, cls)
            if os.path.exists(path):
                for f in os.listdir(path):
                    if f.lower().endswith(('.png','.jpg','.jpeg')):
                        self.paths.append(os.path.join(path, f))
                        self.labels.append(lab)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*28*28,128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,4)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def train():
    epochs = 30
    bs = 16
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.85,1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = SimpleMRIDataset("data/train", transform=train_tf)
    val_ds = SimpleMRIDataset("data/val", transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    # class weights
    if len(train_ds.labels) == 0:
        raise RuntimeError("No training images found. Check data/train/* folders.")
    unique, counts = np.unique(train_ds.labels, return_counts=True)
    total = sum(counts)
    class_weights = [total/(len(counts)*c) for c in counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights.tolist())

    model = SmallCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_dl.dataset)

        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss_v = loss_fn(out, yb)
                val_loss += loss_v.item() * xb.size(0)
                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(yb.cpu().tolist())

        val_loss = val_loss / len(val_dl.dataset) if len(val_dl.dataset) > 0 else 0
        val_acc = correct / total if total > 0 else 0
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  time={(time.time()-t0):.1f}s")

        if epoch % 5 == 0:
            print("Confusion matrix (val):")
            try:
                print(confusion_matrix(all_labels, all_preds))
            except Exception:
                pass

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pt")
            print("Saved models/best_model.pt with val_acc=", best_val_acc)

    print("Training complete. Best val acc:", best_val_acc)

if __name__ == "__main__":
    train()