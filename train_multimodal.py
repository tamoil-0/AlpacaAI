import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import autocast, GradScaler

# ======================================================
# CONFIG
# ======================================================
TRAIN_CSV = "dataset_train_split.csv"
TEST_CSV  = "dataset_test_split.csv"
DATASET_ROOT = "dataset"

FEATURES = ["ph", "L", "a", "b", "acidez"]   # <-- ajusta si tu csv usa otros nombres
LABEL_COL = "fase"
IMG_COL = "imagen"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR_IMG = 1e-4     # para la cabeza + último bloque (si se descongela)
LR_TAB = 1e-3     # para MLP tabular
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "runs_multimodal"
os.makedirs(OUT_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True

# ======================================================
# DATASET MULTIMODAL
# ======================================================
class MeatMultiModalDataset(Dataset):
    def __init__(self, df, img_transform, feature_cols):
        self.df = df.reset_index(drop=True)
        self.img_transform = img_transform
        self.feature_cols = feature_cols

        # asegurar float32 para tabular
        self.X_tab = self.df[self.feature_cols].astype(np.float32).values
        self.y = self.df["label"].astype(np.int64).values
        self.img_paths = self.df[IMG_COL].astype(str).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # imagen
        img_path = os.path.join(DATASET_ROOT, self.img_paths[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        # tabular
        x_tab = torch.tensor(self.X_tab[idx], dtype=torch.float32)

        # label
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return img, x_tab, y

# ======================================================
# MODELO MULTIMODAL
# ======================================================
class MultiModalNet(nn.Module):
    def __init__(self, num_classes, tab_in_dim):
        super().__init__()

        # ---- CNN backbone (ResNet18)
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # congelar todo el backbone al inicio (evita overfit)
        for p in self.cnn.parameters():
            p.requires_grad = False

        # quitamos la fc y dejamos extractor
        cnn_feat_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # proyección de features de imagen
        self.img_head = nn.Sequential(
            nn.Linear(cnn_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ---- MLP tabular
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ---- FUSIÓN
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, x_tab):
        img_feat = self.cnn(img)              # (B, cnn_feat_dim)
        img_feat = self.img_head(img_feat)    # (B, 256)
        tab_feat = self.tab_mlp(x_tab)        # (B, 64)
        fused = torch.cat([img_feat, tab_feat], dim=1)  # (B, 320)
        out = self.fusion(fused)
        return out

# ======================================================
# HELPERS
# ======================================================
def get_class_weights(y_train, device):
    classes = np.unique(y_train)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return torch.tensor(w, dtype=torch.float32).to(device)

def evaluate(model, loader, le):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, x_tab, y in loader:
            img = img.to(DEVICE, non_blocking=True)
            x_tab = x_tab.to(DEVICE, non_blocking=True)

            logits = model(img, x_tab)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(y.numpy())
            y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return cm, rep, acc, prec, rec, f1

# ======================================================
# MAIN
# ======================================================
def main():
    print("CUDA disponible:", torch.cuda.is_available())
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------
    # Cargar split oficial
    # --------------------------
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    # Validaciones rápidas
    missing = [c for c in [IMG_COL, LABEL_COL] + FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f"❌ Faltan columnas en TRAIN: {missing}")
    missing2 = [c for c in [IMG_COL, LABEL_COL] + FEATURES if c not in test_df.columns]
    if missing2:
        raise ValueError(f"❌ Faltan columnas en TEST: {missing2}")

    # Labels
    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df[LABEL_COL])
    test_df["label"]  = le.transform(test_df[LABEL_COL])

    print("\n📊 Distribución TRAIN:")
    print(train_df[LABEL_COL].value_counts())
    print("\n📊 Distribución TEST:")
    print(test_df[LABEL_COL].value_counts())

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
    print("Clases:", dict(zip(le.classes_, range(len(le.classes_)))))

    # --------------------------
    # Transforms
    # --------------------------
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # --------------------------
    # DataLoaders
    # --------------------------
    train_loader = DataLoader(
        MeatMultiModalDataset(train_df, train_tf, FEATURES),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        MeatMultiModalDataset(test_df, test_tf, FEATURES),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --------------------------
    # Modelo
    # --------------------------
    model = MultiModalNet(num_classes=len(le.classes_), tab_in_dim=len(FEATURES)).to(DEVICE)

    # class weights
    class_w = get_class_weights(train_df["label"].values, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # Optimizador: solo cabezas (cnn congelada)
    params = [
        {"params": model.img_head.parameters(), "lr": LR_IMG},
        {"params": model.tab_mlp.parameters(), "lr": LR_TAB},
        {"params": model.fusion.parameters(), "lr": LR_IMG},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    scaler = GradScaler()

    best_loss = float("inf")

    # --------------------------
    # TRAIN
    # --------------------------
    print("\n🚀 Entrenando Multimodal (Imagen + Tabular)...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for img, x_tab, y in train_loader:
            img = img.to(DEVICE, non_blocking=True)
            x_tab = x_tab.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                logits = model(img, x_tab)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        scheduler.step(epoch_loss)

        # Guardar mejor (por loss train)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "model": model.state_dict(),
                "classes": le.classes_,
                "features": FEATURES
            }, os.path.join(OUT_DIR, "best_multimodal.pth"))

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}")

    # --------------------------
    # EVALUACIÓN FINAL
    # --------------------------
    cm, rep, acc, prec, rec, f1 = evaluate(model, test_loader, le)

    print("\n📊 CONFUSION MATRIX")
    print(cm)

    print("\n📋 CLASSIFICATION REPORT")
    print(rep)

    print("\n📈 MÉTRICAS GLOBALES")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)

    print("\n✅ PASO 5 COMPLETADO — MULTIMODAL (Imagen + Tabular)")

if __name__ == "__main__":
    main()
