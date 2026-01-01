# app_multimodal_ux_pro.py
# ============================================================
# ✅ CLASIFICADOR DE FRESCURA DE CARNE (MULTIMODAL)
# Imagen (RGB+LAB) + (temp,dia,pH,L,a,b) opcionales
# - Segmentación + recorte robusto (tipo tu PASO 1)
# - Predicción + Probabilidades
# - Grad-CAM real (zona decisiva)
# - LAB (L*, a*, b*) + estadísticas + histogramas
# - Índices colorimétricos (WI, RI) + comparación vs promedios por clase (del dataset_full.csv)
# - UI/UX tipo “dashboard” (cards), responsive, pastel, scroll
#
# Requisitos:
#   pip install torch torchvision opencv-python pillow matplotlib pandas numpy
#
# Archivos:
#   runs_multimodal/best_multimodal.pth
#   runs_multimodal/meta.json
#   dataset_full.csv
#   dataset/...(rutas relativas del CSV)
# ============================================================

import os
import json
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Fix común Windows (OpenMP duplicado) ---
# Si te crashea con: libomp.dll vs libiomp5md.dll
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from PIL import Image, ImageTk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "runs_multimodal/best_multimodal.pth"
META_PATH  = "runs_multimodal/meta.json"
CSV_PATH   = "dataset_full.csv"
DATASET_ROOT = "dataset"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["FRESCO", "SEMIFRESCO", "NO_APTO"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

TAB_COLS = ["temp", "dia", "pH", "L", "a", "b"]

# Segmentación/recorte robusto (tu PASO 1)
MIN_AREA_RATIO = 0.01
PAD_PX = 12

# UI sizes
IMG_SIZE_MODEL = 320  # (del entrenamiento)
CARD_PAD = 12

# ============================================================
# UTILIDADES DE IMAGEN
# ============================================================

def norm_path(rel):
    return os.path.join(DATASET_ROOT, str(rel).replace("\\", "/"))

def bgr_read(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def resize_keep_aspect_center(img_rgb, size=320, bg=255):
    h, w = img_rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_r = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), bg, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = img_r
    return canvas

def clean_mask(mask_u8):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask_u8

def largest_component(mask_u8):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8) * 255

def get_bbox(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def overlay_mask_bgr(bgr, mask_u8, alpha=0.45):
    overlay = bgr.copy()
    color = np.zeros_like(bgr)
    color[:, :, 2] = 255  # rojo
    m = (mask_u8 > 0)
    overlay[m] = cv2.addWeighted(bgr[m], 1 - alpha, color[m], alpha, 0)
    return overlay

def segment_mask_and_crop(rgb):
    """
    Implementación estilo tu PASO 1:
    - Otsu en S, Otsu invertido en V
    - combine
    - morph
    - componente más grande
    - bbox + pad
    """
    bgr = rgb_to_bgr(rgb)
    h, w = bgr.shape[:2]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    _, s_th = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, v_th = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask0 = cv2.bitwise_and(s_th, cv2.bitwise_not(v_th))
    mask1 = clean_mask(mask0)
    mask2 = largest_component(mask1)

    if (mask2 > 0).sum() < int(MIN_AREA_RATIO * h * w):
        mask2 = largest_component(clean_mask(s_th))

    bbox = get_bbox(mask2)
    if bbox is None:
        crop = rgb.copy()
        mask_crop = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.uint8) * 255
        return crop, mask_crop, mask2, (0, 0, rgb.shape[1]-1, rgb.shape[0]-1)

    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - PAD_PX); y1 = max(0, y1 - PAD_PX)
    x2 = min(w - 1, x2 + PAD_PX); y2 = min(h - 1, y2 + PAD_PX)

    crop = rgb[y1:y2+1, x1:x2+1].copy()
    mask_crop = mask2[y1:y2+1, x1:x2+1].copy()
    return crop, mask_crop, mask2, (x1, y1, x2, y2)

def rgb_to_lab(rgb):
    # OpenCV LAB en BGR->LAB
    lab = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
    return lab

def lab_channels_images(lab_u8):
    # L,a,b en OpenCV: L [0..255], a,b [0..255] con 128 ~ 0
    L = lab_u8[:, :, 0]
    A = lab_u8[:, :, 1]
    B = lab_u8[:, :, 2]
    return L, A, B

def estimate_lab_stats_from_crop(crop_rgb):
    """
    Devuelve:
      L*, a*, b* aproximados (escala tipo CIELAB),
      y stats por canal (media, std, p10, p50, p90).
    Nota: conversión aproximada desde OpenCV:
      L* ~ Lcv * 100/255
      a* ~ (Acv - 128)
      b* ~ (Bcv - 128)
    """
    lab = rgb_to_lab(crop_rgb)
    Lcv, Acv, Bcv = lab_channels_images(lab)

    L_star = Lcv.astype(np.float32) * (100.0 / 255.0)
    a_star = Acv.astype(np.float32) - 128.0
    b_star = Bcv.astype(np.float32) - 128.0

    def stats(x):
        x = x.reshape(-1)
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    return {
        "L_star_mean": float(np.mean(L_star)),
        "a_star_mean": float(np.mean(a_star)),
        "b_star_mean": float(np.mean(b_star)),
        "L_stats": stats(L_star),
        "a_stats": stats(a_star),
        "b_stats": stats(b_star),
        "lab_u8": lab,
        "L_star": L_star,
        "a_star": a_star,
        "b_star": b_star,
    }

def color_indices(L_star, a_star, b_star):
    # Índices derivados (con medias)
    # WI (Whiteness Index) y RI (Redness Index)
    WI = 100.0 - math.sqrt((100.0 - L_star)**2 + (a_star**2) + (b_star**2))
    RI = a_star / (math.sqrt(L_star**2 + b_star**2) + 1e-6)
    return float(WI), float(RI)

def to_6ch_tensor(rgb_crop, size=IMG_SIZE_MODEL):
    rgb_s = resize_keep_aspect_center(rgb_crop, size=size)
    lab = rgb_to_lab(rgb_s)

    x_rgb = rgb_s.astype(np.float32) / 255.0
    x_lab = lab.astype(np.float32) / 255.0
    x = np.concatenate([x_rgb, x_lab], axis=2)  # H,W,6
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).float(), rgb_s, lab


# ============================================================
# MODELO MULTIMODAL + GRADCAM
# ============================================================

class MultiModalNet(nn.Module):
    def __init__(self, tab_dim=6, num_classes=3):
        super().__init__()
        self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        old = self.cnn.features[0][0]
        new = nn.Conv2d(6, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        self.cnn.features[0][0] = new

        self.feat_dim = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()

        self.tab = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_tab):
        feat = self.cnn(x_img)          # (B,feat_dim)
        tabf = self.tab(x_tab)          # (B,64)
        fused = torch.cat([feat, tabf], dim=1)
        return self.head(fused)

class GradCAM:
    """
    Grad-CAM sobre la rama CNN.
    Hook en self.model.cnn.features[-1] (último bloque conv).
    """
    def __init__(self, model: MultiModalNet, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        try:
            self.h1.remove()
            self.h2.remove()
        except:
            pass

    def __call__(self, x_img, x_tab, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x_img, x_tab)  # (1,C)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        # activations: (1, C, h, w)
        acts = self.activations
        grads = self.gradients

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=False)  # (1,h,w)
        cam = torch.relu(cam)

        cam_np = cam[0].cpu().numpy()
        cam_np = cam_np - cam_np.min()
        cam_np = cam_np / (cam_np.max() + 1e-8)
        return cam_np, logits.detach()


# ============================================================
# DATASET STATS (para comparaciones y defaults)
# ============================================================

def load_dataset_stats():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No encuentro {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    for c in ["imagen", "temp", "dia", "pH", "L", "a", "b", "fase"]:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c} en {CSV_PATH}")

    df["fase"] = df["fase"].astype(str).str.upper().str.strip()
    df = df[df["fase"].isin(CLASS_NAMES)].reset_index(drop=True)

    # global means (fallback)
    global_mean = df[TAB_COLS].mean().to_dict()

    # class means
    class_means = {}
    for cls in CLASS_NAMES:
        sub = df[df["fase"] == cls]
        class_means[cls] = sub[TAB_COLS].mean().to_dict()

    return df, global_mean, class_means

def plot_colorbar(cmap_name, label):
    fig = plt.figure(figsize=(3.2, 0.5), dpi=120)
    ax = fig.add_axes([0.05, 0.4, 0.9, 0.3])

    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=0, vmax=1)

    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal'
    )
    cb.set_label(label, fontsize=9)

    fig.patch.set_facecolor("white")
    return fig_to_pil(fig)

# ============================================================
# PLOTS -> PIL images
# ============================================================

def fig_to_pil(fig):
    fig.canvas.draw()

    # Matplotlib >= 3.8 (forma correcta)
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3]  # quitamos canal alpha

    plt.close(fig)
    return Image.fromarray(img)

def plot_prob_bars(probs, pred_name):
    fig = plt.figure(figsize=(5.2, 3.2), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title(f"Probabilidades por clase (pred: {pred_name})", fontsize=10)
    ax.bar(CLASS_NAMES, probs)
    ax.set_ylim(0, 1.0)
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p*100:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Prob.")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig_to_pil(fig)

def plot_hist_three(L_star, a_star, b_star):
    fig = plt.figure(figsize=(6.2, 3.4), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title("Histogramas (ROI) — L*, a*, b*", fontsize=10)

    ax.hist(L_star.reshape(-1), bins=40, alpha=0.55, label="L*")
    ax.hist(a_star.reshape(-1), bins=40, alpha=0.55, label="a*")
    ax.hist(b_star.reshape(-1), bins=40, alpha=0.55, label="b*")

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig_to_pil(fig)

def plot_compare_to_class_means(values_used, class_means):
    """
    values_used: dict {temp,dia,pH,L,a,b}
    class_means: dict[class]->dict cols
    Mostramos barras agrupadas para pH, L, a, b (lo más explicable).
    """
    keys = ["pH", "L", "a", "b"]
    x = np.arange(len(keys))

    fig = plt.figure(figsize=(6.2, 3.6), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title("Comparación vs promedios del dataset (por clase)", fontsize=10)

    width = 0.22
    ax.bar(x - width, [class_means["FRESCO"][k] for k in keys], width, label="FRESCO")
    ax.bar(x,         [class_means["SEMIFRESCO"][k] for k in keys], width, label="SEMIFRESCO")
    ax.bar(x + width, [class_means["NO_APTO"][k] for k in keys], width, label="NO_APTO")

    # valor actual
    ax.plot(x, [values_used[k] for k in keys], marker="o", linewidth=2.0, label="MUESTRA")

    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9, ncols=2)
    fig.tight_layout()
    return fig_to_pil(fig)


# ============================================================
# PIPELINE INFERENCIA
# ============================================================

def safe_float(s):
    try:
        if s is None:
            return None
        s = str(s).strip()
        if s == "":
            return None
        return float(s)
    except:
        return None

def build_model_and_meta():
    # meta (tab mean/std)
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"No encuentro {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tab_mean = np.array(meta["tab_mean"], dtype=np.float32)
    tab_std  = np.array(meta["tab_std"], dtype=np.float32)

    model = MultiModalNet(tab_dim=6, num_classes=3).to(DEVICE)
    model.eval()

    # weights_only evita warning y es más seguro
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd, strict=True)

    # gradcam target layer
    target_layer = model.cnn.features[-1]
    cam = GradCAM(model, target_layer)
    return model, cam, tab_mean, tab_std, meta

def standardize_tab(tab, mean, std):
    tab = (tab - mean) / (std + 1e-6)
    return tab
def overlay_lab_gradcam(lab_channel, cam_norm):
    cam = cv2.resize(cam_norm, lab_channel.shape[::-1])
    cam = np.uint8(255 * cam)

    lab_norm = cv2.normalize(lab_channel, None, 0, 255, cv2.NORM_MINMAX)
    lab_rgb = cv2.cvtColor(lab_norm, cv2.COLOR_GRAY2RGB)

    heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    out = cv2.addWeighted(lab_rgb, 0.6, heat, 0.4, 0)
    return out

def plot_lab_radar(values, fresco_means):
    labels = ["L*", "a*", "b*"]
    sample = [values["L"], values["a"], values["b"]]
    fresh  = [fresco_means["L"], fresco_means["a"], fresco_means["b"]]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    sample += sample[:1]
    fresh  += fresh[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(4,4), dpi=120)
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, sample, label="Muestra", linewidth=2)
    ax.fill(angles, sample, alpha=0.25)

    ax.plot(angles, fresh, label="FRESCO (prom.)", linestyle="--")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    fig.tight_layout()
    return fig_to_pil(fig)

def make_heatmap_overlay(rgb_base, cam_norm, alpha=0.45):
    """
    rgb_base: (H,W,3) uint8
    cam_norm: (h,w) float 0..1 -> resize to base
    """
    H, W = rgb_base.shape[:2]
    cam_r = cv2.resize(cam_norm, (W, H), interpolation=cv2.INTER_CUBIC)
    cam_u8 = np.uint8(255 * cam_r)
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)   # BGR
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (rgb_base.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return out, cam_r

def build_explanation(values_used, probs, pred_idx, lab_stats, WI, RI, class_means):
    pred_name = IDX_TO_CLASS[pred_idx]
    p = probs[pred_idx] * 100.0

    # Heurísticas interpretativas (no reemplaza el modelo; guía humana)
    ph = values_used["pH"]
    Lm = values_used["L"]
    am = values_used["a"]
    bm = values_used["b"]
    delta_L = values_used["L"] - class_means["FRESCO"]["L"]
    delta_a = values_used["a"] - class_means["FRESCO"]["a"]
    delta_b = values_used["b"] - class_means["FRESCO"]["b"]

    # Comparación simple contra medias de clase (más cercano)
    dists = {}
    for cls in CLASS_NAMES:
        cm = class_means[cls]
        vec_c = np.array([cm["pH"], cm["L"], cm["a"], cm["b"]], dtype=np.float32)
        vec_x = np.array([ph, Lm, am, bm], dtype=np.float32)
        dists[cls] = float(np.linalg.norm(vec_x - vec_c))
    closest = min(dists, key=dists.get)

    lines = []
    lines.append(f"Predicción: {pred_name} ({p:.1f}%).")
    lines.append("")
    lines.append("Evidencia cuantitativa (muestra):")
    lines.append(f"• pH={ph:.2f} | L*={Lm:.2f} | a*={am:.2f} | b*={bm:.2f}")
    lines.append(f"• WI (blancura)={WI:.2f} | RI (enrojecimiento)={RI:.3f}")
    lines.append("")
    lines.append("Guía interpretativa (colorimetría):")
    lines.append("• a* alto → más rojo; a* bajo → menos rojo (oxidación).")
    lines.append("• b* alto → más amarillento (cambios por deterioro).")
    lines.append("• L* bajo → más oscuro (frecuente en degradación).")
    lines.append("")
    lines.append("Comparación con el dataset (medias por clase):")
    lines.append(f"• Perfil más cercano (pH/L/a/b): {closest} (dist={dists[closest]:.2f})")
    lines.append("")
    lines.append("Interpretación IA (Grad-CAM):")
    lines.append("• Zonas cálidas = regiones que más influyeron en la decisión.")
    lines.append("• Si el calor cae en manchas/textura anormal → evidencia visual del modelo.")
    return "\n".join(lines)

def infer_multimodal(
    model, cam,
    image_path,
    tab_inputs_optional,    # dict con valores o None
    global_mean, class_means,
    tab_mean, tab_std
):
    # --- load original ---
    bgr = bgr_read(image_path)
    rgb = bgr_to_rgb(bgr)

    # --- segment + crop robust ---
    crop_rgb, crop_mask_u8, full_mask_u8, bbox = segment_mask_and_crop(rgb)

    # overlay analyzed zone on original (mask)
    overlay_full = bgr_to_rgb(overlay_mask_bgr(bgr, full_mask_u8, alpha=0.38))

    # --- 6ch tensor ---
    x_img_6ch, crop_rgb_320, lab_320 = to_6ch_tensor(crop_rgb, size=IMG_SIZE_MODEL)

    # --- estimate LAB stats from crop (real ROI, no 320) ---
    lab_stats = estimate_lab_stats_from_crop(crop_rgb)
    Lm = lab_stats["L_star_mean"]
    am = lab_stats["a_star_mean"]
    bm = lab_stats["b_star_mean"]
    WI, RI = color_indices(Lm, am, bm)

    # --- build final tab values (optional fields) ---
    used = {}
    used_src = {}

    # temp/dia/pH default global mean (si no hay)
    for k in ["temp", "dia", "pH"]:
        if tab_inputs_optional.get(k) is None:
            used[k] = float(global_mean[k])
            used_src[k] = "media dataset"
        else:
            used[k] = float(tab_inputs_optional[k])
            used_src[k] = "usuario"

    # L/a/b: si usuario no da -> estimados desde imagen ROI
    # OJO: tu dataset_full.csv parece usar L,a,b ya en escala tipo CIELAB (no 0..255).
    # Nosotros estimamos CIELAB aproximado desde OpenCV.
    for k, v in [("L", Lm), ("a", am), ("b", bm)]:
        if tab_inputs_optional.get(k) is None:
            used[k] = float(v)
            used_src[k] = "estimado imagen"
        else:
            used[k] = float(tab_inputs_optional[k])
            used_src[k] = "usuario"

    tab = np.array([used[c] for c in TAB_COLS], dtype=np.float32)
    tab_stdzd = standardize_tab(tab, tab_mean, tab_std)

    x_img = x_img_6ch.unsqueeze(0).to(DEVICE)
    x_tab = torch.from_numpy(tab_stdzd).unsqueeze(0).to(DEVICE)

    # --- Grad-CAM + logits ---
    with torch.set_grad_enabled(True):
        cam_map, logits = cam(x_img, x_tab, class_idx=int(torch.argmax(model(x_img, x_tab), dim=1).item()))

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_name = IDX_TO_CLASS[pred_idx]

    # GradCAM overlay on crop (320)
    crop_show = crop_rgb_320.copy()
    grad_overlay, cam_resized = make_heatmap_overlay(crop_show, cam_map, alpha=0.42)

    # LAB channel images for display (from 320)
    # Show L*, a*, b* as grayscale derived from OpenCV channels
    Lcv, Acv, Bcv = lab_channels_images(lab_320)
    lab_L_img = cv2.cvtColor(Lcv, cv2.COLOR_GRAY2RGB)
    lab_A_img = cv2.cvtColor(Acv, cv2.COLOR_GRAY2RGB)
    lab_B_img = cv2.cvtColor(Bcv, cv2.COLOR_GRAY2RGB)

    # --- plots ---
    prob_plot = plot_prob_bars(probs, pred_name)
    hist_plot = plot_hist_three(lab_stats["L_star"], lab_stats["a_star"], lab_stats["b_star"])
    comp_plot = plot_compare_to_class_means(used, class_means)

    # explanation text
    explain = build_explanation(used, probs, pred_idx, lab_stats, WI, RI, class_means)

    return {
        "pred_idx": pred_idx,
        "pred_name": pred_name,
        "probs": probs,
        "used": used,
        "used_src": used_src,
        "WI": WI,
        "RI": RI,
        "bbox": bbox,

        "img_original": rgb,
        "img_overlay_mask": overlay_full,
        "img_crop": crop_show,
        "img_gradcam": grad_overlay,

        "img_lab_L": lab_L_img,
        "img_lab_A": lab_A_img,
        "img_lab_B": lab_B_img,

        "plot_probs": prob_plot,
        "plot_hist": hist_plot,
        "plot_compare": comp_plot,

        "lab_stats": lab_stats,
        "explain_text": explain,
    }


# ============================================================
# UI/UX: Dashboard con Cards + Scroll
# ============================================================

PASTEL_BG = "#F6F4FF"
CARD_BG   = "#FFFFFF"
CARD_EDGE = "#E6E2F5"
TXT_MAIN  = "#1F2430"
TXT_SUB   = "#5C6270"
ACCENT    = "#6C5CE7"
ACCENT_2  = "#00B894"
WARN      = "#E17055"

def tkimg_from_np(rgb, max_w=560, max_h=340):
    h, w = rgb.shape[:2]
    scale = min(max_w / max(w,1), max_h / max(h,1), 1.0)
    nw, nh = int(w*scale), int(h*scale)
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(img)
    return ImageTk.PhotoImage(pil)

def tkimg_from_pil(pil_img, max_w=560, max_h=340):
    w, h = pil_img.size
    scale = min(max_w / max(w,1), max_h / max(h,1), 1.0)
    nw, nh = int(w*scale), int(h*scale)
    pil2 = pil_img.resize((nw, nh), Image.LANCZOS)
    return ImageTk.PhotoImage(pil2)

class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=PASTEL_BG)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.inner_id, width=event.width)

    def _on_mousewheel(self, event):
        # windows
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

class Card(ttk.Frame):
    def __init__(self, parent, title, subtitle=None):
        super().__init__(parent)
        self.configure(style="Card.TFrame")
        self.title = ttk.Label(self, text=title, style="CardTitle.TLabel")
        self.title.pack(anchor="w", padx=12, pady=(10, 2))
        if subtitle:
            self.sub = ttk.Label(self, text=subtitle, style="CardSub.TLabel")
            self.sub.pack(anchor="w", padx=12, pady=(0, 8))

        self.body = ttk.Frame(self, style="CardBody.TFrame")
        self.body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificador de Frescura de Carne — Multimodal (Imagen + pH + LAB) + Grad-CAM")
        self.state("zoomed")
        self.configure(bg=PASTEL_BG)

        self._style()

        # Load model + stats
        try:
            self.df, self.global_mean, self.class_means = load_dataset_stats()
            self.model, self.cam, self.tab_mean, self.tab_std, self.meta = build_model_and_meta()
        except Exception as e:
            messagebox.showerror("Error", f"No pude inicializar el sistema:\n\n{e}")
            raise

        self.current_path = None
        self.last_result = None
        self._photo_refs = []  # keep references

        # Layout
        self._build_header()
        self._build_main()

    def _style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", font=("Segoe UI", 10), background=PASTEL_BG, foreground=TXT_MAIN)
        style.configure("Header.TFrame", background=PASTEL_BG)
        style.configure("HeaderTitle.TLabel", background=PASTEL_BG, foreground=TXT_MAIN, font=("Segoe UI Semibold", 16))
        style.configure("HeaderSub.TLabel", background=PASTEL_BG, foreground=TXT_SUB, font=("Segoe UI", 10))

        style.configure("Card.TFrame", background=CARD_BG, relief="solid", borderwidth=1)
        style.map("Card.TFrame", background=[("active", CARD_BG)])
        style.configure("CardTitle.TLabel", background=CARD_BG, foreground=TXT_MAIN, font=("Segoe UI Semibold", 12))
        style.configure("CardSub.TLabel", background=CARD_BG, foreground=TXT_SUB, font=("Segoe UI", 9))
        style.configure("CardBody.TFrame", background=CARD_BG)

        style.configure("Accent.TButton", background=ACCENT, foreground="white", padding=(12, 10))
        style.map("Accent.TButton", background=[("active", "#5A4DE0")])

        style.configure("Soft.TButton", background="#ECE9FF", foreground=TXT_MAIN, padding=(12, 10))
        style.map("Soft.TButton", background=[("active", "#E3DFFF")])

        style.configure("Danger.TLabel", foreground="#D63031", background=CARD_BG, font=("Segoe UI Semibold", 12))
        style.configure("OK.TLabel", foreground="#00B894", background=CARD_BG, font=("Segoe UI Semibold", 12))
        style.configure("Warn.TLabel", foreground="#E17055", background=CARD_BG, font=("Segoe UI Semibold", 12))

        style.configure("Mini.TLabel", foreground=TXT_SUB, background=CARD_BG, font=("Segoe UI", 9))
        style.configure("Mono.TLabel", foreground=TXT_MAIN, background=CARD_BG, font=("Consolas", 9))

        style.configure("Field.TEntry", padding=(8, 6))

    def _build_header(self):
        hdr = ttk.Frame(self, style="Header.TFrame")
        hdr.pack(fill="x", padx=18, pady=(14, 8))

        title = ttk.Label(hdr, text="Clasificador de Frescura de Carne — Multimodal (Imagen + pH + LAB) + Grad-CAM",
                          style="HeaderTitle.TLabel")
        title.pack(anchor="w")

        sub = ttk.Label(
            hdr,
            text="Imagen obligatoria. Datos opcionales: temp, día, pH, L*, a*, b*. "
                 "Si no ingresas, se estiman (LAB desde imagen) o se usan medias del dataset.",
            style="HeaderSub.TLabel"
        )
        sub.pack(anchor="w", pady=(2, 0))

    def _build_main(self):
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # Left panel (inputs)
        self.left = ttk.Frame(container)
        self.left.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        self.left.configure(style="Card.TFrame")

        # Right panel (scroll dashboard)
        self.right = ScrollFrame(container)
        self.right.grid(row=0, column=1, sticky="nsew")

        self._build_left_panel()
        self._build_dashboard_skeleton()

    def _build_left_panel(self):
        wrap = ttk.Frame(self.left, style="CardBody.TFrame")
        wrap.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Label(wrap, text="Acciones", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))

        ttk.Button(wrap, text="📂 Cargar imagen", style="Accent.TButton", command=self.pick_image).pack(fill="x", pady=(0, 10))

        ttk.Label(wrap, text="Datos opcionales", style="CardTitle.TLabel").pack(anchor="w", pady=(10, 6))

        self.var_temp = tk.StringVar()
        self.var_dia  = tk.StringVar()
        self.var_ph   = tk.StringVar()
        self.var_L    = tk.StringVar()
        self.var_a    = tk.StringVar()
        self.var_b    = tk.StringVar()

        self._field(wrap, "temp (°C)", self.var_temp)
        self._field(wrap, "día",       self.var_dia)
        self._field(wrap, "pH",        self.var_ph)
        self._field(wrap, "L*",        self.var_L)
        self._field(wrap, "a*",        self.var_a)
        self._field(wrap, "b*",        self.var_b)

        ttk.Label(
            wrap,
            text="Tip:\n"
                 "• Si no pones L*,a*,b*, la app los estima desde la imagen recortada.\n"
                 "• Si no pones temp/día/pH, usa medias del dataset.\n"
                 "• Mientras más datos, mejor explicación (y usualmente mayor precisión).",
            style="Mini.TLabel",
            justify="left"
        ).pack(anchor="w", pady=(8, 10))

        btns = ttk.Frame(wrap, style="CardBody.TFrame")
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="▶ Analizar", style="Accent.TButton", command=self.run_analysis).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(btns, text="↺ Limpiar", style="Soft.TButton", command=self.clear_all).pack(side="left", fill="x", expand=True, padx=(6, 0))

        self.status_label = ttk.Label(wrap, text="Estado: —", style="Mini.TLabel")
        self.status_label.pack(anchor="w", pady=(14, 0))

    def _field(self, parent, label, var):
        row = ttk.Frame(parent, style="CardBody.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, style="Mini.TLabel").pack(anchor="w")
        e = ttk.Entry(row, textvariable=var, style="Field.TEntry")
        e.pack(fill="x", pady=(2, 0))

    def _build_dashboard_skeleton(self):
        # Cards arrangement (grid)
        inner = self.right.inner
        inner.configure(style="Header.TFrame")
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)
        inner.columnconfigure(2, weight=1)

        # Row 0: top big result + summary
        self.card_result = Card(inner, "Resultado", "Predicción y explicación compacta del modelo.")
        self.card_result.grid(row=0, column=0, sticky="nsew", padx=8, pady=8, columnspan=1)

        self.card_used = Card(inner, "Datos usados (final)", "Qué se tomó del usuario vs estimado/media.")
        self.card_used.grid(row=0, column=1, sticky="nsew", padx=8, pady=8, columnspan=2)

        # Row 1: images original/crop/gradcam
        self.card_img1 = Card(inner, "Imagen original", "Entrada capturada.")
        self.card_img1.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        self.card_img2 = Card(inner, "Recorte (carne)", "ROI segmentada (sin fondo) — preprocesamiento robusto.")
        self.card_img2.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)

        self.card_img3 = Card(inner, "Grad-CAM (zona decisiva)", "Zonas cálidas = regiones que más influyeron en la decisión.")
        self.card_img3.grid(row=1, column=2, sticky="nsew", padx=8, pady=8)

        # Row 2: LAB channels
        self.card_lab1 = Card(inner, "LAB — L* (luminosidad)", "Más bajo = más oscuro. Cambios pueden asociarse a degradación.")
        self.card_lab1.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)

        self.card_lab2 = Card(inner, "LAB — a* (rojo↔verde)", "Más alto = más rojo. Baja con oxidación.")
        self.card_lab2.grid(row=2, column=1, sticky="nsew", padx=8, pady=8)

        self.card_lab3 = Card(inner, "LAB — b* (amarillo↔azul)", "Más alto = más amarillento. Puede aumentar en deterioro.")
        self.card_lab3.grid(row=2, column=2, sticky="nsew", padx=8, pady=8)

        # Row 3: plots
        self.card_plot1 = Card(inner, "Probabilidades por clase", "Lectura directa de la salida del modelo.")
        self.card_plot1.grid(row=3, column=0, sticky="nsew", padx=8, pady=8)

        self.card_plot2 = Card(inner, "Histogramas LAB (ROI)", "Distribución de valores: homogeneidad vs variación.")
        self.card_plot2.grid(row=3, column=1, sticky="nsew", padx=8, pady=8)

        self.card_plot3 = Card(inner, "Comparación vs medias por clase", "Tu muestra vs perfiles promedio del dataset.")
        self.card_plot3.grid(row=3, column=2, sticky="nsew", padx=8, pady=8)

        # Row 4: stats + legend
        self.card_stats = Card(inner, "Estadística cuantitativa (ROI)", "Valores numéricos que sustentan la decisión.")
        self.card_stats.grid(row=4, column=0, sticky="nsew", padx=8, pady=8, columnspan=2)

        self.card_legend = Card(inner, "Leyenda interpretativa", "Guía humana (no reemplaza al modelo) para entender el resultado.")
        self.card_legend.grid(row=4, column=2, sticky="nsew", padx=8, pady=8)

        # Fill placeholders
        self._fill_placeholders()

    def _fill_placeholders(self):
        for card in [
            self.card_result, self.card_used,
            self.card_img1, self.card_img2, self.card_img3,
            self.card_lab1, self.card_lab2, self.card_lab3,
            self.card_plot1, self.card_plot2, self.card_plot3,
            self.card_stats, self.card_legend
        ]:
            for w in card.body.winfo_children():
                w.destroy()
            ttk.Label(card.body, text="Carga una imagen para ver resultados aquí.", style="Mini.TLabel").pack(anchor="w")

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------
    def pick_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.current_path = path
        self.status_label.configure(text=f"Imagen cargada: {os.path.basename(path)}")

    def clear_all(self):
        self.current_path = None
        self.last_result = None
        self.var_temp.set("")
        self.var_dia.set("")
        self.var_ph.set("")
        self.var_L.set("")
        self.var_a.set("")
        self.var_b.set("")
        self._photo_refs.clear()
        self.status_label.configure(text="Estado: —")
        self._fill_placeholders()

    def run_analysis(self):
        if not self.current_path:
            messagebox.showwarning("Falta imagen", "Primero carga una imagen.")
            return

        # parse optional inputs
        tab_opt = {
            "temp": safe_float(self.var_temp.get()),
            "dia":  safe_float(self.var_dia.get()),
            "pH":   safe_float(self.var_ph.get()),
            "L":    safe_float(self.var_L.get()),
            "a":    safe_float(self.var_a.get()),
            "b":    safe_float(self.var_b.get()),
        }

        # basic validation (si ingresan)
        if tab_opt["pH"] is not None and not (0.0 <= tab_opt["pH"] <= 14.0):
            messagebox.showerror("Dato inválido", "pH debe estar entre 0 y 14.")
            return

        try:
            self.status_label.configure(text="Analizando... (usa GPU si está disponible)")
            self.update_idletasks()

            res = infer_multimodal(
                self.model, self.cam,
                self.current_path,
                tab_opt,
                self.global_mean,
                self.class_means,
                self.tab_mean,
                self.tab_std
            )
            self.last_result = res
            self._render_results(res)

        except Exception as e:
            messagebox.showerror("Error en análisis", f"Ocurrió un error:\n\n{e}")
            self.status_label.configure(text="Error en análisis.")

    # --------------------------------------------------------
    # Rendering
    # --------------------------------------------------------
    def _render_results(self, res):
        self._photo_refs.clear()

        pred = res["pred_name"]
        conf = float(res["probs"][res["pred_idx"]] * 100.0)

        # result label color
        if pred == "FRESCO":
            style_lbl = "OK.TLabel"
        elif pred == "SEMIFRESCO":
            style_lbl = "Warn.TLabel"
        else:
            style_lbl = "Danger.TLabel"

        # --- Card Resultado ---
        self._clear_card(self.card_result)
        ttk.Label(self.card_result.body, text=f"Estado: {pred}", style=style_lbl).pack(anchor="w")
        ttk.Label(self.card_result.body, text=f"Confianza: {conf:.1f}%", style="Mini.TLabel").pack(anchor="w", pady=(2, 10))

        txt = tk.Text(self.card_result.body, height=18, wrap="word", bd=0, highlightthickness=0)
        txt.insert("1.0", res["explain_text"])
        txt.configure(state="disabled", bg=CARD_BG, fg=TXT_MAIN, font=("Segoe UI", 9))
        txt.pack(fill="both", expand=True)

        # --- Card Datos usados ---
        self._clear_card(self.card_used)
        used = res["used"]
        src = res["used_src"]

        summary = []
        for k in TAB_COLS:
            summary.append(f"{k:>4}: {used[k]:.3f}   ({src[k]})")
        summary.append("")
        summary.append(f"WI (blancura): {res['WI']:.2f}")
        summary.append(f"RI (enrojecimiento): {res['RI']:.3f}")

        lab_s = res["lab_stats"]
        summary.append("")
        summary.append("Stats ROI (percentiles):")
        summary.append(f"L*: p10={lab_s['L_stats']['p10']:.2f}  p50={lab_s['L_stats']['p50']:.2f}  p90={lab_s['L_stats']['p90']:.2f}")
        summary.append(f"a*: p10={lab_s['a_stats']['p10']:.2f}  p50={lab_s['a_stats']['p50']:.2f}  p90={lab_s['a_stats']['p90']:.2f}")
        summary.append(f"b*: p10={lab_s['b_stats']['p10']:.2f}  p50={lab_s['b_stats']['p50']:.2f}  p90={lab_s['b_stats']['p90']:.2f}")
        cb_a = plot_colorbar("RdYlGn_r", "a*: verde → rojo")
        cb_b = plot_colorbar("PuOr", "b*: azul → amarillo")

        lbl = ttk.Label(self.card_used.body, text="\n".join(summary), style="Mono.TLabel", justify="left")
        lbl.pack(anchor="w")

        # --- Images cards ---
        self._render_image_card(self.card_img1, res["img_original"], max_w=520, max_h=290)
        self._render_image_card(self.card_img2, res["img_crop"],     max_w=520, max_h=290)
        self._render_image_card(self.card_img3, res["img_gradcam"],  max_w=520, max_h=290)

        # --- LAB cards ---
        self._render_image_card(self.card_lab1, res["img_lab_L"], max_w=520, max_h=240)
        self._render_image_card(self.card_lab2, res["img_lab_A"], max_w=520, max_h=240)
        self._render_image_card(self.card_lab3, res["img_lab_B"], max_w=520, max_h=240)

        # --- Plots cards ---
        self._render_pil_card(self.card_plot1, res["plot_probs"],   max_w=520, max_h=260)
        self._render_pil_card(self.card_plot2, res["plot_hist"],    max_w=520, max_h=260)
        self._render_pil_card(self.card_plot3, res["plot_compare"], max_w=520, max_h=260)

        # --- Stats card ---
        self._clear_card(self.card_stats)

        # small table-like view
        st = res["lab_stats"]
        rows = [
            ("L* (mean)", st["L_stats"]["mean"], "std", st["L_stats"]["std"]),
            ("a* (mean)", st["a_stats"]["mean"], "std", st["a_stats"]["std"]),
            ("b* (mean)", st["b_stats"]["mean"], "std", st["b_stats"]["std"]),
            ("L* (min/max)", st["L_stats"]["min"], "→", st["L_stats"]["max"]),
            ("a* (min/max)", st["a_stats"]["min"], "→", st["a_stats"]["max"]),
            ("b* (min/max)", st["b_stats"]["min"], "→", st["b_stats"]["max"]),
        ]

        grid = ttk.Frame(self.card_stats.body, style="CardBody.TFrame")
        grid.pack(fill="x")

        for r, (k1, v1, k2, v2) in enumerate(rows):
            ttk.Label(grid, text=k1, style="Mini.TLabel").grid(row=r, column=0, sticky="w", padx=(0, 10), pady=2)
            ttk.Label(grid, text=f"{v1:.3f}", style="Mono.TLabel").grid(row=r, column=1, sticky="w", pady=2)
            ttk.Label(grid, text=k2, style="Mini.TLabel").grid(row=r, column=2, sticky="w", padx=(18, 10), pady=2)
            ttk.Label(grid, text=f"{v2:.3f}" if isinstance(v2, (int,float)) else str(v2), style="Mono.TLabel").grid(row=r, column=3, sticky="w", pady=2)

        ttk.Separator(self.card_stats.body).pack(fill="x", pady=10)

        # overlay mask image
        ttk.Label(self.card_stats.body, text="Overlay máscara (zona analizada en la imagen original)", style="Mini.TLabel").pack(anchor="w", pady=(0, 6))
        self._render_image_inline(self.card_stats.body, res["img_overlay_mask"], max_w=820, max_h=260)

        # --- Legend card ---
        self._clear_card(self.card_legend)
        legend_text = (
            "Rangos orientativos (según tu dataset):\n"
            "• pH: ~6.1–6.3 (Fresco) | ~6.3–7.0 (Semifresco) | >7.0 (No apto)\n"
            "• a*: más alto = más rojo (mejor aspecto), baja con oxidación\n"
            "• b*: más alto = más amarillento (tendencia a deterioro)\n"
            "• L*: más bajo = más oscuro (frecuente en degradación)\n\n"
            "Cómo leer los gráficos:\n"
            "• Barras de probabilidades: salida directa del modelo.\n"
            "• Histogramas LAB: dispersión alta = color/estado heterogéneo.\n"
            "• Comparación por clase: tu muestra vs perfiles promedio del dataset.\n\n"
            "Grad-CAM:\n"
            "• Zonas calientes (rojo/amarillo) = regiones más influyentes para la IA.\n"
            "• Si se concentra en manchas/texturas → evidencia visual del deterioro."
        )
        t = tk.Text(self.card_legend.body, height=18, wrap="word", bd=0, highlightthickness=0)
        t.insert("1.0", legend_text)
        t.configure(state="disabled", bg=CARD_BG, fg=TXT_MAIN, font=("Segoe UI", 9))
        t.pack(fill="both", expand=True)

        self.status_label.configure(text=f"Listo ✅  |  {pred} ({conf:.1f}%)")

    def _clear_card(self, card):
        for w in card.body.winfo_children():
            w.destroy()

    def _render_image_card(self, card, rgb, max_w=520, max_h=290):
        self._clear_card(card)
        img = tkimg_from_np(rgb, max_w=max_w, max_h=max_h)
        self._photo_refs.append(img)
        lbl = ttk.Label(card.body, image=img, background=CARD_BG)
        lbl.pack(anchor="center")

    def _render_pil_card(self, card, pil_img, max_w=520, max_h=290):
        self._clear_card(card)
        img = tkimg_from_pil(pil_img, max_w=max_w, max_h=max_h)
        self._photo_refs.append(img)
        lbl = ttk.Label(card.body, image=img, background=CARD_BG)
        lbl.pack(anchor="center")

    def _render_image_inline(self, parent, rgb, max_w=820, max_h=260):
        img = tkimg_from_np(rgb, max_w=max_w, max_h=max_h)
        self._photo_refs.append(img)
        lbl = ttk.Label(parent, image=img, background=CARD_BG)
        lbl.pack(anchor="center")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Pequeña validación de paths
    missing = []
    for p in [MODEL_PATH, META_PATH, CSV_PATH]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        print("❌ Faltan archivos:", missing)
        print("Asegúrate de tener:")
        print(" - runs_multimodal/best_multimodal.pth")
        print(" - runs_multimodal/meta.json")
        print(" - dataset_full.csv")
        raise SystemExit(1)

    app = App()
    app.mainloop()
