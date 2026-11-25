"""
YOLO11n — Pascal VOC Nesne Tespiti | Gradio Uygulaması

Colab'da eğitilen weights/best.pt modelini yükler. Kullanıcı bir fotoğraf yükler;
uygulama kutucukları, sınıf etiketlerini, güven skorlarını ve inference süresini gösterir.

Çalıştırma:  python app.py   →  tarayıcıda http://127.0.0.1:7860
"""
import csv
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

# ----------------------------- Ayarlar -----------------------------
ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "test_outputs"
EXAMPLES_DIR = ROOT / "test_images"
IMG_SIZE = 640          # eğitimdeki giriş boyutu ile aynı
DEFAULT_CONF = 0.25     # test değerlendirmesinde (Hücre 10) kullanılan eşik
DEFAULT_IOU = 0.70      # NMS IoU eşiği (Ultralytics varsayılanı)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# ----------------------------- Model -------------------------------
if not WEIGHTS.exists():
    raise FileNotFoundError(f"Model bulunamadı: {WEIGHTS}\n"
                            "Colab'dan indirdiğiniz best.pt dosyasını weights/ klasörüne koyun.")

model = YOLO(str(WEIGHTS))
CLASS_NAMES = model.names  # {0: 'aeroplane', ..., 19: 'tvmonitor'}

# Isınma: ilk çalıştırmadaki yükleme gecikmesi ölçümü bozmasın
_dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for _ in range(2):
    model.predict(_dummy, imgsz=IMG_SIZE, device=DEVICE, verbose=False)


# ----------------------------- Tahmin ------------------------------
def detect(image: Image.Image, conf: float, iou: float, save: bool):
    if image is None:
        raise gr.Error("Lütfen önce bir fotoğraf yükleyin.")

    image = ImageOps.exif_transpose(image).convert("RGB")  # telefon fotoğrafı yönü düzeltilir

    t0 = time.perf_counter()
    result = model.predict(image, imgsz=IMG_SIZE, conf=conf, iou=iou,
                           device=DEVICE, verbose=False)[0]
    total_ms = (time.perf_counter() - t0) * 1000

    annotated = result.plot()[:, :, ::-1]  # BGR → RGB

    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append([CLASS_NAMES[int(box.cls)], round(float(box.conf), 3),
                     int(x1), int(y1), int(x2), int(y2)])
    rows.sort(key=lambda r: r[1], reverse=True)

    sp = result.speed
    summary = (
        f"### {len(rows)} nesne tespit edildi\n"
        f"| Süre | ms |\n|---|---|\n"
        f"| **Toplam (uçtan uca)** | **{total_ms:.1f}** |\n"
        f"| Ön işleme | {sp['preprocess']:.1f} |\n"
        f"| Model (inference) | {sp['inference']:.1f} |\n"
        f"| Son işleme (NMS) | {sp['postprocess']:.1f} |\n\n"
        f"Cihaz: `{DEVICE_NAME}` · Görüntü: {image.width}×{image.height} → {IMG_SIZE} · "
        f"conf ≥ {conf:.2f} · NMS IoU {iou:.2f}"
    )

    if save:
        summary += "\n\n" + save_result(annotated, rows, total_ms, sp, conf, iou)

    return annotated, rows, summary


def save_result(annotated, rows, total_ms, speed, conf, iou):
    """Rapor için: işaretli görüntüyü ve tespitleri test_outputs/ klasörüne kaydeder."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = OUTPUT_DIR / f"result_{stamp}.jpg"
    Image.fromarray(annotated).save(img_path, quality=95)

    log_path = OUTPUT_DIR / "detections_log.csv"
    new_file = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["file", "class", "confidence", "x1", "y1", "x2", "y2",
                        "total_ms", "inference_ms", "conf_thr", "iou_thr", "device"])
        for r in rows or [["-", "", "", "", "", ""]]:
            w.writerow([img_path.name, *r, round(total_ms, 2), round(speed["inference"], 2),
                        conf, iou, DEVICE_NAME])
    return f"💾 Kaydedildi: `test_outputs/{img_path.name}`"


# ----------------------------- Arayüz ------------------------------
examples = sorted(p for p in EXAMPLES_DIR.glob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if EXAMPLES_DIR.exists() else []

with gr.Blocks(title="YOLO11n Pascal VOC Nesne Tespiti") as demo:
    gr.Markdown(
        "# YOLO11n ile Nesne Tespiti (Pascal VOC)\n"
        "Bir fotoğraf yükleyin; model nesneleri kutucuk, etiket ve güven skoru ile işaretlesin.\n\n"
        f"**Tanınan 20 sınıf:** {', '.join(CLASS_NAMES.values())}"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Fotoğraf yükle", sources=["upload", "clipboard", "webcam"])
            conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Güven eşiği (confidence)")
            iou = gr.Slider(0.30, 0.90, value=DEFAULT_IOU, step=0.05, label="NMS IoU eşiği")
            save = gr.Checkbox(value=True, label="Sonucu test_outputs/ klasörüne kaydet")
            btn = gr.Button("Tespit et", variant="primary")
        with gr.Column():
            out_img = gr.Image(type="numpy", label="Tespit sonucu")
            out_info = gr.Markdown()
    out_table = gr.Dataframe(headers=["Sınıf", "Güven", "x1", "y1", "x2", "y2"],
                             label="Tespit edilen nesneler", interactive=False)

    btn.click(detect, inputs=[inp, conf, iou, save], outputs=[out_img, out_table, out_info])
    if examples:
        gr.Examples(examples=[[str(p)] for p in examples], inputs=[inp], label="Örnek fotoğraflar")

if __name__ == "__main__":
    demo.launch()
