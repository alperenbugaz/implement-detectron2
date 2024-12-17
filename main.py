import cv2
import matplotlib.pyplot as plt
import torch
import sys

print(torch.__version__)
print(sys.version)

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# Segmentasyon için Detectron2 ayarlarını yapılandır
def setup_model():
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Güven eşiği (0.5 olarak ayarlandı)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU veya CPU
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    return DefaultPredictor(cfg), cfg  # cfg'yi döndürüyoruz


# Görüntüyü segmente et ve sonucu görselleştir
def segment_image(image_path, predictor, cfg):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mask R-CNN ile tahmin yap
    outputs = predictor(image_rgb)

    # Segmentasyon sonuçlarını görselleştir
    v = Visualizer(image_rgb[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Segmentasyon sonuçlarını göster
    plt.figure(figsize=(15, 10))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.axis("off")
    plt.title("Segmented Image with Mask R-CNN")
    plt.show()

    # Engelleri (nesneleri) ekrana yazdırma
    instances = outputs["instances"].to("cpu")
    labels = instances.pred_classes.numpy()
    print("Detected objects (class labels):", labels)

    # Yol üzerinde engelleri göstermek için sınır kutuları ekleyelim
    for box in instances.pred_boxes:
        box = box.tensor.numpy().astype(int)[0]
        cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)  # Engelleri yeşil kutu ile göster

    # Görüntü üzerinde yol tespiti yapılacaksa, yolun koordinatlarını manuel olarak belirleyebilirsiniz
    # Örneğin, yol bir çizgi olarak verildiyse (x1, y1) ve (x2, y2) noktalarını çizelim
    # Bu kısmı özelleştirebilirsiniz (örnek yol koordinatları)
    cv2.line(image_rgb, (50, 200), (450, 200), (255, 0, 0), 2)  # Mavi çizgiyle yolu göstermek için

    # Segmentasyon ve yol çizgisi ile güncellenmiş görüntüyü göster
    plt.figure(figsize=(15, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Image with Path and Obstacles")
    plt.show()


if __name__ == "__main__":  # Yazım hatası düzeltildi
    # Modeli yükle ve cfg'yi al
    predictor, cfg = setup_model()

    # Segmentasyon yapılacak görüntüyü seç
    image_path = "32_A.jpg"  # Buraya görüntü dosya yolunu ekleyin
    segment_image(image_path, predictor, cfg)


#git clone https://github.com/facebookresearch/detectron2.git
#cd detectron2
#pip install -e .