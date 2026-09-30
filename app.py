import cv2
import numpy as np


# =========================
# KONFIGURASI
# =========================

IMAGE_PATH = "images/images2.jpg"
MODEL_PATH = "models/centerface.onnx"

CONFIDENCE_THRESHOLD = 0.5


# =========================
# FUNGSI UKURAN INPUT
# =========================

def transform(height, width):
    new_height = int(np.ceil(height / 32) * 32)
    new_width = int(np.ceil(width / 32) * 32)

    scale_height = new_height / height
    scale_width = new_width / width

    return new_height, new_width, scale_height, scale_width


# =========================
# FUNGSI NMS
# =========================

def nms(boxes, scores, threshold=0.3):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(scores)[::-1]

    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(order)):

        current = order[i]

        if suppressed[current]:
            continue

        keep.append(current)

        for j in range(i + 1, len(order)):

            other = order[j]

            if suppressed[other]:
                continue

            xx1 = max(x1[current], x1[other])
            yy1 = max(y1[current], y1[other])
            xx2 = min(x2[current], x2[other])
            yy2 = min(y2[current], y2[other])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            intersection = w * h

            overlap = intersection / (
                areas[current] + areas[other] - intersection
            )

            if overlap >= threshold:
                suppressed[other] = True

    return keep


# =========================
# MEMBACA FOTO
# =========================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Foto tidak ditemukan!")
    exit()


height, width = image.shape[:2]


# =========================
# LOAD MODEL CENTERFACE
# =========================

net = cv2.dnn.readNetFromONNX(MODEL_PATH)


# =========================
# UKURAN INPUT CENTERFACE
# =========================

input_height, input_width, scale_height, scale_width = transform(
    height,
    width
)


# =========================
# PREPROCESSING
# =========================

blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1.0,
    size=(input_width, input_height),
    mean=(0, 0, 0),
    swapRB=True,
    crop=False
)

net.setInput(blob)


# =========================
# INFERENCE CENTERFACE
# =========================

heatmap, scale, offset, landmarks = net.forward(
    ["537", "538", "539", "540"]
)


# =========================
# DECODE DETECTION
# =========================

heatmap = np.squeeze(heatmap)

scale0 = scale[0, 0, :, :]
scale1 = scale[0, 1, :, :]

offset0 = offset[0, 0, :, :]
offset1 = offset[0, 1, :, :]

ys, xs = np.where(heatmap > CONFIDENCE_THRESHOLD)

boxes = []

for y, x in zip(ys, xs):

    score = heatmap[y, x]

    box_height = np.exp(scale0[y, x]) * 4
    box_width = np.exp(scale1[y, x]) * 4

    offset_y = offset0[y, x]
    offset_x = offset1[y, x]

    x1 = max(
        0,
        (x + offset_x + 0.5) * 4 - box_width / 2
    )

    y1 = max(
        0,
        (y + offset_y + 0.5) * 4 - box_height / 2
    )

    x1 = min(x1, input_width)
    y1 = min(y1, input_height)

    x2 = min(x1 + box_width, input_width)
    y2 = min(y1 + box_height, input_height)

    boxes.append([
        x1,
        y1,
        x2,
        y2,
        score
    ])


# =========================
# NMS
# =========================

if len(boxes) > 0:

    boxes = np.array(boxes, dtype=np.float32)

    keep = nms(
        boxes[:, :4],
        boxes[:, 4],
        threshold=0.3
    )

    boxes = boxes[keep]

else:

    boxes = np.empty(
        shape=(0, 5),
        dtype=np.float32
    )


# =========================
# KEMBALIKAN KE UKURAN FOTO
# =========================

if len(boxes) > 0:

    boxes[:, 0] /= scale_width
    boxes[:, 2] /= scale_width

    boxes[:, 1] /= scale_height
    boxes[:, 3] /= scale_height


# =========================
# GAMBAR BOUNDING BOX
# =========================

print(f"Jumlah wajah terdeteksi: {len(boxes)}")

for box in boxes:

    x1, y1, x2, y2, confidence = box

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Face {confidence:.2f}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# =========================
# TAMPILKAN HASIL
# =========================

cv2.imshow(
    "Face Detection - CenterFace + OpenCV",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()