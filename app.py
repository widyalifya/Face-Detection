import cv2
from deepface import DeepFace

# Lokasi foto
image_path = "images/images2.jpg"

# Baca foto menggunakan OpenCV
image = cv2.imread(image_path)

if image is None:
    print("Foto tidak ditemukan!")
    exit()

# Deteksi wajah menggunakan CenterFace
faces = DeepFace.extract_faces(
    img_path=image,
    detector_backend="centerface",
    enforce_detection=False,
    align=True
)

print(f"Jumlah wajah terdeteksi: {len(faces)}")

# Gambar kotak pada setiap wajah
for face in faces:
    area = face["facial_area"]

    x = area["x"]
    y = area["y"]
    w = area["w"]
    h = area["h"]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        "Face",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# Tampilkan hasil
cv2.imshow("Face Detection - CenterFace", image)

cv2.waitKey(0)
cv2.destroyAllWindows()