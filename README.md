# Face Detection - CenterFace + OpenCV

Project ini merupakan aplikasi deteksi wajah pada gambar menggunakan **CenterFace** dan **OpenCV**.

Aplikasi menerima sebuah foto sebagai input, kemudian model CenterFace digunakan untuk mendeteksi wajah. Hasil deteksi ditampilkan menggunakan bounding box pada wajah yang ditemukan.

## Deskripsi

Face Detection merupakan proses untuk menemukan lokasi wajah manusia pada sebuah gambar.

Pada project ini, **CenterFace** digunakan sebagai algoritma deteksi wajah, sedangkan **OpenCV** digunakan untuk membaca gambar, menjalankan model ONNX, melakukan pemrosesan gambar, dan menampilkan hasil deteksi.

## Teknologi yang Digunakan

- Python
- OpenCV
- NumPy
- CenterFace
- ONNX

## Alur Sistem

```text
Input Foto
    ↓
OpenCV
    ↓
Preprocessing
    ↓
CenterFace
    ↓
Post-processing
    ↓
Deteksi Wajah
    ↓
Bounding Box
    ↓
Output Foto

Struktur Project

face_detection/
├── app.py
├── images/
│   └── images2.jpg
├── models/
│   └── centerface.onnx
├── README.md
├── requirements.txt
├── .gitignore
└── venv/




Masuk ke folder project
cd Face-Detection

2. Membuat Virtual Environment
python -m venv venv

3. Install Library
pip install -r requirements.txt


Menyiapkan Model
Model CenterFace ONNX disimpan di dalam folder:

models/centerface.onnx

Menyiapkan Foto
Letakkan foto yang ingin digunakan di folder:

images/

Menjalankan Program
python app.py
