# Face Detection

Project ini merupakan aplikasi deteksi wajah pada gambar menggunakan Python dengan bantuan **OpenCV**, **DeepFace**, dan **CenterFace**.

## Deskripsi

Aplikasi ini digunakan untuk mendeteksi wajah pada sebuah foto. Foto dibaca menggunakan OpenCV, kemudian DeepFace digunakan untuk menjalankan proses deteksi dengan **CenterFace** sebagai detector backend.

Wajah yang berhasil terdeteksi akan diberikan **bounding box** atau kotak pada area wajah.

## Teknologi yang Digunakan

- Python
- OpenCV
- DeepFace
- CenterFace
- TensorFlow

## Alur Sistem

```text
Input Foto
    ↓
OpenCV
    ↓
DeepFace
    ↓
CenterFace
    ↓
Deteksi Wajah
    ↓
Bounding Box
    ↓
Output Foto
