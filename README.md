# PaddyCare AI 🌾

PaddyCare AI adalah aplikasi berbasis **Computer Vision** untuk membantu mendeteksi kondisi kesehatan tanaman padi melalui gambar. Sistem ini menggunakan model **EfficientNet-B0** untuk mengklasifikasikan 10 kondisi tanaman dan memberikan rekomendasi penanganan awal.

## Fitur Utama

- Deteksi 10 kelas kondisi tanaman padi.
- Hasil prediksi beserta confidence score.
- Rekomendasi penanganan dan pencegahan.
- Antarmuka bilingual (Indonesia & Inggris).
- Riwayat pemeriksaan tanaman.

## Teknologi

- **Deep Learning:** PyTorch, EfficientNet-B0
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Database:** SQLite

## Performa Model

| Metrik | Hasil |
|---|---|
| Test Accuracy | 93,66% |
| Macro F1-Score | 93,57% |
| Jumlah Kelas | 10 |

Hasil evaluasi diperoleh dari data pengujian model, bukan jaminan ketepatan setiap diagnosis di lapangan.

## Menjalankan Aplikasi

Instal dependensi:

```bash
python -m pip install -r requirements.txt
```

Pastikan model tersedia di `models/paddycare_model.pth`.

Jalankan backend:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Jalankan Streamlit di terminal lain:

```bash
python -m streamlit run frontend/app.py
```

Buka aplikasi melalui `http://localhost:8501`.

---

**PaddyCare AI** — Smart Rice Disease Detection

*by Farhan Fitran Maulana_DSML42.*
