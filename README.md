# 📊 Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institute

## 👤 Identitas
- **Nama**: Ahmad Firman
- **Email**: ahmadfirman9998@gmail.com
- **ID Dicoding**: ahmad_firman

## 📁 Deskripsi Proyek
Proyek ini bertujuan untuk menyelesaikan masalah prediksi menggunakan machine learning di institusi pendidikan Jaya Jaya Institute. Dataset yang digunakan berasal dari Google Drive dan diunduh dalam format CSV. Proyek melibatkan beberapa tahap utama: eksplorasi data, pembersihan data, preprocessing, pemodelan, evaluasi, dan penyimpanan model akhir.

## 📌 Tahapan Proyek

### Persiapan
- Import library seperti `pandas`, `sklearn`, `xgboost`, `lightgbm`, `imblearn`, dll.
- Download dan muat data (`data.csv`) dari Google Drive.

### Data Understanding
- Analisis struktur data (`df.info()`, `df.describe()`).
- Visualisasi korelasi antar fitur menggunakan heatmap.

### Preprocessing
- Penanganan missing values.
- Encoding label dan one-hot pada fitur kategorikal.
- Feature scaling menggunakan `StandardScaler`.
- Penyeimbangan data dengan `SMOTE`.

### Modeling
Model yang digunakan:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- XGBoost
- LightGBM

### Evaluation
- Evaluasi performa model menggunakan metrik: Accuracy, Precision, Recall, F1-Score, ROC AUC.
- Plot confusion matrix dan kurva ROC.

### Deployment Model dan Dashboard
- Model terbaik disimpan ke file `.pkl` menggunakan `pickle` untuk digunakan kembali dalam prediksi.
- Dashboard dapat dilihat di [Dashboard]() yang juga prediksi modelnya dapat dilakukan di sini

## 🚀 Cara Menjalankan Sistem
1. Buka antarmuka prediksi di halaman dashboard.
2. Isi nilai untuk setiap kolom fitur yang diminta.
3. Klik tombol "Predict" untuk melihat hasil prediksi.

## 📦 Dependencies
Beberapa library penting yang digunakan:
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `imblearn`
- `gdown` (untuk mengunduh data dari Google Drive)

Pastikan seluruh dependency telah diinstal sebelum menjalankan notebook:

```bash
pip install -r requirements.txt
```

Atau install secara manual:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn gdown
```

#   d a s h b o a r d - m l  
 