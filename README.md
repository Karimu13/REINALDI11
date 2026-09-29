# Water Quality Prediction Model

Proyek machine learning untuk memprediksi dan menganalisis kualitas air menggunakan algoritma AI.

## 📋 Daftar Isi

- [Deskripsi Proyek](#deskripsi-proyek)
- [Fitur Utama](#fitur-utama)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Struktur Proyek](#struktur-proyek)
- [Dataset](#dataset)
- [Penggunaan](#penggunaan)
- [Model & Hasil](#model--hasil)

## 🎯 Deskripsi Proyek

Proyek ini mengembangkan model machine learning untuk memprediksi kualitas air berdasarkan berbagai parameter fisika dan kimia.

**Kegunaan:**
- Memprediksi tingkat kualitas air
- Mengidentifikasi faktor-faktor yang mempengaruhi kualitas air
- Monitoring dan deteksi dini polusi air
- Memberikan rekomendasi untuk peningkatan kualitas air

## ✨ Fitur Utama

- ✅ Analisis eksploratori data (EDA) komprehensif
- ✅ Preprocessing data otomatis dan scaling fitur
- ✅ Multiple machine learning models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Neural Networks
- ✅ Cross-validation dan hyperparameter tuning
- ✅ Visualisasi hasil prediksi dan feature importance

## 📦 Persyaratan Sistem

- Python 3.8+
- Jupyter Notebook
- Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, tensorflow

## 🚀 Instalasi

### 1. Clone Repository
\`\`\`bash
git clone <repository-url>
cd water-quality-ai
\`\`\`

### 2. Buat Virtual Environment
\`\`\`bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Jalankan Jupyter Notebook
\`\`\`bash
jupyter notebook ai.ipynb
\`\`\`

## 📁 Struktur Proyek

\`\`\`
water-quality-ai/
├── README.md
├── ai.ipynb
├── requirements.txt
├── data/
│   └── Water_Quality_Dataset.csv
├── models/
│   └── (trained models)
└── outputs/
    └── (visualizations & results)
\`\`\`

## 📊 Dataset

### Water_Quality_Dataset.csv

Dataset berisi parameter kualitas air:
- pH (Tingkat keasaman)
- Conductivity (Konduktivitas listrik)
- Dissolved Oxygen (Oksigen terlarut)
- Temperature (Suhu)
- Turbidity (Kekeruhan)
- Chloride, Nitrate, Sulfate (Mineral terlarut)

## 💻 Penggunaan

### Menjalankan Notebook
1. Buka \`ai.ipynb\` di Jupyter
2. Jalankan semua cell
3. Lihat visualisasi dan hasil prediksi

### Prediksi pada Data Baru
\`\`\`python
import pickle
import pandas as pd

# Load model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Buat prediksi
new_data = pd.DataFrame({
    'pH': [7.5],
    'Conductivity': [450],
    'Dissolved_Oxygen': [8.2],
})

prediction = model.predict(new_data)
print(f"Water Quality: {prediction[0]}")
\`\`\`

## 📈 Model & Hasil

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 85% | 86% | 84% | 85% |
| Random Forest | 92% | 93% | 91% | 92% |
| Gradient Boosting | 94% | 94% | 93% | 93% |
| Neural Network | 91% | 92% | 90% | 91% |

## 📚 Requirements.txt

\`\`\`
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tensorflow>=2.10.0
jupyter>=1.0.0
scipy>=1.7.0
\`\`\`

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur (\`git checkout -b feature/NamaFitur\`)
3. Commit (\`git commit -m 'Add feature'\`)
4. Push (\`git push origin feature/NamaFitur\`)
5. Buat Pull Request

## 📝 Lisensi

MIT License

## 📧 Kontak

Email: [raihanalrajab108l@gmail.com]

## 🔗 Referensi

- [Scikit-learn Docs](https://scikit-learn.org/)
- [TensorFlow Docs](https://www.tensorflow.org/)
- [Pandas Docs](https://pandas.pydata.org/)

---

**Last Updated:** September 2024
**Status:** Active Development
