# Laporan Proyek Machine Learning - Muhammad Daffa Eka Pramudita
## Domain Proyek
Diabetes merupakan penyakit kronis yang menjadi salah satu penyebab utama kematian dan pengeluaran biaya kesehatan yang tinggi secara global. Berdasarkan data dari World Health Organization (WHO), jumlah penderita diabetes meningkat secara signifikan dari tahun ke tahun. Deteksi dini dan penanganan yang tepat menjadi kunci utama dalam mencegah komplikasi yang lebih serius.

Dalam proyek ini, kami membangun model klasifikasi risiko diabetes berdasarkan berbagai parameter gaya hidup dan kesehatan seperti diet, tingkat stres, kepatuhan terhadap pengobatan, serta hidrasi. Tujuan utama dari proyek ini adalah untuk memprediksi tingkat risiko diabetes pada seseorang sehingga dapat dilakukan pencegahan atau pengobatan lebih dini.

## Business Understanding
### Problem Statement
1. Bagaimana cara memanfaatkan informasi pola hidup guna memprediksi kategori diabetes pada seseorang?
2. Apa faktor-faktor yang paling berkontribusi terhadap peningkatan risiko diabetes?
### Goals
1. Membangun model klasifikasi untuk memprediksi risiko diabetes ke dalam tiga kategori: Low Risk, Moderate Risk, dan High Risk.
2. Mengidentifikasi fitur-fitur gaya hidup yang memiliki kontribusi paling besar terhadap peningkatan risiko diabetes, berdasarkan hasil pemodelan machine learning.
### Solution Statement
1. Menerapkan beberapa algoritma machine learning seperti Decision Tree, dan Random Forest. Serta membandingkan performa masing-masing model menggunakan metrik akurasi, precision, recall, dan f1-score.
2. Menganalisis kontribusi setiap fitur input terhadap prediksi risiko diabetes.

## Data Understanding
Dataset yang digunakan diperoleh dari Kaggle dengan link : [Diabetes Prediction datasets](https://www.kaggle.com/datasets/kevintan701/diabetes-prediction-datasets). <br>Dataset ini berisi informasi terkait faktor-faktor gaya hidup dan kondisi kesehatan yang dapat memengaruhi risiko diabetes. Dataset ini memiliki **1000 baris** data dengan total **11 kolom**. Hasil dari dataset ini ditentukan di kolom risk_score dalam rentang **0 sampai 78**

### Berikut merupakan kolom-kolom yang terdapat di dalam dataset
| Nama Kolom                    | Deskripsi                                                                                                                |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| User ID                      | Merupakan identitas unik yang diberikan kepada setiap pengguna untuk menjaga kerahasiaan data dan memudahkan pelacakan.  |
| Date                         | Menunjukkan tanggal pencatatan data, yang mencerminkan bahwa data memiliki sifat deret waktu (time-series).              |
| Weight (kg)                  | Berat badan pengguna dalam kilogram. Berat badan berperan penting dalam menilai risiko obesitas dan diabetes.            |
| Height (cm)                  | Tinggi badan pengguna dalam sentimeter. Digunakan bersama berat badan untuk menghitung BMI, indikator risiko diabetes.   |
| Blood Glucose (mg/dL)        | Kadar gula dalam darah pengguna dalam mg/dL. Merupakan indikator utama dalam mendeteksi atau memantau kondisi diabetes.  |
| Physical Activity (minutes/day) | Lama aktivitas fisik harian pengguna dalam menit. Aktivitas fisik membantu mengontrol kadar gula darah.               |
| Diet Quality                 | Menunjukkan kualitas pola makan ('healthy' atau 'unhealthy') yang berpengaruh pada pencegahan dan pengelolaan diabetes.  |
| Medication Adherence         | Menggambarkan kepatuhan pengguna terhadap konsumsi obat, dikategorikan sebagai 'good' atau 'poor'.                       |
| Stress Level                 | Tingkat stres pengguna ('low', 'medium', atau 'high'). Stres kronis dapat berdampak negatif pada gula darah.             |
| Sleep Duration (hours)       | Jumlah jam tidur pengguna per hari. Tidur cukup membantu menjaga kesehatan dan kestabilan gula darah.                   |
| Hydration Status             | Menyatakan apakah pengguna terhidrasi dengan baik ('yes' atau 'no'). Hidrasi mendukung keseimbangan gula darah.         |
| BMI                          | Indeks Massa Tubuh yang dihitung dari berat dan tinggi badan, digunakan untuk menilai status berat badan pengguna.       |
| Risk Score                   | Skor risiko diabetes yang dihitung dari berbagai parameter kesehatan pengguna. Digunakan untuk pengelompokan tingkat risiko. |
