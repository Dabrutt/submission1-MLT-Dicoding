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

## EDA
### Univariative Analysis
![image](https://github.com/user-attachments/assets/b10a0d55-4c57-4a17-b329-1aec61c8a6a6)

Dari visualisasi diatas dapat diinterpretasikan bahwa :
*   `Weight`: mayoritas 50-90 kg, puncak 65-70 kg.
*   `Height`: mayoritas 160-180 cm, puncak 170-175 cm.
*   `Blood_glucose`:  mayoritas 120-160.
*   `Physical_activity`: mayoritas memiliki aktivitas fisik rendah.
*   `Diet`: mayoritas menerapkan diet yang sehat.
*   `Medication_adherence`: mayoritas patuh (nilai 1).
*   `Stress_level`: mayoritas memiliki tingkat stres sangat rendah.
*   `Sleep_hours`: mayoritas tidur 6-8 jam, puncak 7-7.5 jam.
*   `Hydration_level`: mayoritas memiliki tingkat hidrasi yang cukup.
*   `BMI`: mayoritas 20-30.
*   `Risk_score`: mayoritas 25-45.

Kesimpulan Umum:
*   Beberapa variabel seperti `weight`, `height`, dan `bmi` berdistribusi cukup simetris/normal.
*   Beberapa variabel seperti `physical_activity`, `stress_level` sangat miring ke kanan, menunjukkan mayoritas pada nilai rendah.

### Multivariative Analysis
![image](https://github.com/user-attachments/assets/c55c9e03-47bc-4eb1-a81f-d8d61c8ddde2)

**Risiko Diabetes vs Berat Badan**:
*   Tidak ada pola linear atau korelasi yang jelas terlihat. Titik-titik tersebar acak, menunjukkan bahwa secara individual, berat badan dan tinggi badan saja tidak menunjukkan hubungan linear yang kuat dengan risiko diabetes dalam dataset ini.

**Risiko Diabetes vs Tinggi Badan:**
*   Mirip dengan berat badan, tidak ada pola linear yang jelas. Titik-titik tersebar secara acak. Tidak ada hubungan linear yang kuat antara tinggi badan dan risiko diabetes.

**Risiko Diabetes vs Gula Darah:**
*   Terdapat korelasi yang terlihat, meski dengan penyebaran yang cukup luas. Semakin tinggi kadar gula darah, cenderung semakin tinggi pula risiko diabetes. Hal ini konsisten dengan fakta medis, namun menunjukkan bahwa gula darah bukan satu-satunya penentu risiko.

**Risiko Diabetes vs Aktivitas Fisik:**
*   Ada indikasi korelasi lemah. Sedikit kecenderungan risiko diabetes menurun seiring peningkatan aktivitas fisik. Namun, ini masih sangat samar dan menunjukkan pengaruh yang tidak kuat secara individual dalam plot ini.

**Risiko Diabetes vs Lama Waktu Tidur:**
*   Tidak ada pola linear atau korelasi yang jelas. Titik-titik tersebar acak, mengindikasikan bahwa jumlah jam tidur secara langsung tidak menunjukkan hubungan linear yang signifikan dengan risiko diabetes di data ini.

**Risiko Diabetes vs Indeks Massa Tubuh (BMI):**
*   Menunjukkan korelasi positif yang paling jelas di antara semua fitur. Peningkatan BMI cenderung berkorelasi dengan peningkatan risiko diabetes. Ini sangat sesuai dengan bukti ilmiah bahwa BMI tinggi adalah faktor risiko utama diabetes. Meskipun demikian, masih ada variasi risiko pada setiap tingkat BMI.

![image](https://github.com/user-attachments/assets/1303a461-6ce0-4299-8dda-d933766ffac4)

**Risiko Diabetes vs Diet:**
   *   Sebagian besar orang dalam data ini memiliki pola diet "sehat" (nilai 1). Baik individu dengan diet sehat maupun tidak sehat tersebar di seluruh rentang risiko diabetes. Tidak terlihat korelasi linear langsung antara jenis diet dan skor risiko.

**Risiko Diabetes vs Kepatuhan Pengobatan:**
   *   Mayoritas besar partisipan sangat patuh minum obat (nilai 1). Seperti diet, tidak ada pola linear yang jelas antara kepatuhan dan risiko diabetes dari plot ini.

**Risiko Diabetes vs Level Stres:**
   *   Kebanyakan orang memiliki level stres rendah (0 atau 1), dengan sebagian kecil di level lebih tinggi (2). Individu dari semua level stres tersebar di seluruh rentang risiko diabetes. Tidak ada hubungan linear yang tampak.

**Risiko Diabetes vs Tingkat Hidrasi:**
   *   Hampir semua individu dalam dataset memiliki tingkat hidrasi yang "cukup" (nilai 1). Karena sangat sedikit data untuk "hidrasi kurang" (nilai 0), sulit menarik kesimpulan kuat tentang hubungannya dengan risiko diabetes.

## Data Preparation
### Memeriksa Missing Value

![image](https://github.com/user-attachments/assets/c00ec89a-ea56-4a98-992a-b282b7c3a262)
- Tidak terdapet missing value pada dataset ini

### Memeriksa Data Duplikat
![image](https://github.com/user-attachments/assets/29947a49-f46c-4852-91f9-0627be9aab6c)
- Tidak terdapat data duplikat pada dataset ini

### Memeriksa Outlier
![image](https://github.com/user-attachments/assets/106a3878-a2ba-4ae9-9e26-7f6e85633cc3)

![image](https://github.com/user-attachments/assets/5d659581-5970-4018-a36e-86f8835c2b35)
Berdasarkan informasi yang diberikan data ini mendapatkan total 22 outlier, dapat disimpulkan bahwa beberapa kolom dalam dataset yang mengandung nilai outlier seperti.
- Kolom weight memiliki 4 outlier, yang kemungkinan merepresentasikan individu dengan berat badan yang sangat rendah atau sangat tinggi dibandingkan populasi umum.
- Kolom height juga menunjukkan adanya 2 data yang menyimpang, bisa berasal dari individu yang sangat pendek atau sangat tinggi.
- Kolom physical_activity memiliki 3 outlier, yang dapat mencerminkan pengguna dengan tingkat aktivitas fisik harian yang jauh lebih rendah atau lebih tinggi dari mayoritas.
- Kolom blood_glucose memiliki 6 outlier, yang menunjukkan adanya kadar gula darah yang sangat rendah atau sangat tinggi.
- Kolom bmi mamiliki 8 outlier. Nilai-nilai BMI ekstrem ini dapat mengindikasikan masalah berat badan serius, seperti obesitas parah atau kekurangan gizi.

### Data Cleansing
![image](https://github.com/user-attachments/assets/e466f500-28e8-4e23-b9df-1ac9e44ce84c)

Kami hanya berfokus pada penghapusan outlier karena tidak adanya missing values dan data duplikat dalam dataset. Proses cleaning ini mengakibatkan berkurangnya jumlah data dari 1000 menjadi 979.

### Data Splitting
![image](https://github.com/user-attachments/assets/b76b394a-c75e-4c0f-be8a-452875ce914a)

Bagian ini berfungsi untuk mengubah nilai numerik dari kolom risk_score menjadi kategori kelas klasifikasi.
Kategorinya dibagi menjadi tiga:
*  Low Risk: Skor di bawah 35
*  Moderate Risk: Skor antara 35 hingga kurang dari 60
*  High Risk: Skor 60 atau lebih

Setelah dilakukan pembagian, outputnya menunjukkan:
*  470 data termasuk kategori Low Risk
*  442 data kategori Moderate Risk
*  67 data kategori High Risk

![image](https://github.com/user-attachments/assets/e8dc7917-e41e-40a9-81ec-5cda3c464dae)

Dataset dibagi menjadi bagian pelatihan dan pengujian. Sebanyak 685 data dialokasikan untuk training, sementara 294 data untuk testing, sehingga total keseluruhan data yang digunakan adalah 979.


## Modelling
### Algoritma Random Forest 
Random Forest Classifier adalah salah satu algoritma machine learning yang termasuk dalam metode ensemble learning, yaitu pendekatan yang menggabungkan beberapa model untuk meningkatkan akurasi dan kestabilan prediksi. Random Forest terdiri dari sejumlah pohon keputusan (decision trees) yang dibangun berdasarkan subset acak dari data pelatihan. Saat melakukan prediksi, setiap pohon akan memberikan “suara” terhadap suatu kelas, dan keputusan akhir ditentukan berdasarkan mayoritas suara (voting) dari seluruh pohon dalam ensemble tersebut.

Model ini memiliki keunggulan dalam mengurangi risiko overfitting, serta mampu menangani data yang kompleks dengan fitur-fitur yang saling berinteraksi. Random Forest juga berguna untuk mengidentifikasi feature importance, yakni fitur mana yang paling berkontribusi terhadap prediksi model.

Dalam implementasi model pada proyek ini, digunakan kode berikut:
```# === Model Random Forest ===  
rf_model = RandomForestClassifier(random_state=42)  
rf_model.fit(X_train, y_train)  
y_pred_rf = rf_model.predict(X_test)  
make_evaluation(y_test, y_pred_rf, title="Random Forest")
```

Parameter yang digunakan adalah:
*  random_state=42: Parameter ini digunakan untuk menjaga konsistensi dan memastikan bahwa hasil pelatihan model dapat direproduksi. Random Forest melibatkan proses acak dalam pembuatan setiap pohon (misalnya pemilihan subset data dan subset fitur), sehingga dengan menetapkan nilai random_state, hasil yang diperoleh akan selalu sama jika kode dijalankan ulang. Ini penting terutama saat melakukan evaluasi model atau perbandingan antar model.

### Algoritma Desicion Tree
Decision Tree Classifier adalah algoritma pembelajaran mesin yang berbasis pada struktur pohon untuk melakukan prediksi. Model ini bekerja dengan cara memecah data ke dalam simpul-simpul berdasarkan fitur yang paling informatif, membentuk cabang yang mengarah pada keputusan akhir. Setiap node dalam pohon memuat aturan berbasis fitur, dan proses pemisahan berlanjut hingga mencapai daun (leaf) yang berisi label kelas.

Proses pemisahan data dilakukan dengan tujuan memaksimalkan homogenitas kelas dalam setiap cabang. Untuk mengukur kualitas pemisahan ini, algoritma biasanya menggunakan metrik seperti Gini Impurity (bawaan) atau Entropy. Kelebihan Decision Tree terletak pada interpretabilitasnya yang tinggi, karena hasil model bisa divisualisasikan sebagai diagram pohon yang mudah dipahami.

Dalam implementasi proyek ini, digunakan kode berikut:
```# === Model Decision Tree ===
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
make_evaluation(y_test, y_pred_dt, title="Decision Tree")
```

Parameter yang digunakan adalah:
*   random_state=42: Parameter ini digunakan untuk menjamin bahwa hasil pelatihan model bisa direproduksi secara konsisten. Sama seperti pada Random Forest, nilai ini menetapkan seed acak yang digunakan oleh algoritma dalam proses pemecahan data dan pembentukan struktur pohon.

## Evaluation
Dalam proyek ini, metrik evaluasi yang digunakan untuk mengukur performa model adalah Confusion Matrix. Metrik ini dipilih karena sangat relevan untuk kasus klasifikasi multi-kelas seperti prediksi kategori risiko diabetes (Low Risk, Moderate Risk, dan High Risk). Confusion matrix memberikan informasi yang jauh lebih kaya dibandingkan hanya mengandalkan akurasi semata.

### Penjelasan Confusion Matrix
Confusion matrix merupakan sebuah tabel yang menggambarkan kinerja model klasifikasi dengan membandingkan label yang diprediksi oleh model dengan label sebenarnya. Dalam konteks proyek ini, setiap baris pada confusion matrix menunjukkan jumlah aktual (true labels), sedangkan setiap kolom menunjukkan jumlah prediksi yang dilakukan oleh model.
![662c42677529a0f4e97e4f96_644aea65cefe35380f198a5a_class_guide_cm08](https://github.com/user-attachments/assets/2cc01981-367c-4de4-bc0f-d8b6e859ae49)

Dengan menggunakan confusion matrix, kita dapat mengevaluasi beberapa aspek penting dari model:
*  True Positives (TP): Jumlah data yang benar diklasifikasikan ke dalam kelas yang tepat.
*  False Positives (FP): Jumlah data yang salah diklasifikasikan ke dalam suatu kelas, padahal seharusnya tidak termasuk kelas tersebut.
*  False Negatives (FN): Jumlah data yang seharusnya termasuk suatu kelas, tetapi diklasifikasikan ke kelas lain oleh model.
*  True Negatives (TN): Jumlah data yang benar tidak diklasifikasikan ke dalam kelas tertentu.

Selain memberikan gambaran umum terhadap kesalahan klasifikasi, confusion matrix juga menjadi dasar untuk menghitung metrik turunan lainnya, seperti:
*  Precision: Seberapa akurat model saat memprediksi suatu kelas.
*  Recall: Seberapa baik model dalam menemukan semua data yang termasuk dalam suatu kelas.
*  F1-score: Harmoni antara precision dan recall, terutama berguna saat data tidak seimbang antar kelas.

Penggunaan confusion matrix dalam proyek ini memungkinkan kita untuk:
*  Menilai apakah model bias terhadap kelas tertentu (misalnya hanya bagus di kelas Low Risk, tetapi lemah di High Risk),
*  Mengidentifikasi klasifikasi yang sering salah, dan
*  Memberikan dasar untuk penyesuaian parameter atau pemilihan model yang lebih optimal.

### Penerapan Model Random Forest

![image](https://github.com/user-attachments/assets/a1553646-1a69-42ed-9b9e-aa50eef8072c)

Hasil Training Model Random Forest
*   **Akurasi**: 85.4%
*   **Precision** tertinggi dimiliki oleh kategori low risk (1.00), yang artinya prediksi yang diberikan untuk low risk sangat tepat, meskipun jumlah data kategori ini kecil.
*   **Recall** rendah pada low risk (0.45) mengindikasikan banyak data low risk yang tidak terklasifikasi dengan benar — hanya 45% dari low risk yang berhasil dikenali oleh model.
*   Kinerja terbaik terlihat pada kategori moderate risk dan high risk, keduanya memiliki nilai **f1-score** 0.89.
*   Confusion matrix menunjukkan bahwa:
   *   Sebagian besar moderate risk dan low risk diklasifikasikan dengan benar.
   *   Terdapat kekeliruan dalam mengklasifikasikan high risk ke moderate risk (11 data).

### Penerapan Model Decision Tree
![image](https://github.com/user-attachments/assets/b92f7469-3589-4b4c-b6a7-033fb6c4b90b)

Hasil Training Model Random Forest
*  **Akurasi**: 78.6%
*  **Recall** untuk semua kelas lebih seimbang dibanding Random Forest (semuanya sekitar 0.77–0.83).
*  **Precision** terendah ada pada kategori low risk (0.60), mengindikasikan bahwa prediksi low risk sering salah.
*  **F1-score** tertinggi ada di moderate risk (0.83), menunjukkan keseimbangan yang baik antara precision dan recall.

*  Confusion matrix:
   *  Kategori low risk memiliki 24 kesalahan klasifikasi ke moderate risk.
   *  Beberapa moderate risk salah diklasifikasikan menjadi low atau high risk.
   *  Kategori high risk sering diklasifikasikan sebagai moderate risk.

### Perbandingan Model
![image](https://github.com/user-attachments/assets/c65679fe-3851-4fc4-90ba-38dc5b220dea)

Dari keempat metrik, Random Forest mengungguli Decision Tree dalam seluruh aspek. Hal ini menjadikan Random Forest sebagai model yang lebih baik secara keseluruhan.

Model Random Forest adalah model terbaik dalam studi ini karena mampu memberikan:
*   Akurasi yang tinggi
*   Precision yang sangat baik
*   Keseimbangan recall dan F1 score

## Menjawab Permasalahan
### 1. Bagaimana cara memanfaatkan informasi pola hidup guna memprediksi kategori diabetes pada seseorang?
![image](https://github.com/user-attachments/assets/5dff344e-14d5-4b4a-b31f-56215dccebe8)

Dengan menggunakan model machine learning yang telah kita buat, kita dapat menentukan kategori risiko diabetes seseorang berdasarkan informasi pola hidup yang dimasukkan. Model ini bekerja dengan memproses data seperti tinggi dan berat badan, kadar gula darah, durasi tidur, aktivitas fisik, pola diet, kepatuhan terhadap pengobatan, tingkat stres, dan status hidrasi.

Setelah pengguna memasukkan data tersebut, model akan menghitung nilai BMI dan menggabungkan seluruh variabel ke dalam sebuah struktur input yang sesuai. Kemudian, model akan mengklasifikasikan individu tersebut ke dalam salah satu dari tiga kategori risiko diabetes, yaitu:

*   Low Risk (Risiko Rendah),
*   Moderate Risk (Risiko Sedang), atau
*   High Risk (Risiko Tinggi).

### 2. Apa faktor-faktor yang paling berkontribusi terhadap peningkatan risiko diabetes?
![image](https://github.com/user-attachments/assets/432c617a-4760-4421-8332-902851954d14)

**Korelasi Fitur dengan Risiko Diabetes**
| **Fitur**              | **Korelasi** | **Interpretasi**                                                              |
|------------------------|--------------|-------------------------------------------------------------------------------|
| **bmi**                | +0.393       | Semakin tinggi BMI, semakin tinggi risiko diabetes.                          |
| **weight**             | +0.316       | Semakin besar berat badan, semakin tinggi risiko diabetes.                   |
| **diet**               | −0.346       | Pola diet sehat berkorelasi negatif, artinya mengurangi risiko diabetes.     |
| **physical_activity**  | −0.386       | Semakin banyak aktivitas fisik, semakin rendah risiko diabetes.              |
| **medication_adherence** | −0.496     | Kepatuhan terhadap pengobatan sangat menurunkan risiko diabetes.             |

Singkatnya, BMI dan berat badan memiliki korelasi positif dengan risiko diabetes, sementara diet sehat, aktivitas fisik, dan kepatuhan pengobatan memiliki korelasi negatif, yang berarti faktor-faktor ini dapat membantu mengurangi risiko diabetes.
