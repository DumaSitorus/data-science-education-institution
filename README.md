# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan berhasil meluluskan banyak mahasiswa dengan reputasi yang baik. Namun, mereka menghadapi tantangan serius berupa tingginya angka mahasiswa yang dropout atau tidak menyelesaikan pendidikannya.

Permasalahan ini berdampak pada citra dan kualitas institusi, sehingga perlu diambil langkah preventif. Salah satu upaya yang dapat dilakukan adalah mendeteksi secara dini mahasiswa yang berpotensi mengalami dropout, agar institusi dapat memberikan bimbingan atau intervensi yang sesuai.

Untuk itu, Jaya Jaya Institut ingin memanfaatkan data mahasiswa untuk melakukan analisis dan prediksi menggunakan metode machine learning. Hasil prediksi ini diharapkan dapat membantu dalam menyusun strategi pembinaan yang lebih efektif. Selain itu, mereka juga membutuhkan sebuah dashboard visualisasi untuk memantau performa mahasiswa secara berkala.

### Permasalahan Bisnis
Permasalahan bisnis yang akan diselesaikan dalam proyek ini adalah sebagai berikut.
1. Tingginya angka dropout yang berdampak negatif pada reputasi dan kualitas institusi.
2. Belum adanya pemahaman yang jelas terkait faktor-faktor utama penyebab dropout.
3. Belum tersedianya alat yang dapat digunakan untuk melakukan prediksi mahasiswa yang beresiko mengalami dropout.
4. Tidak tersedianya dashboard analitik untuk memantau kondisi mahasiswa dan indikator yang berhubungan dengan potensi dropout.

### Cakupan Proyek
Tuliskan cakupan proyek yang akan dikerjakan.
Proyek ini mencakup beberapa tahap utama yaitu sebagai berikut.

1. Data Understanding dan Preprocessing: Memahami struktur dan kualitas data student preformance Institusi Jaya Jaya. Setelah memahami dan mengidentifikasi kecacatan dalam data selanjutnya dilakukan pembersihan data, seleksi dan transformasi fitur.
2. Modeling: Mengembangkan model prediktif untuk mengidentifikasi faktor-faktor utama yang mempengaruhi angka mahasiswa dropout.
3. Pembuatan Business Dashboard: Mendesain dan membangun dashboard yang dapat menampilkan indikator dropout secara visual, serta mendukung pengambilan keputusan pihak Jaya jaya institution berdasarkan data yang ada.
4. Insight dan Rekomendasi: Memperoleh insight berbasis data dan memberikan rekomendasi konkret (recommendation action item) untuk menurunkan tingkat dropout.

### Persiapan

Sumber data: Dataset yang digunakan pada proyek ini merupakan data mahasiswa perguruan tinggi Jaya Jaya Institut yang dapat diakses melalui [Data Students Jaya Jaya Institut]("https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance"). 

Kumpulan dataset Students Performance diperoleh dari lembaga pendidikan tinggi (diperoleh dari beberapa basis data terpisah) yang terkait dengan mahasiswa yang terdaftar dalam berbagai gelar sarjana, seperti agronomi, desain, pendidikan, keperawatan, jurnalisme, manajemen, layanan sosial, dan teknologi. Kumpulan data tersebut mencakup informasi yang diketahui pada saat pendaftaran mahasiswa (jalur akademik, demografi, dan faktor sosial-ekonomi) dan kinerja akademik mahasiswa pada akhir semester pertama dan kedua.

Dataset Students Performance terdiri dari 37 fitur dan 4424 entry data. Penjelasan fitur yang terdapat pada dataset ini termasuk:
1. `Marital_status` : Status pernikahan mahasiswa
2. `Application_mode` : Metode aplikasi yang digunakan oleh mahasiswa 
3. `Application_order` : Urutan Pendaftaran mahasiswa
4. `Course` : Jurusan yang diambil oleh mahasiswa
5. `Daytime_evening_attendance` :  Waktu kelas yang dihadiri 
6. `Previous_qualification` :  Kualifikasi yang diperoleh oleh mahasiswa sebelum mendaftar di pendidikan tinggi
7. `revious_qualification_grade` : Nilai kualifikasi sebelumnya
8. `Nacionality` : Kewarganegaraan mahasiswa
9. `Mothers_qualification` :  Kualifikasi ibu mahasiswa
10. `Fathers_qualification` : Kualifikasi ayah mahasiswa
11. `Mothers_occupation` : Pekerjaan ibu siswa
12. `Fathers_occupation` : Pekerjaan ayah siswa
13. `Admission_grade` : Nilai penerimaan
14. `Displaced` : Menunjukkan apakah mahasiswa tersebut orang terlantar atau tidak
15. `Educational_special_needs` : menunjukkan apakah mahasiswa memiliki kebutuhan pendidikan khusus
16. `Debtor` : Menunjukkan apakah siswa adalah debitur
17. `Tuition_fees_up_to_date` : Menunjukkan apakah biaya kuliah menunggak
18. `Gender` : Jenis kelamin
19. `Scholarship_holder` : Menunjukkan apakah mahasiswa mendapatkan beasiswa
20. `Age_at_enrollment`: Usia mahasiswa ketika mendaftar.
21. `International` :  Menunjukkan apakah mahasiswa merupakan international student atau bukan.
22. `Curricular_units_1st_sem_credited` : Jumlah satuan kurikulum yang dikreditkan oleh mahasiswa pada semester pertama
23. `Curricular_units_1st_sem_enrolled` :  Satuan kurikulum yang didaftarkan oleh mahasiswa pada semester pertama
24. `Curricular_units_1st_sem_evaluations` : Jumlah evaluasi terhadap satuan kurikulum di semester pertama
25. `Curricular_units_1st_sem_approved` :  Jumlah satuan kurikulum yang disetujui oleh mahasiswa pada semester pertama.
26. `Curricular_units_1st_sem_grade` :  Nilai yang diiperolah pada semester pertama
27. `Curricular_units_1st_sem_without_evaluations` : Jumlah satuan kurikulum tanpa evaluasi pada semester pertama
28. `Curricular_units_2nd_sem_credited`: Jumlah satuan kurikulum yang dikreditkan pada semester kedua
29. `Curricular_units_2nd_sem_enrolled` : Jumlah satuan kurikulum yang didaftarkan oleh mahasiswa pada semester kedua
30. `Curricular_units_2nd_sem_evaluations` : Jumlah evaluasi terhadap satuan kurikulum di semester pertama
31. `Curricular_units_2nd_sem_approved` : Satuan kurikulum yang disetujui oleh mahasiswa pada semester kedua.
32. `Curricular_units_2nd_sem_grade` : Nilai yang diiperolah pada semester kedua
33. `Curricular_units_2nd_sem_without_evaluations` :  Jumlah satuan kurikulum tanpa evaluasi pada semester kedua
34. `Unemployment_rate` : Tingkat pengangguran
35. `Inflation_rate` : Tingkat inflasi
36. `GDP` : Indikator ekonomi yang mempengaruhi kesuksesan akademik dan kemungkinan siswa mengalami dropout
37. `Status` : Target: berupa status mahasiswa saat ini

**Setup environment:**
Berikut langkah-langkah untuk mempersiapkan environment:
1. Menjalankan Notebook
- Jika menggunakan Anaconda, jalankan perintah berikut:
    ```
    conda create --name main-ds python=3.9
    conda activate main-ds
    pip install -r requirements.txt
    ```
- Jika menggunakan Shell/Terminal, jalankan perintah berikut:
    ```
    pip install pipenv
    pipenv install
    pipenv shell
    pip install -r requirements.txt
    ```
    
-  Jalankan file `notebook.ipynb` untuk melihat seluruh hasil analisis data beserta insight yang diperoleh dari data. Anda dapat menggunakan Jupyter notebook atau IDE lokal lainnya, atau anda dapat memanfaatkan IDE lain seperti Google Collab jika ingin menjalanakan proyek secara online.

2. Menjalankan Streamlit
- Akses melalui link berikut: https://data-science-education-institution-xtwef9ezhpn8iwmwmxvzpl.streamlit.app/
    
- Jika ingin menjalankan di lokal, jalankan perintah : 
    ```
    streamlit run app.py
    ```

3. Menjalankan Dashboard
- Jalankan perintah berikut pada Terminal/Command Prompt/PowerShell guna memanggil (pull) Docker image untuk menjalankan Metabase.
    ```
    docker pull metabase/metabase:v0.46.4
    ```

- Apabila proses pembuatan docker image telah selesai, Anda dapat menjalankan image tersebut menggunakan perintah berikut.
    ```
    docker run -p 3000:3000 --name metabase metabase/metabase
    ```

- Login ke metabase menggunakan username dan password berikut
    ```
    username: root@gmail.com
    password: root123
    ```

## Business Dashboard

Jaya Jaya Institution Dashboard dirancang untuk membantu institusi dalam melakukan analisis serta monitoring terhadap data mahasiswa yang berhubungan dengan resiko dropout.

Dashboard ini menampilkan:
1. Ringkasan jumlah mahasiswa lulus sebanyak 2.209 orang, Dropout sebanyak 1.421 orang, mahasiswa aktif sebanyak 794 orang, dan hasil prediksi model menunjukkan ada 348 mahasiswa aktif yang berpotensi dropout.
2. Dashboard juga menampilkan 15 fitur penting yang paling berpengaruh terhadap hasil prediksi model.
3. Pie chart perbandingan persentase mahasiswa graduate dan dropout
3. Diagram yang menampilkan distribusi jumlah mahasiswa drop out dan graduate berdasarkan penerimaan beasiswa, pembayaran uang kuliah tepat waktu atau menunggak, average grade pada semester 1 dan 2, jumlah SKS atau Approval rate, serta program studi.
4. Hasil Prediksi data mahasiswa aktif alias mahasiswa yang mengemban pendidikan saat ini di Jaya Jaya Institut. Hasil data ini kemudian dianalisis menggunakan cara yang sama dengan cara sebelumnya.

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning saya dijalankan menggunakan Streamlit, sebuah framework Python untuk membangun antarmuka web interaktif. Untuk menjalankan sistem, pengguna cukup mengakses link berikut:
🌐https://data-science-education-institution-xtwef9ezhpn8iwmwmxvzpl.streamlit.app/🌐

Di halaman tersebut, pengguna dapat menginput data mahasiswa sesuai dengan field yang teredia pada antar muka.
Setelah mengisi seluruh data dengan benar, anda dapat menekan tombol Predict untuk menampilkan hasil prediksi dari mahasiswa tersebut.
Dan sistem akan menampilkan hasil prediksi berupa status risiko (dropout atau graduate) serta probabilitas risikonya.
Sistem ini menggunakan model Logistic Regression yang telah dilatih sebelumnya dengan akurasi 91%.
Selain itu anda juga dapat menjalankan proyek ini di kompputer lokal anda dengan menjalankan perintah berikut.

```
streamlit run app.py
```

## Conclusion

Pada proyek ini telah dilakukan analisis data untuk menentukan faktor-faktor apa saja yang mempengaruhi angka Dropout mahasiswa pada Jaya Jaya Instition. Selain itu dilakukan pelatihan model yang mampu  melakukan prediksi apakah mahasiswa akan mengalami dropout(keluar) ataupun graduated(lulus) berdasarkan data akademik, demografis dan sosial, serta Ekonomi/Keaungan.

**1. Faktor-Faktor yang Mempengaruhi Dropout maupun Graduate**

- `Approval_rate`
  `Approval_rate` memiliki pengaruh terhadap hasil predikisi yang menunjukkan semakin tinggi nilai dari `Approval_rate`(persentase jumlah satuan unit berhasil diselesaikan terhadap jumlah satuan yang dimasuki) mahasiswa maka semakin kecil kemungkinan untuk mengalami dropout.

- `Course`
  Program studi yang diambil mahasiswa jug aberpengaruh terhadap resiko dropout. Beberapa program studi meungkin memiliki tingkat kesulitan dan beban akademik yang lebih tinggi, sehingga meningkatkan kemungkinan dropout. Program studi dengan jumlah dropout terbanyak antara lain Manajemen,nursing, Advertising and Marketing Management, dan Informatics Engineering.

- `Tuition_fees_up_to_date`
  Ini menunjukkan hubungan antara pembayawan biaya kuliah yang tepat waktu dengan status mahasiswa. mahasiswa yang menunggak pembayaran cenderung lebih rentan mengalamai kesulitan administratif atau keonomi yang dapat menyebabkan dropout.

- `Scholarship_holder`
  Mahasiswa penerima beasiswa cenderung memiliki komitmen akademik yang tinggi dan bantuan finansial, yang mengurangi tekanan ekonomi dan meningkatkan peluang kelulusan. Mahasiswa dengan beasiswa umumnya menunjukkan performa akademik yang baik dan memiliki dukungan ekonomi yang memadai.

- `Average_grade`

  Nilai rata-rata akademik merupakan indikator langsung dari kinerja belajar mahasiswa.
  Mahasiswa dengan nilai rata-rata tinggi memiliki kecenderungan lebih besar untuk lulus tepat waktu, sedangkan nilai rendah bisa menjadi sinyal risiko dropout.

- Dan beberapa fitur lain berdasarkan important feature yaitu  `Debtor`(penanggung utang), `Age_at_enrollment`(usia), `Mather_occupation`(pekerjaan ibu), `applicatioon model`(cara mendaftar perguruan tinggi), `Total_approve_curricular_units`(total unit semester yang disetujui)

**2. Pelatihan Model Prediksi**
  Model terbaik yang digunakan dalam proyek ini adalah Logistic Regression, dengan hasil evaluasi berikut:

*   Accuracy: 0.913223
*   Precision: 0.907173
*   Recall: 0.957684
*   F1-Score: 0.931744

Model Logistic Regression merupakan model terbaik dengan akurasi tertinggi dibandingkan dengan model lain seperti KNN dan SVM. Model Logistic Regression memiliki performa model yang seimbang secara keseluruhan dimana akurasi, presisi, dan recall cukup bagus.

### Rekomendasi Action Items

Berikut beberapa rekomendasi action items yang dapat dilakukan Institut guna menyelesaikan permasalahan atau mencapai target mereka untuk menurunkan angka dropout dan meningkatkan keberhasilan akademik mahasiswa.

1. Monitoring dan Intervensi Dini Berdasarkan Approval Rate

  Mahasiswa dengan nilai approval rate rendah dapat dipantau secara berkala. Institusi dapat membentuk sistem peringatan dini untuk mahasiswa yang terdeteksi dengan tingkat penyelesaian mata kuliah yang rendah, melalui Sesi konseling akademik, Pemberian tutor sebaya atau kelas remedial.

2. Penyesuaian Kurikulum pada Program Studi dengan Dropout Tinggi

  Fitur Course memiliki pengaruh tinggi terhadap tingkat dropout, sehingga perlu dilakukan evaluasi lebih lanjut pada program-program dengan dropout tinggi seperti Manajemen, Keperawatan, dan Teknik Informatika. Solusinya baik dengan melakukan peninjauan kembali beban akademik dan beban kerja mata kuliah, Perkuat pendekatan pembelajaran aktif atau berbasis proyek, Adakan pelatihan untuk dosen dalam metode pengajaran yang lebih inklusif dan suportif.

3. Kebijakan Finansial yang Lebih Fleksibel

  Mahasiswa yang menunggak pembayaran biaya kuliah cenderung lebih berisiko mengalami dropout. Maka, pihak institusi bisa mempertimbangkan untuk menyediakan opsi cicilan biaya kuliah, memberikan tenggang waktu tambahan bagi mahasiswa yang memiliki kendala keuangan atau melakukan pendekatan personal sebelum memberikan sanksi akademik karena tunggakan.

4. Peningkatan Akses terhadap Beasiswa

  Penerima beasiswa memiliki peluang kelulusan yang lebih tinggi, institusi dapat mempertimbangkan untuk menambah kuota dan jenis beasiswa (berbasis akademik, sosial-ekonomi, atau prestasi non-akademik). Lalu dapat juga memberikan sosialisasi yang lebih masif tentang cara mengakses beasiswa. Selain itu dapat juga menyediakan beasiswa darurat bagi mahasiswa terdampak kondisi ekonomi mendadak.

5. Program Penguatan Akademik untuk Mahasiswa dengan Nilai Rata-Rata Rendah

  Institusi dapat mempertimbangkan untuk mengadakan sesi bimbingan tambahan secara berkala untuk menginkatkan nilai rata-rata. Selain itu institut dapat mengusung pembagunan komunitas belajar atau kelompok belajar sebagai dukungan sosial dan akademik bekerja sama dengan dosen dan mahasiswa.

6. Penggunaan Model Machine Learning untuk Membantu melakukan Pemantauan serta Prediksi Dini

  Hasil prediksi dapat menjadi landasan untuk mengambil tindakan solutif untuk permasalahan maraknya angka dropout. Selain itu Dapat juga dipertimbangkan untuk pembuatan dashboard bagi pihak institusi untuk memantau mahasiswa dengan resiko tinggi mengalami dropout.