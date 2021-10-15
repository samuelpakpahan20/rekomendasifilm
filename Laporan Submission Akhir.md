# Laporan Proyek Machine Learning - Fahmi Jabbar

## Daftar Isi

-   [Project Overview](#project-overview)
-   [Business Understanding](#business-understanding)
-   [Data Understanding](#data-understanding)
-   [Data Preparation](#data-preparation)
-   [Modeling](#modeling)
-   [Evaluation](#evaluation)
-   [Referensi](#referensi)

## Project Overview
Semua situs web hiburan atau toko online memiliki jutaan/miliar item. Akan menjadi tantangan bagi pelanggan untuk memilih yang tepat. Pada proyek ini, kita akan membuat sistem rekomendasi yang dapat membantu pengguna menemukan item yang tepat dengan meminimalkan opsi.

Sistem Rekomendasi di dunia machine learning telah menjadi sangat populer dan merupakan keuntungan besar bagi perusahaan raksasa teknologi seperti Netflix, Amazon, dan banyak lagi untuk menargetkan konten mereka ke audiens tertentu. Mesin rekomendasi ini sangat kuat dalam prediksinya sehingga mereka dapat secara dinamis mengubah status apa yang dilihat pengguna di halaman mereka berdasarkan interaksi pengguna dengan aplikasi.

Kita ambil contoh [Netflix](https://www.netflixprize.com/rules.html) adalah aplikasi yang menghubungkan orang ke film yang mereka sukai. Untuk membantu pelanggan menemukan film tersebut, mereka mengembangkan sistem rekomendasi film kelas dunia bernama **CinematchSM**. Tugasnya adalah memprediksi apakah seseorang akan menikmati film berdasarkan seberapa besar mereka menyukai atau tidak menyukai film lain. Netflix menggunakan prediksi tersebut untuk membuat rekomendasi film pribadi berdasarkan selera unik setiap pelanggan. Dan sejauh ini Cinematch berjalan cukup baik. Sekarang ada banyak pendekatan alternatif menarik tentang cara kerja Cinematch yang belum dicoba Netflix. Beberapa dijelaskan dalam literatur, beberapa tidak. Pada proyek ini kita ingin mencoba beberapa pendekatan yang digunakan Netlix untuk membuat sebuah sistem rekomendasi.

[← Kembali ke Daftar Isi](#daftar-isi)

## Business Understanding

### Problem Statements

Setelah mengetahui beberapa masalah diatas, berikut ini merupakan rincian masalah yang perlu diselesaikan di proyek ini:

-   Sistem rekomendasi apa yang baik untuk diterapkan pada kasus ini?
-   Bagaimana cara membuat sistem rekomendasi film?

### Goals

Berikut adalah tujuan dari dibuatnya proyek ini:

-   Membuat sistem rekomendasi film.
-   Memberikan rekomendasi film yang kemungkinan disukai pengguna.

### Solution approach

Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini membuat sistem rekomendasi dengan algoritma _content based filtering_ karena sesuai dengan datasetnya. Sehingga sistem rekomendasi dibuat untuk memberikan rekomendasi pada pengguna terhadap film yang sebelumnya disukai/ditonton. 

Algoritma yang digunakan untuk membuat sistem rekomendasi di proyek ini adalah dengan **Cosine Similarity**. Algoritma ini dipilih karena mudah digunakan dan juga sebagai pembanding dengan sistem rekomendasi dengan model. _Cosine similarity_ singkatnya digunakan untuk mengukur kemiripan antara dua buah vektor dan kesamaan arahnya dengan cara menghitung sudut kosinus dari kedua vektornya. Cara menghitungnya adalah dengan rumus berikut ini :

![Rumus Cosine Similarity](https://user-images.githubusercontent.com/58651943/134554771-8f23cc13-ef84-4afa-b614-816b6daf65f6.png)

Dimana nilai x, y adalah nilai vektor dan k adalah nilai _cosine similarity_ dari vektor x dan y.

[← Kembali ke Daftar Isi](#daftar-isi)

## Data Understanding

![Sampul Dataset](https://raw.githubusercontent.com/samuelpakpahan20/rekomendasifilm/master/images/sampul.JPG)

Tabel dibawah ini merupakan informasi dari dataset yang digunakan :

| Jenis                   | Keterangan                                                                                    |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset : The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)  |
| Lisensi                 | CC0: Public Domain                                                                            |
| Kategori                | Movie dan acara TV                                                                            |
| Rating Penggunaan       | 8.2                                                                                           |
| Jenis dan Ukuran Berkas | zip (944 MB)                                                                                  |

Gambar dibawah ini merupakan pratinjau dari dataset pada berkas <code>movie_metadata.csv</code> :

![Pratinjau movie_metadata.csv](https://user-images.githubusercontent.com/58651943/134555896-96029376-fd93-4e90-ace0-24d028e5737d.png)

Berkas <code>movie_metadata.csv</code> berisi informasi mengenai detail sebuah film. Namun pada datanya masih terdapat banyak sekali nilai kosong seperti pada kolom <code>genres</code>, <code>cast</code>, <code>director</code>, dan <code>title</code>. Berikut ini adalah uraian variabel dari setiap kolom pada dataset :

1. Kolom <code>index</code> merupakan index baris.
2. Kolom <code>budget</code> merupakan harga menonton film.
3. Kolom <code>genres</code> merupakan jenis genre film.
4. Kolom <code>homepage</code> merupakan situs tempat streaming film.
5. Kolom <code>id</code> merupakan  id film.
6. Kolom <code>keywords</code> merupakan kata kunci film.
7. Kolom <code>original_language</code> merupakan bahasa yang digunakan pada film.
8. Kolom <code>original_title</code> merupakan bahasa pada judul.
9. Kolom <code>overview</code> merupakan sinopsis film.
10. Kolom <code>popularity</code> merupakan nilai popularitas film.
11. Kolom <code>production_companies</code> merupakan nama perusahaan yang memproduksi film.
12. Kolom <code>production_countries</code> merupakan tempat perusahaan yang memproduksi.
13. Kolom <code>release_date</code> merupakan tanggal film dirilis.
14. Kolom <code>runtime</code> merupakan durasi film.
15. Kolom <code>spoken_languages</code> merupakan bahasa yang digunakan pada film.
16. Kolom <code>status</code> merupakan nilai apakah sudah dirilis atau belum.
17. Kolom <code>title</code> merupakan judul film.
18. Kolom <code>vote_average</code> merupakan rata-rata nilai voting.
19. Kolom <code>vote_count</code> merupakan jumlah voting.
20. Kolom <code>cast</code> merupakan nama-nama pemain film.
21. Kolom <code>crew</code> merupakan nama-nama karyawan.
22. Kolom <code>director</code> merupakan nama direktur film.
23. Kolom <code>revenue</code> merupakan hasil pendapatan dari film.

## Data Preparation

Berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data :
-   Memilih kolom yang dijadikan fitur terbaik, karena kita tidak memakai semua kolomnya. Jadi, kita hanya pakai kolom `keywords`, `cast`, `genre`, `director`, dan `title` untuk digunakan sebagai kumpulan fitur terbaiknya.
-   Mengimputasi data kosong pada kolom-kolom yang kita jadikan fitur terbaik tadi menjadi string kosong. Hal ini dilakukan karena banyak sekali data pada kolom-kolom tersebut yang kosong dan apabila dihapus saja hal ini akan mengakibatkan model yang dibuat kehilangan banyak informasi untuk membangun sistem rekomendasi yang baik.
-   Menggabungkan tiap data pada kolom fitur terbaik, hal ini dilakukan agar kita dapat menghitung nilai matrix kesamaannya.

[← Kembali ke Daftar Isi](#daftar-isi)

## Modeling

Setelah dilakukan pra-pemrosesan data, selanjutnya adalah membuat sistem rekomendasi _content based filtering_.

### Dengan Cosine Similarity

Untuk menghitung _cosine similarity_ dari setiap data di dataset kita menggunakan fungsi [cosine_similarity](https://scikitlearn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) dari sklearn. Prosesnya adalah dengan memanggil fungsi `cosine_similarity` dengan argumen _dataframe_ sebagai objeknya. Kemudian hasil dari perhitungannya disimpan pada variabel baru. Untuk tahapan pemberian rekomendasinya, dibuat fungsi `get_title_from_index` dan `get_index_from_title` dimana fungsi tersebut akan memberikan rekomendasi terhadap suatu judul film dari indeks film dan sebaliknya.

Kemudian kita akan menemukan indeks dari film itu. 

Pada fungsi tersebut, akan dilakukan pencarian index dari suatu nama film pada _dataframe_ baru hasil perhitungan _cosine similarity_. Setelah itu, kita akan mengakses baris yang sesuai dengan film tersebut dengan matriks Similarity dan kemudian diurutkan nilainya berdasarkan nilai _cosine similarity_ tertinggi. Untuk lebih jelasnya hasil rekomendasi dapat dilihat seperti berikut ini :

![Rekomendasi Cosine Similarity](https://user-images.githubusercontent.com/58651943/134568493-043292d8-9002-4e2e-ac9d-7f534d4595a6.png)

[← Kembali ke Daftar Isi](#daftar-isi)

## Evaluation

Untuk mengukur kinerja model KNN untuk sistem rekomendasi digunakan beberapa metriks diantaranya :

1. Skor Calinski Harabasz

    Skor Calinski Harabasz digunakan untuk menghitung kriteria rasio varian. Metriks ini digunakan pada model clustering seperti yang saat ini sedang digunakan. Skor semakin tinggi ketika kluster padat dan terpisah dengan baik. Dikutip dari laman dokumentasi scikit-learn, Skor ini dihitung dengan formula [[4](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)] :

    ![Formula Calinski Harabasz - 1](https://user-images.githubusercontent.com/58651943/134569725-1f36ec4f-fe26-46f0-bbf2-ca06381c69a8.png)

    ![Formula Calinski Harabasz - 2](https://user-images.githubusercontent.com/58651943/134571888-be63b7ce-bcf7-4564-af28-4af9f12c20fb.png)

    Kelebihan dari metriks ini adalah :

    - Skornya tinggi apabila kluster padat dan terpisah dengan baik, yang mana bergantung pada konsep standar dari sebuah kluster.
    - Skornya cepat untuk dihitung.

    Sedangkan kekurangannya :

    - Metriks ini hanya baik digunakan pada kasus _convex cluster_.

    Penerapannya pada kode adalah dengan menggunakan fungsi [calinski_harabasz_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html) dari sklearn. Fungsi tersebut menerima argumen dari sebuah data yang digunakan untuk membuat model dan labelnya. Berikut ini adalah hasil penerapannya pada model KNN.

    ![Skor Calinski Harabasz](https://user-images.githubusercontent.com/58651943/134569489-c29aa25c-f56b-438c-8eb4-8c2b18769fd0.png)

    Pada model ini, nampaknya kluster masih belum padat dan terpisahkan dengan baik karena nilai skornya masih cukup rendah. Memungkinkan rekomendasi pada beberapa aplikasi masih terdapat rekomendasi yang tidak sesuai dengan aplikasi yang disukai pengguna.

2. Skor Davies Bouldin

    Skor Davies Bouldin digunakan untuk menilai separasi tiap kluster dari model. Metriks ini digunakan pada model clustering seperti yang saat ini sedang digunakan. Skor rendah ketika separasi tiap kluster di model terpisahkan dengan baik. Dikutip dari laman dokumentasi scikit-learn, Skor ini dihitung dengan formula [[4](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)] :

    ![Formula Davies Bouldin - 1](https://user-images.githubusercontent.com/58651943/134571953-0f466793-eb7d-4c69-8a8e-39f430074502.png)

    ![Formula Davies Bouldin - 2](https://user-images.githubusercontent.com/58651943/134572201-f518acd6-eb48-49ca-aea4-96d46f6483cb.png)

    Kelebihan dari metriks ini adalah :

    - Komputasinya lebih mudah daripada Skor Silhouette.
    - Skor yang dihitung hanya jumlah dan fitur yang melekat pada dataset.

    Kekurangan dari metriks ini adalah :

    - Metriks ini hanya baik digunakan pada kasus _convex cluster_.
    - Penggunaan jarak centroid membatasi metriks jarak ke ruang Euclidean

    ![Skor Davies Bouldin](https://user-images.githubusercontent.com/58651943/134569215-57cee0eb-ad46-4e08-a702-b4f7f65ae62f.png)

    Pada model ini skornya cukup kecil sehingga menandakan modelnya sudah memiliki separasi kluster yang baik. Hal ini dibuktikan juga dengan hasil rekomendasi aplikasi yang cukup baik dan sesuai kategorinya.

[← Kembali ke Daftar Isi](#daftar-isi)

# Referensi

[[1](https://irjet.com/archives/V8/i4/IRJET-V8I4745.pdf)] Sunasara, A. A., Jaiswal, N., Poojari, S., & Chaturvedi, A. K. (2021). _Play Store App Analysis_. International Research Journal of Engineering and Technology (IRJET).

[[2](https://journals.sagepub.com/doi/10.1155/2015/475163)] Choi, S.-M., Lee, H., Han, Y.-S., Man, K. L., & Chong, W. K. (2015). _A Recommendation Model Using the Bandwagon Effect for E-Marketing Purposes in IoT_. International Journal of Distributed Sensor Networks. https://doi.org/10.1155/2015/475163

[[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)] Harrison, O. (2019, July 14). _Machine Learning Basics with the K-Nearest Neighbors Algorithm_. Medium. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

[[4](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)] scikit-learn. (2021). Clustering - Performance Evaluation. https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

[← Kembali ke Daftar Isi](#daftar-isi)
