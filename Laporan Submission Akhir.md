# Laporan Proyek Machine Learning - Samuel Partogi Pakpahan

## Project Overview
Semua situs web hiburan atau toko online memiliki jutaan/miliar item. Akan menjadi tantangan bagi pelanggan untuk memilih yang tepat. Pada proyek ini, kita akan membuat sistem rekomendasi yang dapat membantu pengguna menemukan item yang tepat dengan meminimalkan opsi.

Sistem Rekomendasi di dunia machine learning telah menjadi sangat populer dan merupakan keuntungan besar bagi perusahaan raksasa teknologi seperti Netflix, Amazon, dan banyak lagi untuk menargetkan konten mereka ke audiens tertentu. Mesin rekomendasi ini sangat kuat dalam prediksinya sehingga mereka dapat secara dinamis mengubah status apa yang dilihat pengguna di halaman mereka berdasarkan interaksi pengguna dengan aplikasi.

Kita ambil contoh [Netflix](https://www.netflixprize.com/rules.html) adalah aplikasi yang menghubungkan orang ke film yang mereka sukai. Untuk membantu pelanggan menemukan film tersebut, mereka mengembangkan sistem rekomendasi film kelas dunia bernama **CinematchSM**. Tugasnya adalah memprediksi apakah seseorang akan menikmati film berdasarkan seberapa besar mereka menyukai atau tidak menyukai film lain. Netflix menggunakan prediksi tersebut untuk membuat rekomendasi film pribadi berdasarkan selera unik setiap pelanggan. Dan sejauh ini Cinematch berjalan cukup baik. Sekarang ada banyak pendekatan alternatif menarik tentang cara kerja Cinematch yang belum dicoba Netflix. Beberapa dijelaskan dalam literatur, beberapa tidak. Pada proyek ini kita ingin mencoba beberapa pendekatan yang digunakan Netlix untuk membuat sebuah sistem rekomendasi.

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

![Rumus Cosine Similarity](https://raw.githubusercontent.com/samuelpakpahan20/rekomendasifilm/master/images/rumuscs.png)

Dimana nilai x, y adalah nilai vektor dan k adalah nilai _cosine similarity_ dari vektor x dan y.

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

![Pratinjau movie_metadata.csv](https://raw.githubusercontent.com/samuelpakpahan20/rekomendasifilm/master/images/pratinjau.JPG)

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

## Modeling

Setelah dilakukan pra-pemrosesan data, selanjutnya adalah membuat sistem rekomendasi _content based filtering_.

### Dengan Cosine Similarity

Untuk menghitung _cosine similarity_ dari setiap data di dataset, kita menggunakan fungsi [cosine_similarity](https://scikitlearn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) dari sklearn. Prosesnya adalah dengan memanggil fungsi `cosine_similarity` dan objek _dataframe_ sebagai argumen/parameternya. Kemudian hasil dari perhitungannya disimpan pada variabel baru. Untuk menggunakan fungsi tersebut, masukkan kode berikut:

```
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)
```

Setelah kita mendapatkan hasil perhitungan _cosine similarity_ langkah kita selanjutnya adalah membuat fungsi `get_title_from_index` dan `get_index_from_title` yang akan mendapatkan judul film yang saat ini disukai pengguna. Kemudian kita akan menemukan indeks dari judul film itu dan sebaliknya. Setelah itu, kita akan mengakses baris yang sesuai dengan film ini dengan matriks Similarity. Masukkan kode berikut.

```
# Definisikan dua fungsi untuk mendapatkan judul film dari indeks film dan sebaliknya
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
```

Setelah kita akan mendapatkan skor kesamaan semua film lain dari film pengguna. Lalu kita akan menghitung semua skor kesamaan film itu untuk membuat tupel indeks film dan skor kesamaan. Gunakan kode berikut.

```
movie_user_likes = "Star Trek Beyond"
movie_index = get_index_from_title(movie_user_likes)
# mengakses baris yang sesuai dengan film yang diinput untuk menemukan semua skor kesamaan untuk film itu 
# dan kemudian menghitungnya.
similar_movies = list(enumerate(cosine_sim[movie_index]))
```

Kemudian urutkan nilainya berdasarkan nilai _cosine similarity_ tertinggi (_Descending_), tuliskan kode ini.

```
# Mengurutkan daftar film dari skor kesamaan terbesar.
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
```

Gunakan kode berikut untuk melihat 10 list film pertama yang direkomendasikan.

```
# Membuat Perulangan untuk mencetak 10 list pertama dari daftar film
i=0
print("Top 10 Film yang mirip dengan "+movie_user_likes+" adalah:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>10:
        break
```

Hasilnya seperti berikut.

![Rekomendasi Cosine Similarity](https://raw.githubusercontent.com/samuelpakpahan20/rekomendasifilm/master/images/hasilrekomendasi.JPG)

Dari output tersebut, kita dapat membandingkan antara film inputan user **Star Trek Beyond** dan Top 10 film recommendation untuk user. 

## Evaluation

Seperti yang dijelaskan sebelumnya untuk membuat sistem rekomendasi ini digunakan metriks **Cosine Similarity** yang mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai _cosine similarity_.

![Vektor Cosine Similarity](https://raw.githubusercontent.com/samuelpakpahan20/rekomendasifilm/master/images/sudutcs.jpeg)

Metrik ini sering digunakan untuk mengukur kesamaan dokumen dalam analisis teks. Sebagai contoh, dalam studi kasus ini, _cosine similarity_ digunakan untuk mengukur kesamaan film.

_Cosine similarity_ dirumuskan sebagai berikut.

![Formula Cosine Similarity](https://raw.githubusercontent.com/samuelpakpahan20/rekomendasifilm/master/images/formulacs.JPG)

Cosine similarity pada Python menghitung kesamaan sebagai dot product yang dinormalisasi dari masukan sampel X dan Y. Penerapannya pada kode adalah dengan menggunakan fungsi [cosine_similarity](https://scikitlearn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) dari sklearn untuk mendapatkan nilai cosinus dua vektor dalam matriks. Berikut ini adalah hasil penerapannya pada model _content based filtering_.

*ps: Untuk kode lengkapnya dapat dilihat pada Tab [Modeling](#modeling)*

![Kode CS]()

Dengan menerapkan definisi kesamaan, ini sebenarnya akan sama dengan 1 jika kedua vektor identik, dan akan menjadi 0 jika keduanya ortogonal. Dengan kata lain, kesamaan adalah angka yang dibatasi antara 0 dan 1 yang memberi tahu kita seberapa mirip kedua vektor tersebut.

Dilihat dari output hasil rekomendasi pada Tab [Modeling](#modeling) tadi, coba kita bandingkan dengan mencari di Google untuk film yang mirip dengan **Star Trek Beyond** dan inilah hasilnya.

![Rekomendasi Google](https://raw.githubusercontent.com/samuelpakpahan20/rekomendasifilm/master/images/test.PNG)

> **Ini adalah bagian akhir laporan**
