<div align="center">
T.C.
SAKARYA ÜNİVERSİTESİ
BİLGİSAYAR VE BİLİŞİM BİLİMLERİ FAKÜLTESİ

BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ




BSM 461 BÜYÜK VERİYE GİRİŞ




MOVIE RECOMMENDATION SYSTEM


Waasiq MASOOD
Demba SOW


Bölüm
Danışman	:
:	BİLGİSAYAR MÜHENDİSLİĞİ
Öğr. Gör. Dr. Sümeyye KAYNAK







2022-2023 Güz Dönemi
</div>

  
# 1. PROJENİN AMACI

Projemiz amacı büyük veriler teknolojiler ve kütüphaneler kullanarak bir film öneri sistem gerçekleşmektedir. Bunun için yeterince büyük bir veri set gerekiyor çünkü Model’e eğitmek için ve arasındaki fark görmek için ne kadar iyi veri varsa o kadar iyidir.

# 2. VERİ SETİ (DATASET)
	
Kaggle’den filmlerin veri seti alıp üzerinde veri işlemler yapılacaktır. Filmler veri setinde TMDB sitesinde 45,000 filmler olan bir veri seti mevcuttur fakat bizim aldığımız veri sette 5000 veri 20 sütün kapsamaktadır.

Kullandığımız Teknolojiler: 
1.	Python
2.	Scikit-learn
3.	Pandas
4.	NLTK (Natural Language Processing Library)
5.	Streamlit (Web için )

## 2.1. Veri Önişleme

Veriler önişleme kısımında önce verilere incelememiz lazım, boş satır ve sütün varsa o veriler yerinde farklı veriler koymamız lazım. Projeyi göre hangi veriler bizim için en önemli olduğunu bakmamız lazım ve ona göre sütün ve satır seçmemiz lazım. Veriler iki tane CSV dosyalarda bulunmaktadır:
1.	Movies.csv
Movies.csv içindeki sütunlar aşağıdaki verilere kapsıyor:
1.	Budget: Filminin bütçesi
2.	Genres: Filminin türleri
3.	Homepage: Filminin sitesi
4.	Id: TMDB sitesindeki Filminin ID
5.	Keywords: Filmi anlatılan kelimeler
6.	Original_language: Dili
7.	Original_title: Film yaptığı dilindeki başlık
8.	Overview: Filmi hakkında bilgi
9.	Popularity: Film ne kadar meşhur hakkında bilgi
10.	Production_companies: Filmi yapan firmalar
11.	Production_countries: Çektiği ülkeler
12.	Release_date: Çıktığı tarih
13.	Revenue: Kazandığı paranın miktarı
14.	Runtime: Filminin uzunluğu
15.	Spoken_languages: Konuştuğu diller
16.	Status: Çıkmış mı çıkmamış mı 
17.	Tagline: Film hakkında tagler
18.	Title: Filmi adı
19.	Vote_average: Verildiği puanların ortalaması
20.	Vote_count: Verildiği oylar
2.	Credits.csv
1.	Movie_id: Filmin ID
2.	Title: Filminin Adı
3.	Cast: Filmdeki oyuncular
4.	Crew: Filmi yapan kişiler 

### 2.1.1	 Önişleme için kullanılacak kutüpkhaneler
```bash
	import numpy as np
	import pandas as pd
	import ast as ast
	
	import nltk as nltk
	from nltk.stem.porter import PorterStemmer
```
### 2.1.2	 Veriler okuma ve kullanmayacağı sütünler

#* Veriler okumak için pandas dataframe kullanılmaktadır:
```bash
Movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
Credits = pd.read_csv('dataset/tmdb_5000_credits.csv')
```

#* Only keeping the important columns
```bash
md = Movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]
md.dropna(inplace=True)
```

* Veriler okuduktan sonra boş olan satırlar dataframe’den silenecektir. Ilk olan sütüne bakarsak böyle bir çıktı elde ediliyor
 
* Credit dosyasında veriler böyledir
		 

### 2.1.3	Veriler Keşfetme
 Verilerin dataframeleri keşfetmek için aşağıdaki kodu kullandık:
 
	
Farklı sütünlerdeki veriler keşfettikten sonra algoritma için kullanacağımız:
```bash
movies = movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]

```
 
* Filmler hakkında bilgi

### 2.1.4 Genre, Cast ve Keywords sütünlere değiştirme

Genre, cast ve Keywords sütünlerde veriler JSON türünde bulunmaktadır ve bunlarda sadece bazı bilgileri istenir. Bu bilgiler almak için aşağıda verilmiş kodlar yazılmaktadir:
Genre ve Cast  sütünler sadece isimler alınmaktadır:

```bash
def formatGenre(obj):
    List = []
    for item in ast.literal_eval(obj):
        List.append(item['name'])
    return List
		
			def formatCast(obj):
    List = []
    counter = 0
    for item in ast.literal_eval(obj):
        if counter != 6:
            List.append(item['name'])
            counter += 1
        else:
            break
    return List
 ```
Crew sütünden sadece Director alınmaktadır:
```bash
def formatCrew(obj):
    List = []
    for item in ast.literal_eval(obj):
        if item['job'] == 'Director':
            List.append(item['name'])
            break
    return List

```

Sonrasından sütünlerde listelere bir araya birşleşmek için Lamda fonksiyonlar kullanılmaktadır:


```bash
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
```
### 2.1.5 Tag sütün oluşmak
   

### 2.1.6 Stemming Algoritma Kullanmak

Aynı kökten oluşan kelimeler mesala loves, love, lover bir kelime olarak almak için stemmin algoritma kullanılmaktadır ve bu algoritma kelimeler azalıp bir kelime verilmektedir. 

Yukarıdaki örnek için love verir.

```bash
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_movies['tags'] = new_movies['tags'].apply(stem)
```
# 3. ALGORITMA KULLANMASI
Filmleri arasında bağlantılar bulmak için ilk önce kelimelere vector’e dönüştürmemiz lazım ve bunun için Bag of Words algoritması kullanılmaktadır.

#* Applying Text Vectorization (Bag of Words):

```bash
cv = CountVectorizer(max_features=5000, stop_words='english') 
vectorized_movies = cv.fit_transform(new_movies['tags']).toarray()
```

Vectorization’dan sonra cosine similarlity algoritma uygulayarak her bir film arasındaki vector fark anlayıp o filmi önerilecektir:
```bash	
   similarity = cosine_similarity(vectorized_movies)
```

# 4. FILMI ÖNERİLER
 
    
Filmi önermek için recommend fonksiyon kullanılmaktadır ve bir filmini index kullanararak similarlity listeden bulunmaktadır.

# 5. WEB SİTESİNDE UYGULAMA

Model kullanarak sitesinde film önermek için API kullanarak önerdiği filmler gösterilmektedir. Spiderman iki için böyle filmler önermektedir:


# 6. SONUÇ:
Bu projede, günümüzde çok yaygın olan öneri sistemlerde(Recommendation systems) kullanılan İçerik tabanlı filtreme “Content-based recommendation system” türünü implemente ettik.  Bu filtremenin avantajlarından biri, kullanıcıdan hiç bir bilgiye ihtıyacı duyulmamasıdır. Sadece kullanıcının ilgi alanlarına gerek duymaktadır.  Bu nedeniyle bu tür filtremenin en iyi sonuca elde edilmesini sağlamaktadır. 

# 7. BAĞLANTILAR:

7.1 [Proje Kaggle Dataset Linki](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

7.2 [Proje Frontend Linki](https://github.com/waasiq?tab=repositories)

7.3 [Proje Web Arayüz Linki](https://github.com/waasiq?tab=repositories)


