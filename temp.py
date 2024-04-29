import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

# Kelimelerin köklerini belirlemek için gereken fonksiyon
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return None

# İşlenmiş verileri içeren CSV dosyasını oku
data = pd.read_csv(r"C:\Users\merye\OneDrive\Masaüstü\derin öğrenme\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

# Duygu değerlerini hesaplamak için gereken fonksiyon
def sentiment_score(rating):
    if rating > 3:
        return 1  # Olumlu
    elif rating < 3:
        return -1  # Olumsuz
    else:
        return 0  # Tarafsız

# İşlenmiş yorumları saklamak için bir liste oluştur
processed_reviews = []

# Stopword'lerin listesini oluştur
stop_words = set(stopwords.words('english'))

# Lemmatizer oluştur
lemmatizer = WordNetLemmatizer()

# Yorumları önişleme
for index, row in data.iterrows():
    if isinstance(row['reviews.text'], str):  # Metin verisi olduğundan emin ol
        review = row['reviews.text']
        # Belirteçleme
        tokens = nltk.word_tokenize(review)
        # Kök belirleme ve etiketleme (lemmatization ve pos tagging)
        tagged_tokens = nltk.pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_tokens if get_wordnet_pos(tag)]
        # Stopword'leri çıkar
        filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
        # Tüm harfleri küçük harfe dönüştür
        lowercased_tokens = [word.lower() for word in filtered_tokens]
        processed_reviews.append(" ".join(lowercased_tokens))

# Duygu sütununu oluştur
data['sentiment'] = data['reviews.rating'].apply(sentiment_score)

# İşlenmiş verileri yeni bir CSV dosyasına kaydet
data.to_csv('processed_data_with_sentiment.csv', index=False)

# Duygu değerlerine göre gruplayarak sayıları hesapla
positive_count = (data['sentiment'] == 1).sum()
negative_count = (data['sentiment'] == -1).sum()
neutral_count = (data['sentiment'] == 0).sum()

# Tabloyu oluştur
sentiment_table = pd.DataFrame({'Veri(Duygu)': ['+1 (Olumlu)', '-1 (Olumsuz)', '0 (Tarafsız)'],
                                'Kayıt Sayısı': [positive_count, negative_count, neutral_count]})

# Sonuçları ekrana yazdır
print("Tablo 1. Duygu Değerlendirmeleri Kayıt Sayıları")
print(sentiment_table)

