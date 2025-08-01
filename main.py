import os
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import re
nltk.download('punkt')


# 1. Làm sạch văn bản
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # bỏ số
    text = re.sub(r'[^\w\s]', '', text)  # bỏ dấu câu
    text = re.sub(r'\s+', ' ', text).strip()  # bỏ khoảng trắng thừa
    return text

# ----------------------------------
# 2. Đọc dữ liệu và tiền xử lý
with open("data/cleaned_articles.data", "r", encoding="utf-8") as f:
    raw_lines = [line.strip() for line in f if line.strip()]

corpus = [clean_text(line) for line in raw_lines]

print(f"✅ Số văn bản sau khi làm sạch: {len(corpus)}")

# ----------------------------------
# 3. TF-IDF
vietnamese_stopwords = [
    'và', 'là', 'của', 'có', 'cho', 'với', 'những', 'các', 'được',
    'trên', 'tại', 'một', 'này', 'đã', 'rằng', 'thì', 'lại', 'sẽ',
    'khi', 'đến', 'đi', 'ở', 'về', 'đó', 'nên', 'vì', 'nếu', 'tôi',
    'bạn', 'chúng', 'tôi', 'anh', 'chị', 'nó', 'họ', 'vẫn', 'đang'
]

vectorizer = TfidfVectorizer(
    stop_words=vietnamese_stopwords,
    token_pattern=r'(?u)\b[^\d\W]+\b'  # chỉ lấy từ chứa chữ cái
)

tfidf_matrix = vectorizer.fit_transform(corpus)
print(f"🔹 TF-IDF shape: {tfidf_matrix.shape}")
print("🔹 Một số từ khóa TF-IDF:", vectorizer.get_feature_names_out()[:10])

# ----------------------------------
# 4. Huấn luyện Word2Vec với underthesea

# Tách từ cho mỗi văn bản
tokenized_corpus = [word_tokenize(doc) for doc in corpus]

# Huấn luyện Word2Vec
w2v_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,  # dùng Skip-Gram
    compute_loss=True
)
w2v_model.build_vocab(tokenized_corpus)

loss_values = []
previous_loss = 0.0
epochs = 10

for epoch in range(1, epochs + 1):
    w2v_model.train(
        tokenized_corpus,
        total_examples=w2v_model.corpus_count,
        epochs=1,
        compute_loss=True
    )
    current_loss = w2v_model.get_latest_training_loss()
    epoch_loss = current_loss - previous_loss  # tính chênh lệch
    loss_values.append(epoch_loss)
    previous_loss = current_loss
    print(f"🔁 Epoch {epoch}: Loss = {epoch_loss:.2f}")
# Kiểm tra vector của một từ
sample_word = "khách"
if sample_word in w2v_model.wv:
    print(f"🔹 Vector từ '{sample_word}':\n", w2v_model.wv[sample_word])
else:
    print(f"⚠️ Từ '{sample_word}' không có trong từ điển Word2Vec.")

# ----------------------------------
# 5. Biểu diễn văn bản bằng trung bình vector từ

def document_vector(doc):
    words = word_tokenize(doc)
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

# Biểu diễn toàn bộ văn bản
doc_vectors = [document_vector(doc) for doc in corpus]

# Kiểm tra
print(f"\n🔸 Vector trung bình cho văn bản 1:\n{doc_vectors[0]}")
print(f"✅ Kích thước vector mỗi văn bản: {doc_vectors[0].shape}")

# GensimWord2Vec_model = Word2Vec(corpus,
#                                 vector_size=100,
#                                 min_count=1,  # số lần xuất hiện thấp nhất của mỗi từ vựng
#                                 window=2,  # khai báo kích thước windows size
#                                 sg=8,  # sg = 1 sử dụng mô hình skip-grams - sg=0 -> sử dụng CBOW
#                                 workers=1
#                                 )

# print('Tìm top-10 từ tương đồng với từ: [khách]')
# # for index, word_tuple in enumerate(GensimWord2Vec_model.wv.similar_by_word("khách")):
# #     print('%s.%s\t\t%s' % (index, word_tuple[0], word_tuple[1]))
# word = "khách"

# if word in GensimWord2Vec_model.wv:
#     for index, (similar_word, similarity) in enumerate(GensimWord2Vec_model.wv.similar_by_word(word), start=1):
#         print(f"{index}. {similar_word}\t\t{similarity:.4f}")
# else:
#     print(f"⚠️ Từ '{word}' không có trong từ điển của mô hình.")
print("\n📉 Tổng quan loss theo từng epoch:")
for i, loss in enumerate(loss_values, 1):
    print(f"Epoch {i}: Loss = {loss:.2f}")
print("\n📌 Top từ tương đồng với 'khách':")
if "khách" in w2v_model.wv:
    for index, (similar_word, similarity) in enumerate(w2v_model.wv.similar_by_word("khách", topn=10), start=1):
        print(f"{index}. {similar_word}\t\t{similarity:.4f}")
else:
    print(f"⚠️ Từ 'khách' không có trong từ điển.")