import pandas as pd
import re
from underthesea import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

# Chọn chế độ chạy:
mode = "full_evaluation"  # predict_only hoặc "full_evaluation"

# Danh sách từ dừng tiếng Việt
stop_words = set(['là', 'của', 'và', 'có', 'trong', 'được', 'tại', 'cho', 'với', 'trên', 'từ', 'để'])

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', '', text)
    tokens = word_tokenize(text, format="text").split()
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Đọc dữ liệu
data = pd.read_excel("DataWithLabel.xlsx")
sentences = data['sentence'].tolist()
labels = data['label'].tolist()

# Tạo TaggedDocument cho Doc2Vec
tagged_data = [TaggedDocument(words=preprocess_text(s), tags=[str(i)]) for i, s in enumerate(sentences)]

# Huấn luyện mô hình Doc2Vec
doc2vec_model = Doc2Vec(vector_size=100, min_count=2, epochs=40)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

print("\n====== VECTOR BIỂU DIỄN BỞI DOC2VEC ======\n")
for i in range(len(sentences)):
    vector = doc2vec_model.dv[i]
    print(f"Văn bản {i} ({labels[i]}):")
    print(vector, "\n")


# Biểu diễn văn bản
X = [doc2vec_model.infer_vector(preprocess_text(s)) for s in sentences]
y = labels

# Chia tập huấn luyện & kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Các bộ phân loại (3 thuật toán phân loại)
classifiers = {
    "Rocchio": NearestCentroid(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

# Kết quả đánh giá
results = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if mode == "predict_only":
        print(f"\n--- {name} ---")
        # print("10 kết quả dự đoán đầu tiên:")
        # print("Dự đoán :", y_pred[:10])
        # print("Thực tế :", y_test[:10])
        ##chạy full kết quả
        print("Tất cả kết quả dự đoán:")
        print("Dự đoán :", y_pred.tolist())
        print("Thực tế :", y_test)
    else:
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }

        print(f"\nBáo cáo phân loại cho {name}:")
        print(classification_report(y_test, y_pred, zero_division=0))

# Tóm tắt nếu chọn đánh giá
if mode == "full_evaluation":
    print("\nTóm tắt các độ đo đánh giá:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")

# Lưu mô hình Doc2Vec
doc2vec_model.save("doc2vec_model_vn")
#load lại model dùng:
# from gensim.models import Doc2Vec
# model = Doc2Vec.load("doc2vec_model_vn")
