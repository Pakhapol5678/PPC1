"""
Machine Learning Module: Thai Review Sentiment Analysis & Aspect Extraction
Performs sentiment classification (positive/neutral/negative) and aspect analysis.
"""
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Try importing PyThaiNLP, fallback to regex/n-gram if not available
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    HAS_PYTHAINLP = True
    STOPWORDS = list(thai_stopwords())
except ImportError:
    HAS_PYTHAINLP = False
    STOPWORDS = []

def thai_tokenizer(text):
    if not isinstance(text, str):
        return []
    # Clean non-alphanumeric except Thai characters and spaces
    cleaned = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', ' ', text)
    if HAS_PYTHAINLP:
        tokens = word_tokenize(cleaned, engine='newmm', keep_whitespace=False)
        return [t.strip() for t in tokens if len(t.strip()) > 1 and t.strip() not in STOPWORDS]
    else:
        # Fallback 3-gram character tokenization for Thai text
        tokens = re.findall(r'[\u0E00-\u0E7F]+|[a-zA-Z0-9]+', cleaned)
        sub_tokens = []
        for word in tokens:
            if len(word) <= 4:
                sub_tokens.append(word)
            else:
                for i in range(len(word) - 2):
                    sub_tokens.append(word[i:i+3])
        return sub_tokens

# Aspect categories mapping
ASPECT_KEYWORDS = {
    "ที่จอดรถและการเดินทาง": ["ที่จอดรถ", "จอดรถ", "การจราจร", "รถติด", "ทางเข้า", "โบกรถ", "วนหา"],
    "อาหารและของกิน": ["ของกิน", "อาหาร", "อร่อย", "ขนม", "กาแฟ", "เครื่องดื่ม", "ของฝาก", "ร้านค้า"],
    "บรรยากาศและความสวยงาม": ["สวยงาม", "บรรยากาศ", "ร่มรื่น", "วิว", "ถ่ายรูป", "สงบ", "ธรรมชาติ", "มุม"],
    "ความสะอาดและสิ่งอำนวยความสะดวก": ["ห้องน้ำ", "สะอาด", "เหม็น", "ทรุดโทรม", "ทางเดิน", "วีลแชร์"],
    "ราคาและความคุ้มค่า": ["ราคา", "แพง", "ไม่แพง", "คุ้มค่า", "ค่าเข้า", "ค่าบริการ"],
    "การบริการและความแออัด": ["บริการ", "พนักงาน", "พูดจา", "คนเยอะ", "รอนาน", "ต่อคิว", "แออัด"]
}

class ThaiSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=thai_tokenizer,
            token_pattern=None,
            min_df=1,
            ngram_range=(1, 2)
        )
        self.classifier = LogisticRegression(max_iter=500, C=1.5, class_weight='balanced')
        self.is_trained = False

    def train(self, df):
        print("\n" + "="*60)
        print(" [ML MODEL 2] Training Thai Review Sentiment Classifier")
        print(f" (PyThaiNLP available: {HAS_PYTHAINLP})")
        print("="*60)
        
        X = df['review_text']
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f" Training samples: {len(X_train)} | Test samples: {len(X_test)}")
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        self.classifier.fit(X_train_vec, y_train)
        self.is_trained = True
        
        y_pred = self.classifier.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n[+] Test Accuracy: {acc * 100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        return acc

    def predict(self, text):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
            
        vec = self.vectorizer.transform([text])
        pred_sentiment = self.classifier.predict(vec)[0]
        probs = self.classifier.predict_proba(vec)[0]
        classes = self.classifier.classes_
        
        prob_dict = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probs)}
        aspects = self.extract_aspects(text)
        
        return pred_sentiment, prob_dict, aspects

    def extract_aspects(self, text):
        matched_aspects = []
        for aspect, keywords in ASPECT_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    matched_aspects.append(aspect)
                    break
        return matched_aspects if matched_aspects else ["ทั่วไป"]

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier, "is_trained": self.is_trained}, file_path)
        print(f"[+] Sentiment Model saved to: {file_path}")

    def load(self, file_path):
        payload = joblib.load(file_path)
        self.vectorizer = payload["vectorizer"]
        self.classifier = payload["classifier"]
        self.is_trained = payload["is_trained"]
        print(f"[+] Sentiment Model loaded from: {file_path}")


def train_and_save_sentiment_model(reviews_csv_path, model_output_path):
    df = pd.read_csv(reviews_csv_path, encoding='utf-8')
    analyzer = ThaiSentimentAnalyzer()
    analyzer.train(df)
    analyzer.save(model_output_path)
    
    # Test cases with realistic tourist feedback
    sample_tests = [
        "องค์พระสวยงามมาก บรรยากาศร่มรื่น ของกินรอบองค์พระอร่อยทุกอย่าง ประทับใจมากครับ",
        "บรรยากาศพอใช้ได้ แต่คนเยอะมากในวันหยุดเสาร์อาทิตย์ หาที่จอดรถยากวนอยู่สามรอบ",
        "ห้องน้ำไม่สะอาดเลย มีกลิ่นเหม็นมาก ที่จอดรถก็แคบ การจัดการแย่ ไม่ประทับใจ",
        "ตลาดน้ำของกินเยอะมาก ราคาเป็นกันเอง พ่อค้าแม่ค้าน่ารัก",
        "อากาศร้อนอบอ้าวมาก ทางเดินแออัด แต่กาแฟรสชาติดี พักผ่อนได้"
    ]
    
    print("\n" + "-"*60)
    print(" [Inference Test] Analyzing Sample Tourist Reviews")
    print("-"*60)
    for text in sample_tests:
        sent, probs, aspects = analyzer.predict(text)
        label_th = {"positive": "เชิงบวก (+)", "neutral": "เป็นกลาง (o)", "negative": "เชิงลบ (-)"}[sent]
        print(f" ข้อความรีวิว: \"{text}\"")
        print(f"   -> ความรู้สึก: [{label_th}] (ความมั่นใจ: {probs[sent]:.1f}%)")
        print(f"   -> ประเด็นที่พูดถึง: {', '.join(aspects)}\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "tourism_reviews.csv")
    model_path = os.path.join(base_dir, "data", "sentiment_model.joblib")
    train_and_save_sentiment_model(data_path, model_path)
