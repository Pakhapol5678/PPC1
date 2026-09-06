"""
Machine Learning Module: Popularity Prediction for Tourism Attractions
Predicts popularity tier (High / Medium / Low) and rating from District, Category, and Name features.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class TourismPopularityModel:
    def __init__(self):
        # Feature preprocessor: OneHot for district & category, Char/Word N-Gram TF-IDF for Thai names
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['district', 'category']),
                ('text', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2), 'attraction_name')
            ]
        )
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42))
        ])
        self.classes_ = None

    def train(self, df):
        X = df[['attraction_name', 'district', 'category']]
        y = df['popularity_tier']
        
        # Train-test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print("\n" + "="*60)
        print(" [ML MODEL 1] Training Attraction Popularity Classifier")
        print("="*60)
        print(f" Training samples: {len(X_train)} | Test samples: {len(X_test)}")
        
        # Train
        self.pipeline.fit(X_train, y_train)
        self.classes_ = self.pipeline.named_steps['classifier'].classes_
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n[+] Test Accuracy: {acc * 100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
        cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in self.classes_], columns=[f"Pred {c}" for c in self.classes_])
        print("Confusion Matrix:")
        print(cm_df.to_string())
        
        # Feature importance inspection
        self._print_top_features()
        
        return acc

    def _print_top_features(self):
        try:
            clf = self.pipeline.named_steps['classifier']
            pre = self.pipeline.named_steps['preprocessor']
            
            # Get feature names
            cat_features = pre.named_transformers_['cat'].get_feature_names_out(['district', 'category'])
            text_features = [f"name_ngram:{f}" for f in pre.named_transformers_['text'].get_feature_names_out()]
            all_feature_names = np.concatenate([cat_features, text_features])
            
            importances = clf.feature_importances_
            top_indices = np.argsort(importances)[::-1][:15]
            
            print("\n[+] Top 15 Most Influential Features for Popularity:")
            for rank, idx in enumerate(top_indices, 1):
                clean_name = all_feature_names[idx].replace("district_", "อำเภอ: ").replace("category_", "ประเภท: ")
                print(f"  {rank:2d}. {clean_name:<35} (Score: {importances[idx]:.4f})")
        except Exception as e:
            print(f"[-] Could not extract feature importances: {e}")

    def predict(self, attraction_name, district, category):
        input_df = pd.DataFrame([{
            'attraction_name': attraction_name,
            'district': district,
            'category': category
        }])
        pred_tier = self.pipeline.predict(input_df)[0]
        pred_probs = self.pipeline.predict_proba(input_df)[0]
        
        prob_dict = {cls: round(prob * 100, 1) for cls, prob in zip(self.classes_, pred_probs)}
        return pred_tier, prob_dict

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.pipeline, file_path)
        print(f"[+] Popularity Model saved to: {file_path}")

    def load(self, file_path):
        self.pipeline = joblib.load(file_path)
        self.classes_ = self.pipeline.named_steps['classifier'].classes_
        print(f"[+] Popularity Model loaded from: {file_path}")


def train_and_save_popularity_model(enriched_csv_path, model_output_path):
    df = pd.read_csv(enriched_csv_path, encoding='utf-8')
    model = TourismPopularityModel()
    model.train(df)
    model.save(model_output_path)
    
    # Demonstration test with synthetic new tourist spots
    test_cases = [
        ("ตลาดน้ำโบราณบางเลน", "บางเลน", "นันทนาการและพักผ่อน"),
        ("วัดป่าธรรมเจริญ", "เมืองนครปฐม", "ประวัติศาสตร์และวัฒนธรรม"),
        ("ฟาร์มเมล่อนอินทรีย์สามพราน", "สามพราน", "เกษตรกรรม"),
        ("คาเฟ่ริมน้ำนครชัยศรี", "นครชัยศรี", "นันทนาการและพักผ่อน")
    ]
    
    print("\n" + "-"*60)
    print(" [Inference Test] Predicting Popularity for New Attractions")
    print("-"*60)
    for name, dist, cat in test_cases:
        tier, probs = model.predict(name, dist, cat)
        print(f" สถานที่: '{name}' ({dist} | {cat})")
        print(f"   -> ทำนายระดับความนิยม: [{tier}] | โอกาส: {probs}\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "enriched_tourism.csv")
    model_path = os.path.join(base_dir, "data", "popularity_model.joblib")
    train_and_save_popularity_model(data_path, model_path)
