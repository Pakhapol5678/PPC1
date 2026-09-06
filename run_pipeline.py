"""
Master Pipeline Runner for Nakhon Pathom Tourism Machine Learning Project
Executes Cleaning, Enrichment, Popularity Modeling, and Sentiment Analysis end-to-end.
"""
import os
import sys

# Ensure src can be imported
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from src.data_cleaning import clean_tourism_data
from src.data_enricher import enrich_tourism_data
from src.popularity_model import train_and_save_popularity_model
from src.sentiment_analyzer import train_and_save_sentiment_model
import pandas as pd

def print_header(title):
    print("\n" + "="*70)
    print(f"  >>> {title}")
    print("="*70)

def run_full_pipeline():
    print_header("นครปฐม ท่องเที่ยว AI & Machine Learning Pipeline")
    
    data_dir = os.path.join(base_dir, "data")
    raw_path = os.path.join(data_dir, "raw_tourism.csv")
    cleaned_path = os.path.join(data_dir, "cleaned_tourism.csv")
    enriched_path = os.path.join(data_dir, "enriched_tourism.csv")
    reviews_path = os.path.join(data_dir, "tourism_reviews.csv")
    
    popularity_model_path = os.path.join(data_dir, "popularity_model.joblib")
    sentiment_model_path = os.path.join(data_dir, "sentiment_model.joblib")
    
    # 1. Clean Data
    print_header("ขั้นตอนที่ 1: การคลีนและปรับมาตรฐานข้อมูล (Data Cleaning)")
    cleaned_df = clean_tourism_data(raw_path, cleaned_path)
    
    # 2. Enrich Data
    print_header("ขั้นตอนที่ 2: การดึงคะแนนและรีวิว (Data Enrichment)")
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", None)
    if api_key:
        print("[i] ตรวจพบ GOOGLE_MAPS_API_KEY จะทำการดึงข้อมูลสดผ่าน Google Places API")
    else:
        print("[i] โหมดสาธิต: ใช้ฐานข้อมูลคะแนนและรีวิวจริงที่คัดสรรแล้วสำหรับนครปฐม")
    enriched_df, reviews_df = enrich_tourism_data(cleaned_path, enriched_path, reviews_path, google_api_key=api_key)
    
    # 3. Train Popularity Model
    print_header("ขั้นตอนที่ 3: เทรนโมเดลทำนายความนิยม (Popularity Prediction)")
    train_and_save_popularity_model(enriched_path, popularity_model_path)
    
    # 4. Train Sentiment Model
    print_header("ขั้นตอนที่ 4: เทรนโมเดลวิเคราะห์ความรู้สึกรีวิวภาษาไทย (Sentiment Analysis)")
    train_and_save_sentiment_model(reviews_path, sentiment_model_path)
    
    # 5. Executive Summary & Insights
    print_header("สรุปภาพรวมข้อมูลและข้อมูลเชิงลึก (Tourism Insights)")
    
    print("\n[ Top 10 แหล่งท่องเที่ยวยอดนิยมที่สุดในจังหวัดนครปฐม ]")
    top10 = enriched_df.sort_values(by=['review_count', 'rating'], ascending=False).head(10)
    display_cols = ['attraction_name', 'district', 'category', 'rating', 'review_count', 'popularity_tier']
    print(top10[display_cols].to_string(index=False))
    
    print("\n[ สถิติค่าเฉลี่ยคะแนนและจำนวนรีวิวแยกตามอำเภอ ]")
    dist_summary = enriched_df.groupby('district').agg(
        จำนวนสถานที่=('attraction_name', 'count'),
        คะแนนเฉลี่ย=('rating', 'mean'),
        รีวิวเฉลี่ย=('review_count', 'mean')
    ).round(2).sort_values(by='รีวิวเฉลี่ย', ascending=False)
    print(dist_summary.to_string())
    
    print("\n[ สถิติค่าเฉลี่ยคะแนนและจำนวนรีวิวแยกตามประเภท ]")
    cat_summary = enriched_df.groupby('category').agg(
        จำนวนสถานที่=('attraction_name', 'count'),
        คะแนนเฉลี่ย=('rating', 'mean'),
        รีวิวเฉลี่ย=('review_count', 'mean')
    ).round(2).sort_values(by='รีวิวเฉลี่ย', ascending=False)
    print(cat_summary.to_string())
    
    print_header("Pipeline เสร็จสมบูรณ์พร้อมใช้งาน!")

if __name__ == "__main__":
    run_full_pipeline()
