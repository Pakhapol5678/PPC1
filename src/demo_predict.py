"""
Interactive Demonstration Script:
Test Popularity Prediction and Thai Sentiment Analysis
"""
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.popularity_model import TourismPopularityModel
from src.sentiment_analyzer import ThaiSentimentAnalyzer
import pandas as pd

def run_demo():
    print("="*65)
    print("  ระบบทดสอบ Machine Learning: การท่องเที่ยวจังหวัดนครปฐม")
    print("="*65)
    
    enriched_csv = os.path.join(base_dir, "data", "enriched_tourism.csv")
    reviews_csv = os.path.join(base_dir, "data", "tourism_reviews.csv")
    
    print("\n[*] กำลังฝึกสอนโมเดลที่ 1: Popularity Classifier...")
    df_places = pd.read_csv(enriched_csv)
    pop_model = TourismPopularityModel()
    pop_model.train(df_places)
    
    print("\n[*] กำลังฝึกสอนโมเดลที่ 2: Thai Sentiment Analyzer...")
    df_reviews = pd.read_csv(reviews_csv)
    sent_model = ThaiSentimentAnalyzer()
    sent_model.train(df_reviews)
    
    print("\n" + "="*65)
    print("  ผลลัพธ์การทดสอบทำนายสถานที่ท่องเที่ยวใหม่ (Popularity Prediction)")
    print("="*65)
    
    new_places = [
        ("ตลาดน้ำแห่งใหม่ริมท่าจีน", "บางเลน", "นันทนาการและพักผ่อน"),
        ("วัดป่าสายวิปัสสนาศรัทธาธรรม", "เมืองนครปฐม", "ประวัติศาสตร์และวัฒนธรรม"),
        ("ฟาร์มผักไฮโดรโปนิกส์สามพราน", "สามพราน", "เกษตรกรรม"),
        ("พิพิธภัณฑ์ศิลปะพื้นบ้านโบราณ", "นครชัยศรี", "ศิลปะและวิทยาการ")
    ]
    
    for name, dist, cat in new_places:
        tier, probs = pop_model.predict(name, dist, cat)
        print(f"\n[+] สถานที่: '{name}'")
        print(f"    - ที่ตั้ง: อ.{dist} | หมวดหมู่: {cat}")
        print(f"    -> ทำนายระดับความนิยม: [{tier}]")
        print(f"    -> ความน่าจะเป็น: High {probs.get('High', 0)}%, Medium {probs.get('Medium', 0)}%, Low {probs.get('Low', 0)}%")
        
    print("\n" + "="*65)
    print("  ผลลัพธ์การทดสอบวิเคราะห์ความรู้สึกจากรีวิว (Sentiment Analysis)")
    print("="*65)
    
    test_reviews = [
        "องค์พระสวยงามมาก บรรยากาศร่มรื่น ของกินอร่อยทุกอย่าง ประทับใจมาก",
        "บรรยากาศพอใช้ได้ แต่คนเยอะมากในวันหยุด หาที่จอดรถยากวนอยู่สามรอบ",
        "ห้องน้ำไม่สะอาดเลย มีกลิ่นเหม็นมาก ที่จอดรถแคบ การจัดการแย่มาก",
        "ตลาดน้ำของกินเยอะมาก ราคาเป็นกันเอง พ่อค้าแม่ค้าน่ารัก อัธยาศัยดี",
        "ราคาอาหารเครื่องดื่มแพงเกินไป รอนานมาก พนักงานพูดจาไม่ค่อยดี"
    ]
    
    for rev in test_reviews:
        sent, probs, aspects = sent_model.predict(rev)
        th_label = {"positive": "เชิงบวก (+)", "neutral": "เป็นกลาง (o)", "negative": "เชิงลบ (-)"}[sent]
        print(f"\n[+] รีวิว: \"{rev}\"")
        print(f"    -> ความรู้สึก: [{th_label}] (มั่นใจ: {probs[sent]:.1f}%)")
        print(f"    -> ประเด็นที่ตรวจพบ: {', '.join(aspects)}")

if __name__ == "__main__":
    run_demo()
