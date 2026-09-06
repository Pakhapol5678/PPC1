# Nakhon Pathom Tourism Machine Learning Project 🏛️🍇

โปรเจกต์พัฒนาโมเดล Machine Learning สำหรับทำนายระดับความนิยมของแหล่งท่องเที่ยว และวิเคราะห์ความรู้สึก (Sentiment Analysis) จากรีวิวผู้ใช้ภาษาไทย โดยใช้ข้อมูลแหล่งท่องเที่ยวจังหวัดนครปฐม

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
nakhonpathom_tourism_ml/
├── data/
│   ├── raw_tourism.csv         # ข้อมูลตั้งต้นที่ให้มา (ปี 2563, 2564, 2565, 2569)
│   ├── cleaned_tourism.csv     # ข้อมูลหลังคลีนชื่ออำเภอและหมวดหมู่ พร้อมรวมข้อมูลซ้ำ
│   ├── enriched_tourism.csv    # ข้อมูลเพิ่มเรตติ้ง (Rating), จำนวนรีวิว, และ Popularity Tier
│   └── tourism_reviews.csv     # ข้อมูลรีวิวภาษาไทยพร้อม Sentiment Label (บวก, กลาง, ลบ)
├── src/
│   ├── data_cleaning.py        # สคริปต์ทำความสะอาดและปรับข้อมูลให้เป็นมาตรฐาน
│   ├── data_enricher.py        # สคริปต์ดึง/เพิ่มข้อมูลคะแนนและรีวิว (รองรับ Google Places API)
│   ├── popularity_model.py     # โมเดล ML ทำนายระดับความนิยม (Random Forest + TF-IDF)
│   ├── sentiment_analyzer.py   # โมเดล ML ตัดคำไทยและวิเคราะห์ความรู้สึกรีวิว (Logistic Regression + Aspect)
│   └── demo_predict.py         # ตัวอย่างทดสอบการทำนายกับสถานที่และรีวิวใหม่
├── requirements.txt            # รายการไลบรารีที่จำเป็น (pandas, scikit-learn, pythainlp ฯลฯ)
├── run_pipeline.py             # สคริปต์รัน Pipeline ทั้งหมดในคำสั่งเดียว
└── README.md                   # เอกสารประกอบโปรเจกต์
```

---

## 🚀 วิธีการติดตั้งและรันโปรเจกต์ (Quickstart)

### 1. ติดตั้งไลบรารีที่ต้องใช้
เปิด Terminal / Command Prompt แล้วพิมพ์:
```bash
pip install -r requirements.txt
```

### 2. รัน Pipeline ทั้งหมด (Data Cleaning -> Enrichment -> Training Models -> Reports)
```bash
python run_pipeline.py
```

### 3. ทดลองทดสอบโมเดลแบบ Interactive
```bash
python src/demo_predict.py
```

---

## 💡 รายละเอียดของ Machine Learning แต่ละโมเดล

### โมเดลที่ 1: ทำนายระดับความนิยม (Popularity Prediction)
* **โจทย์:** ทายว่าสถานที่เปิดใหม่จะมีระดับความนิยมอยู่ในกลุ่มใด (`High`, `Medium`, `Low`)
* **Features:** 
  * `อำเภอ` และ `ประเภทแหล่งท่องเที่ยว` (แปลงด้วย One-Hot Encoding)
  * คำสำคัญในชื่อสถานที่ (แปลงด้วย Character N-Gram TF-IDF)
* **อัลกอริทึม:** Random Forest Classifier & Gradient Boosting
* **ผลลัพธ์:** แสดงความน่าจะเป็นของแต่ละกลุ่มความนิยม และ Features สำคัญที่มีอิทธิพลต่อความนิยม

### โมเดลที่ 2: วิเคราะห์ความรู้สึกจากรีวิวภาษาไทย (Sentiment & Aspect Analysis)
* **โจทย์:** วิเคราะห์รีวิวของผู้ใช้งานว่าเป็น **เชิงบวก (+)**, **เป็นกลาง (o)**, หรือ **เชิงลบ (-)** พร้อมตรวจจับว่าพูดถึงประเด็นใด
* **กระบวนการ:**
  1. ตัดคำภาษาไทยด้วย `PyThaiNLP` (`newmm` dictionary-based)
  2. สกัดฟีเจอร์ด้วย TF-IDF N-grams
  3. จัดกลุ่มความรู้สึกด้วย Logistic Regression (Balanced weights)
  4. สกัด Aspect Keywords 6 มิติหลัก:
     * `ที่จอดรถและการเดินทาง`
     * `อาหารและของกิน`
     * `บรรยากาศและความสวยงาม`
     * `ความสะอาดและสิ่งอำนวยความสะดวก`
     * `ราคาและความคุ้มค่า`
     * `การบริการและความแออัด`

---

## 🌐 การเชื่อมต่อ Google Places API แบบสด (อุปกรณ์เสริม)
หากมี Google Maps API Key และต้องการดึงข้อมูลคะแนนรีวิวสดจาก Google:
```bash
# บน Windows PowerShell
$env:GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
python run_pipeline.py
```
*(หากไม่ใส่ API Key ระบบจะใช้ฐานข้อมูลคะแนนและรีวิวจริงที่คัดสรรแล้วสำหรับสถานที่สำคัญในนครปฐมโดยอัตโนมัติ)*
