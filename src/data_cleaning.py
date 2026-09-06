"""
Data Cleaning & Harmonization Module for Nakhon Pathom Tourism Dataset
"""
import os
import re
import pandas as pd

def clean_text(val):
    if not isinstance(val, str):
        return ""
    # Normalize multiple whitespaces and newlines
    cleaned = re.sub(r'\s+', ' ', val).strip()
    return cleaned

def standardize_district(district_name):
    district = clean_text(district_name)
    if district in ["เมือง", "เมืองนครปฐม", "อ.เมือง", "อำเภอเมือง"]:
        return "เมืองนครปฐม"
    return district

def standardize_category(category_name):
    cat = clean_text(category_name)
    
    if "ศิลปะ" in cat or "วิทยาการ" in cat:
        return "ศิลปะและวิทยาการ"
    elif "วัฒนธรรม" in cat or "ประวัติศาสตร์" in cat:
        return "ประวัติศาสตร์และวัฒนธรรม"
    elif "ธรรมชาติ" in cat:
        return "ธรรมชาติ"
    elif "เกษตร" in cat:
        return "เกษตรกรรม"
    elif "นันทนาการ" in cat or "พักผ่อน" in cat:
        return "นันทนาการและพักผ่อน"
    return cat

def clean_tourism_data(input_csv_path, output_csv_path):
    print(f"[*] Reading raw data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path, encoding='utf-8')
    
    # Clean text columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(clean_text)
            
    # Standardize District and Category
    df['อำเภอ_standard'] = df['อำเภอ'].apply(standardize_district)
    df['ประเภท_standard'] = df['ประเภทแหล่งท่องเที่ยว'].apply(standardize_category)
    
    # Attraction name cleaning
    df['แหล่งท่องเที่ยว_clean'] = df['แหล่งท่องเที่ยว'].apply(clean_text)
    
    # Sort by year descending to keep latest metadata
    df = df.sort_values(by=['ปี'], ascending=False)
    
    # Deduplicate based on cleaned attraction name and district
    # While keeping track of years it appeared
    year_agg = df.groupby(['แหล่งท่องเที่ยว_clean', 'อำเภอ_standard'])['ปี'].apply(lambda x: sorted(list(set(x)))).reset_index()
    year_agg.rename(columns={'ปี': 'ปีที่บันทึก'}, inplace=True)
    
    # Take latest record
    dedup_df = df.drop_duplicates(subset=['แหล่งท่องเที่ยว_clean', 'อำเภอ_standard'], keep='first').copy()
    dedup_df = dedup_df.merge(year_agg, on=['แหล่งท่องเที่ยว_clean', 'อำเภอ_standard'], how='left')
    
    # Select clean presentation columns
    cleaned_df = pd.DataFrame({
        'attraction_name': dedup_df['แหล่งท่องเที่ยว_clean'],
        'district': dedup_df['อำเภอ_standard'],
        'category': dedup_df['ประเภท_standard'],
        'recorded_years': dedup_df['ปีที่บันทึก'].apply(lambda yrs: ", ".join(map(str, yrs))),
        'latest_year': dedup_df['ปี']
    })
    
    # Sort alphabetically by district then attraction name
    cleaned_df = cleaned_df.sort_values(by=['district', 'category', 'attraction_name']).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    cleaned_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"[+] Data cleaned successfully! Saved to: {output_csv_path}")
    print(f"    - Total raw rows: {len(df)}")
    print(f"    - Unique attractions: {len(cleaned_df)}")
    print("\n[+] Breakdown by District:")
    print(cleaned_df['district'].value_counts().to_string())
    print("\n[+] Breakdown by Category:")
    print(cleaned_df['category'].value_counts().to_string())
    
    return cleaned_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw_tourism.csv")
    cleaned_path = os.path.join(base_dir, "data", "cleaned_tourism.csv")
    clean_tourism_data(raw_path, cleaned_path)
