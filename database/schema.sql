-- Cloudflare D1 Database Schema for Nakhon Pathom Tourism System
-- Dialect: SQLite (Cloudflare D1 compatible)

DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS attractions;

CREATE TABLE attractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    district TEXT NOT NULL,
    category TEXT NOT NULL,
    rating REAL DEFAULT 4.0,
    review_count INTEGER DEFAULT 100,
    popularity_tier TEXT DEFAULT 'Medium',
    recorded_years TEXT,
    latest_year INTEGER DEFAULT 2569,
    tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attraction_id INTEGER,
    attraction_name TEXT NOT NULL,
    district TEXT,
    review_text TEXT NOT NULL,
    sentiment TEXT NOT NULL, -- positive, neutral, negative
    aspect TEXT, -- ที่จอดรถ, อาหาร, บรรยากาศ, ความสะอาด, ราคา, บริการ
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (attraction_id) REFERENCES attractions(id) ON DELETE CASCADE
);

-- Optimized indexes for fast edge queries
CREATE INDEX idx_attractions_district ON attractions(district);
CREATE INDEX idx_attractions_category ON attractions(category);
CREATE INDEX idx_attractions_tier ON attractions(popularity_tier);
CREATE INDEX idx_attractions_rating ON attractions(rating);
CREATE INDEX idx_reviews_attraction ON reviews(attraction_name);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment);
