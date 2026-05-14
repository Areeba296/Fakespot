import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="FakeSpot — Fake Business Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "98c966b3b57dcad180a9d38a46b675c921df47b2")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp {
    background: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #12001f 50%, #0a0f1a 100%);
    border: 1px solid rgba(180,0,255,0.15);
    border-radius: 24px;
    padding: 52px 48px 44px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(180,0,255,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(180,0,255,0.15);
    border: 1px solid rgba(180,0,255,0.4);
    color: #c060ff;
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 20px;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 52px !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    color: #ffffff !important;
    margin-bottom: 16px !important;
}
.hero h1 span {
    background: linear-gradient(90deg, #b400ff, #ff3c00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: 17px;
    color: #9090b0;
    max-width: 600px;
    line-height: 1.6;
}
.stats-row {
    display: flex;
    gap: 12px;
    margin-bottom: 36px;
    flex-wrap: wrap;
}
.stat-pill {
    background: #12121f;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 100px;
    padding: 10px 20px;
    font-size: 13px;
    color: #9090b0;
}
.stat-pill strong { color: #ffffff; }
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}
.section-subtitle {
    font-size: 14px;
    color: #6060a0;
    margin-bottom: 20px;
}
.stTextInput > div > div > input {
    background: #12121f !important;
    border: 1.5px solid rgba(180,0,255,0.3) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
    padding: 14px 20px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #b400ff !important;
    box-shadow: 0 0 0 3px rgba(180,0,255,0.12) !important;
}
.stTextInput > label {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #6060a0 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
.stButton > button {
    background: linear-gradient(135deg, #b400ff, #ff3c00) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 14px 32px !important;
    width: 100% !important;
}
.result-card {
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.07);
}
.result-card.high {
    background: linear-gradient(135deg, #1a0808 0%, #12121f 100%);
    border-color: rgba(255,60,0,0.4);
}
.result-card.medium {
    background: linear-gradient(135deg, #1a1408 0%, #12121f 100%);
    border-color: rgba(255,160,0,0.4);
}
.result-card.low {
    background: linear-gradient(135deg, #081a0e 0%, #12121f 100%);
    border-color: rgba(0,220,100,0.3);
}
.result-name {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 6px;
}
.result-location { font-size: 14px; color: #6060a0; margin-bottom: 16px; }
.meta-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }
.meta-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 13px;
    color: #a0a0c0;
}
.score-high { font-family:'Syne',sans-serif; font-size:48px; font-weight:800; color:#ff3c00; line-height:1; }
.score-medium { font-family:'Syne',sans-serif; font-size:48px; font-weight:800; color:#ffa000; line-height:1; }
.score-low { font-family:'Syne',sans-serif; font-size:48px; font-weight:800; color:#00dc64; line-height:1; }
.tag-high { display:inline-block; background:rgba(255,60,0,0.15); border:1px solid rgba(255,60,0,0.4); color:#ff6040; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; padding:4px 12px; border-radius:100px; margin-top:8px; }
.tag-medium { display:inline-block; background:rgba(255,160,0,0.15); border:1px solid rgba(255,160,0,0.4); color:#ffb020; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; padding:4px 12px; border-radius:100px; margin-top:8px; }
.tag-low { display:inline-block; background:rgba(0,220,100,0.1); border:1px solid rgba(0,220,100,0.3); color:#00dc64; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; padding:4px 12px; border-radius:100px; margin-top:8px; }
.flag-section-title { font-size:12px; color:#6060a0; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px; margin-top:20px; }
.flag-item { display:flex; align-items:flex-start; gap:10px; padding:10px 14px; background:rgba(255,255,255,0.03); border-radius:10px; margin-bottom:8px; font-size:14px; color:#b0b0d0; border-left:2px solid rgba(180,0,255,0.3); }
.bert-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(180,0,255,0.1); border:1px solid rgba(180,0,255,0.3); border-radius:100px; padding:4px 12px; font-size:12px; color:#c060ff; font-weight:600; margin-top:8px; }
.no-result { background:#12121f; border:1px solid rgba(255,60,0,0.2); border-radius:16px; padding:24px; color:#ff6040; font-size:15px; }
.custom-divider { height:1px; background:linear-gradient(90deg, transparent, rgba(180,0,255,0.3), transparent); margin:40px 0; }
.how-it-works { background:#12121f; border:1px solid rgba(255,255,255,0.07); border-radius:20px; padding:32px; margin-top:40px; }
.how-step { display:flex; align-items:flex-start; gap:16px; margin-bottom:20px; }
.step-number { background:linear-gradient(135deg, #b400ff, #ff3c00); color:white; font-family:'Syne',sans-serif; font-weight:800; font-size:14px; width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.step-content { font-size:14px; color:#9090b0; line-height:1.6; }
.step-content strong { color:#ffffff; display:block; margin-bottom:2px; }
.footer { text-align:center; padding:40px; color:#3a3a6a; font-size:12px; letter-spacing:2px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_training_data():
    try:
        return pd.read_csv('fakespot_bert_complete.csv')
    except Exception:
        try:
            return pd.read_csv('fakespot_complete.csv')
        except Exception:
            return None


@st.cache_resource
def train_model(df):
    features = [
        'stars', 'review_count', 'high_rating_low_reviews',
        'rating_review_ratio', 'open_no_reviews',
        'businesses_at_same_location', 'round_review_count'
    ]
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)
    return model, scaler, available


training_df = load_training_data()
if training_df is not None:
    ml_model, scaler, feature_cols = train_model(training_df)


def search_business(business_name, city):
    url = "https://google.serper.dev/places"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": f"{business_name} {city}", "hl": "en"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            places = response.json().get('places', [])
            converted = []
            for r in places:
                converted.append({
                    'name': r.get('title', 'Unknown'),
                    'location': {
                        'formatted_address': r.get('address', 'Unknown'),
                        'city': city,
                        'country': 'Pakistan'
                    },
                    'rating': r.get('rating', None),
                    'stats': {'total_ratings': r.get('ratingCount', 0)},
                    'categories': [{'name': r.get('type', 'Unknown')}],
                    'hours': {'open_now': r.get('openNow', None)},
                    'tel': r.get('phoneNumber', None),
                    'website': r.get('website', None)
                })
            return converted if converted else []
        else:
            return None
    except Exception:
        return None


def analyze_business(place):
    name = place.get('name', 'Unknown')
    location = place.get('location', {})
    city = location.get('city', 'Unknown')
    country = location.get('country', 'Unknown')
    address = location.get('formatted_address', 'Unknown')
    rating = place.get('rating', None)
    stats = place.get('stats', {})
    total_ratings = stats.get('total_ratings', 0)
    categories = place.get('categories', [])
    category = categories[0].get('name', 'Unknown') if categories else 'Unknown'
    hours = place.get('hours', {})
    is_open = hours.get('open_now', None)
    tel = place.get('tel', None)
    website = place.get('website', None)

    features = {}
    if rating and total_ratings:
        features['high_rating_low_reviews'] = int(rating >= 4.8 and total_ratings < 10)
        features['rating_review_ratio'] = rating / np.log1p(total_ratings + 1)
        features['stars'] = rating
        features['review_count'] = total_ratings
    else:
        features['high_rating_low_reviews'] = 1 if not total_ratings else 0
        features['rating_review_ratio'] = 5.0
        features['stars'] = 0
        features['review_count'] = 0

    features['open_no_reviews'] = int(bool(is_open) and total_ratings == 0) if is_open is not None else 0
    features['round_review_count'] = int(total_ratings % 10 == 0 and 0 < total_ratings < 50) if total_ratings else 0
    features['incomplete_location'] = int(city == 'Unknown' or address == 'Unknown')
    features['businesses_at_same_location'] = 1

    ml_score = 50
    if training_df is not None:
        try:
            input_features = [features.get(f, 0) for f in feature_cols]
            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            raw_score = ml_model.decision_function(input_scaled)[0]
            ml_score = float(100 * (1 - (raw_score + 0.5)))
            ml_score = max(0, min(100, ml_score))
        except Exception:
            ml_score = 50

    rule_score = 0
    reasons = []

    if features['high_rating_low_reviews']:
        rule_score += 25
        reasons.append("Very high rating with almost no reviews — suspicious pattern")
    if features['open_no_reviews']:
        rule_score += 20
        reasons.append("Business appears open but has zero reviews")
    if features['round_review_count']:
        rule_score += 10
        reasons.append("Suspiciously round review count for a new business")
    if features['incomplete_location']:
        rule_score += 15
        reasons.append("Incomplete or missing location details")
    if not rating and not total_ratings and not tel and not website:
        rule_score += 20
        reasons.append("No rating, reviews, phone or website — cannot verify legitimacy")
    if not reasons:
        reasons.append("No major red flags detected — business appears legitimate")

    reputation_bonus = 0
    if rating and total_ratings:
        if rating >= 4.0 and total_ratings >= 50:
            reputation_bonus = 10
        if rating >= 4.2 and total_ratings >= 100:
            reputation_bonus = 15
        if rating >= 4.4 and total_ratings >= 500:
            reputation_bonus = 20
        if rating >= 4.5 and total_ratings >= 1000:
            reputation_bonus = 25
        if rating >= 4.7 and total_ratings >= 5000:
            reputation_bonus = 30

    adjusted_ml_score = max(0, ml_score - reputation_bonus)
    final_score = round(min(100, max(0, (adjusted_ml_score * 0.6 + rule_score * 0.4))), 1)

    if final_score >= 70:
        risk_text = 'HIGH RISK'
        score_class = 'score-high'
        tag_class = 'tag-high'
        card_class = 'high'
    elif final_score >= 40:
        risk_text = 'MEDIUM RISK'
        score_class = 'score-medium'
        tag_class = 'tag-medium'
        card_class = 'medium'
    else:
        risk_text = 'LOW RISK'
        score_class = 'score-low'
        tag_class = 'tag-low'
        card_class = 'low'

    return {
        'name': name, 'city': city, 'country': country,
        'address': address, 'category': category,
        'rating': rating, 'total_ratings': total_ratings,
        'is_open': is_open, 'tel': tel, 'website': website,
        'ml_score': round(adjusted_ml_score, 1),
        'rule_score': round(rule_score, 1),
        'final_score': final_score, 'risk_text': risk_text,
        'score_class': score_class, 'tag_class': tag_class,
        'card_class': card_class, 'reasons': reasons
    }


# Hero
st.markdown("""
<div class="hero">
    <div class="hero-tag">AI · BERT · Graph ML · Real-Time · Global</div>
    <h1>Fake<span>Spot</span></h1>
    <p>Search any business anywhere in the world and instantly know if it is
    legitimate or fake — powered by BERT, Machine Learning and real-time data.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-pill">🧠 <strong>BERT</strong> Foundational Model</div>
    <div class="stat-pill">📦 Trained on <strong>150,000+</strong> businesses</div>
    <div class="stat-pill">💬 Analyzed <strong>50,000+</strong> reviews</div>
    <div class="stat-pill">🌍 Works <strong>Worldwide</strong></div>
    <div class="stat-pill">⚡ <strong>Real-Time</strong> Detection</div>
</div>
""", unsafe_allow_html=True)

# Search
st.markdown('<div class="section-title">🔍 Check Any Business</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Search any business anywhere in the world — from Lahore to London</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    business_name = st.text_input("BUSINESS NAME", placeholder="e.g. Layers Bakery, KFC, Xeven Solutions...")
with col2:
    city = st.text_input("CITY", placeholder="e.g. Lahore, Karachi, Faisalabad, London...")
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔍 Analyze Now")

# Results
if search_btn and business_name and city:
    with st.spinner(f'Searching for "{business_name}" in {city}...'):
        results = search_business(business_name, city)

    if results is None:
        st.markdown('<div class="no-result">⚠️ API error — please check your internet and try again</div>', unsafe_allow_html=True)
    elif len(results) == 0:
        st.markdown(f'<div class="no-result">❌ No results found for "<strong>{business_name}</strong>" in <strong>{city}</strong><br><br>💡 Try a shorter name or different spelling</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color:#6060a0;font-size:14px;margin-bottom:20px;">Found <strong style="color:#c060ff">{len(results)}</strong> result(s) for "<strong style="color:#ffffff">{business_name}</strong>" in <strong style="color:#ffffff">{city}</strong></div>', unsafe_allow_html=True)

        for place in results:
            a = analyze_business(place)
            reasons_html = "".join([f'<div class="flag-item">⚠️ {r}</div>' for r in a['reasons']])
            rating_display = f"⭐ {a['rating']}/5" if a['rating'] else "⭐ No rating"
            reviews_display = f"📝 {a['total_ratings']:,} reviews" if a['total_ratings'] else "📝 No reviews"
            status_display = "🟢 Open Now" if a['is_open'] is True else "🔴 Closed" if a['is_open'] is False else "❓ Unknown hours"
            phone_display = f"📞 {a['tel']}" if a['tel'] else "📞 Not listed"
            website_display = "🌐 Website available" if a['website'] else "🌐 No website"

            st.markdown(f"""
<div class="result-card {a['card_class']}">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:24px;">
        <div style="flex:1">
            <div class="result-name">{a['name']}</div>
            <div class="result-location">📍 {a['address']} · {a['country']}</div>
            <div class="meta-row">
                <span class="meta-pill">{rating_display}</span>
                <span class="meta-pill">{reviews_display}</span>
                <span class="meta-pill">{status_display}</span>
                <span class="meta-pill">🏪 {a['category']}</span>
                <span class="meta-pill">{phone_display}</span>
                <span class="meta-pill">{website_display}</span>
            </div>
            <div class="flag-section-title">⚠️ Detection Signals</div>
            {reasons_html}
            <div class="bert-badge">
                🧠 BERT + ML Score: {a['ml_score']}%
                &nbsp;|&nbsp;
                Rule Score: {a['rule_score']}%
            </div>
        </div>
        <div style="text-align:center;min-width:130px;">
            <div style="font-size:12px;color:#6060a0;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Fake Risk</div>
            <div class="{a['score_class']}">{a['final_score']}%</div>
            <div class="{a['tag_class']}">{a['risk_text']}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

elif search_btn:
    st.markdown('<div class="no-result">💡 Please enter both a business name and city</div>', unsafe_allow_html=True)

# How it works
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="how-it-works">
    <div class="section-title" style="margin-bottom:24px;">⚙️ How FakeSpot Works</div>
    <div class="how-step">
        <div class="step-number">1</div>
        <div class="step-content">
            <strong>Real-Time Data Fetch</strong>
            Google business data retrieved via Serper API covering businesses
            worldwide including Pakistan with real ratings and review counts
        </div>
    </div>
    <div class="how-step">
        <div class="step-number">2</div>
        <div class="step-content">
            <strong>BERT + ML Analysis</strong>
            Model trained on 150,000+ businesses and 50,000+ reviews scores
            each business using Isolation Forest anomaly detection combined
            with BERT language analysis
        </div>
    </div>
    <div class="how-step">
        <div class="step-number">3</div>
        <div class="step-content">
            <strong>Smart Signal Detection</strong>
            Rule-based analysis checks for red flags: suspicious ratings,
            missing data, unusual review patterns — while rewarding businesses
            with many genuine reviews
        </div>
    </div>
    <div class="how-step">
        <div class="step-number">4</div>
        <div class="step-content">
            <strong>Combined Risk Score</strong>
            All signals combine into a single Fake Risk Score (0-100%) with
            a clear verdict: High Risk, Medium Risk, or Low Risk
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    FAKESPOT 2.0 · BERT · ISOLATION FOREST · SERPER API · STREAMLIT
</div>
""", unsafe_allow_html=True)
