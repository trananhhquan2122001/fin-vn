import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from rapidfuzz import fuzz
import requests
import json
import time
from collections import Counter
import random
from vnstock3 import Vnstock

# ---------- Machine Learning ----------
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# =============================================================================
# 1. CẤU HÌNH TRANG + CSS
# =============================================================================
st.set_page_config(
    page_title="FINEX VN Terminal - Công Nghệ Định Giá Doanh Nghiệp",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    div[data-testid="stMetricLabel"] {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    div[data-testid="stMetric"] { overflow: hidden; }
    .stApp { background: #0B0F19; color: #F8FAFC; font-family: 'Inter', 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background: #1E293B; border-right: 1px solid #334155; }
    .header-box { background: #1E293B; border: 1px solid #334155; border-radius: 16px; padding: 20px 28px; margin-bottom: 28px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
    .header-title { color: #F8FAFC; font-size: 2em; font-weight: 800; }
    .header-title span { color: #22C55E; }
    .header-sub { color: #94A3B8; font-size: 0.9em; }
    .header-right { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
    .vip-badge {
        background: linear-gradient(135deg, #F59E0B, #EAB308);
        color: #0B0F19;
        padding: 8px 22px;
        border-radius: 40px;
        font-weight: 700;
        font-size: 0.95em;
        cursor: pointer;
        white-space: nowrap;
    }
    .basic-badge {
        background: linear-gradient(135deg, #3B82F6, #60A5FA);
        color: #0B0F19;
        padding: 8px 22px;
        border-radius: 40px;
        font-weight: 700;
        font-size: 0.95em;
        cursor: pointer;
        white-space: nowrap;
    }
    .glass-card { background: #1E293B; border: 1px solid #334155; border-radius: 14px; padding: 20px 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .signal-buy { background: rgba(34, 197, 94, 0.12); border-left: 5px solid #22C55E; padding: 18px 22px; border-radius: 12px; margin: 16px 0; }
    .signal-buy .title { color: #22C55E; font-weight: 700; font-size: 1.2em; }
    .signal-reject { background: rgba(239, 68, 68, 0.12); border-left: 5px solid #EF4444; padding: 18px 22px; border-radius: 12px; margin: 16px 0; }
    .signal-reject .title { color: #EF4444; font-weight: 700; font-size: 1.2em; }
    .recommend-badge { display: inline-block; padding: 4px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9em; background: rgba(34, 197, 94, 0.15); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.3); }
    .rec-sell { background: rgba(239, 68, 68, 0.15); color: #EF4444; border-color: rgba(239, 68, 68, 0.3); }
    .rec-hold { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border-color: rgba(245, 158, 11, 0.3); }
    .zalo-contact-bar { background: #1E293B; border: 1px solid #334155; border-radius: 14px; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; margin-top: 20px; flex-wrap: wrap; }
    .zalo-contact-bar .number { color: #22C55E; font-size: 1.2em; font-weight: 700; text-decoration: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #1E293B; border-radius: 12px; padding: 4px; border: 1px solid #334155; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; color: #94A3B8; font-weight: 500; font-size: 0.85em; white-space: nowrap; }
    .stTabs [aria-selected="true"] { background: rgba(34, 197, 94, 0.15) !important; color: #22C55E !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 2. HÀM HELPER
# =============================================================================
def safe_float(val, default=0.0):
    """Ép kiểu số an toàn, tránh lỗi crash do N/A, None, String"""
    if val is None or val == "N/A":
        return default
    try:
        if isinstance(val, str):
            val = val.replace(',', '').replace('VNĐ', '').replace('đ', '').strip()
        v = float(val)
        return v if not (np.isnan(v) or np.isinf(v)) else default
    except (ValueError, TypeError):
        return default

def safe_divide(a, b, default=0.0):
    a, b = safe_float(a), safe_float(b)
    return a / b if b != 0 else default

def round_float(val, decimals=2):
    return round(safe_float(val), decimals) if safe_float(val) != 0 else 0.0

def format_price_vnd(price):
    p = safe_float(price)
    if p <= 0:
        return "N/A"
    return f"{p:,.0f}đ"

def format_vn_currency(value, unit='auto', decimals=2):
    """Hiển thị VNĐ: >= 1e12 → Nghìn tỷ, >= 1e9 → Tỷ, còn lại VNĐ"""
    v = safe_float(value)
    if v == 0:
        return "N/A"
    sign = "-" if v < 0 else ""
    abs_val = abs(v)
    if unit == 'nghin_ty' or (unit == 'auto' and abs_val >= 1e12):
        return f"{sign}{abs_val / 1e12:,.{decimals}f} Nghìn tỷ"
    if unit == 'ty' or (unit == 'auto' and abs_val >= 1e9):
        return f"{sign}{abs_val / 1e9:,.{decimals}f} Tỷ"
    return f"{sign}{abs_val:,.0f} VNĐ"

def format_percent(val):
    v = safe_float(val)
    if v == 0:
        return "N/A"
    return f"{v:.2f}%"

def format_eps(val):
    return format_price_vnd(val) + "/cp" if safe_float(val) > 0 else "N/A"

def format_pe_pb(val):
    v = safe_float(val)
    return f"{v:.2f}" if v > 0 else "N/A"

# =============================================================================
# 3. TÌM CỘT TRONG DATAFRAME
# =============================================================================
def find_column_by_keywords(df, keywords):
    for col in df.columns:
        col_low = str(col).lower()
        for kw in keywords:
            if kw.lower() in col_low:
                return col
    # fallback fuzzy
    for col in df.columns:
        for kw in keywords:
            if fuzz.partial_ratio(kw.lower(), str(col).lower()) >= 75:
                return col
    return None

def fix_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_indices):
            if i > 0:
                cols[idx] = f"{dup}_{i}"
    df.columns = cols
    return df

# =============================================================================
# 4. DANH SÁCH TICKER NGÂN HÀNG, CHỨNG KHOÁN, BẢO HIỂM
# =============================================================================
BANK_TICKERS = {'MBB','HDB','CTG','TCB','VCB','VPB','BID','ACB','SHB','TPB','MSB','OCB','VIB','STB','EIB','SSB','SGB','NAB','KLB','CBB','PGB','BAB','BSB'}
SECURITIES_TICKERS = {'SSI','VND','HCM','VCI','SHS','MBS','FTS','BSI','CTS','VIX','AGR','ORS','TVS','APS','BVS','PSI','VDS','EVS','DSC','SBS'}
INSURANCE_TICKERS = {'BVH','PVI','BMI','MIG','BIC','PGI','VNR','PTI','ABI','AIC'}

def is_bank(ticker):
    return str(ticker).upper() in BANK_TICKERS

def categorize_industry(df, ticker=None, row=None):
    # Ưu tiên cột mã ngành
    col = find_column_by_keywords(df, ['mã ngành', 'ma nganh', 'industry_code', 'icb_code'])
    if col and row is not None:
        try:
            code = int(float(str(row.get(col, '')).strip()))
            if code in (10, 20, 30):
                return code
            first = int(str(code)[0])
            return 20 if first == 2 else 30 if first == 3 else 10
        except:
            pass
    t = str(ticker).upper() if ticker else ''
    if t in BANK_TICKERS: return 20
    if t in SECURITIES_TICKERS or t in INSURANCE_TICKERS: return 30
    return 10

def get_bank_revenue(row, df):
    for cand in ['netInterestIncome', 'nii', 'totalOperatingIncome', 'toi', 'doanh thu thuần', 'revenue']:
        col = find_column_by_keywords(df, [cand])
        if col:
            v = safe_float(row.get(col, 0))
            if v != 0:
                return v, cand
    return 0.0, "Không tìm thấy"

# =============================================================================
# 5. HÀM TÍNH GIÁ & VỐN HÓA ĐỘNG
# =============================================================================
def get_dynamic_market_data(df, ticker, row=None):
    if row is None:
        ticker_col = find_column_by_keywords(df, ['mã cp', 'ticker', 'symbol'])
        if ticker_col is None:
            return {'price': None, 'market_cap': None, 'shares': None}
        matched = df[df[ticker_col].astype(str).str.upper() == ticker.upper()]
        if matched.empty:
            return {'price': None, 'market_cap': None, 'shares': None}
        row = matched.iloc[0]
    price_col = find_column_by_keywords(df, ['giá hiện tại', 'price', 'close_price'])
    mcap_col = find_column_by_keywords(df, ['vốn hóa', 'market_cap'])
    shares_col = find_column_by_keywords(df, ['cổ phiếu lưu hành', 'shares_outstanding', 'shares'])

    price = safe_float(row.get(price_col, 0)) if price_col else 0.0
    mcap = safe_float(row.get(mcap_col, 0)) if mcap_col else 0.0
    shares = safe_float(row.get(shares_col, 0)) if shares_col else 0.0

    if price <= 0 and mcap > 0 and shares > 0:
        price = mcap / shares
    if mcap <= 0 and price > 0 and shares > 0:
        mcap = price * shares
    return {'price': price if price > 0 else None,
            'market_cap': mcap if mcap > 0 else None,
            'shares': shares if shares > 0 else None}

# =============================================================================
# 6. TRÍCH XUẤT CÁC CHỈ SỐ TÀI CHÍNH
# =============================================================================
def extract_financial_metrics(row, df, ticker):
    # Tìm các cột cần thiết
    profit_col = find_column_by_keywords(df, ['lợi nhuận sau thuế', 'lnst', 'net profit', 'profit'])
    equity_col = find_column_by_keywords(df, ['vốn chủ sở hữu', 'vốn chủ', 'equity'])
    shares_col = find_column_by_keywords(df, ['số lượng cổ phiếu', 'cổ phiếu lưu hành', 'shares_outstanding', 'shares'])

    profit = safe_float(row.get(profit_col, 0)) if profit_col else 0.0
    equity = safe_float(row.get(equity_col, 0)) if equity_col else 0.0
    shares = safe_float(row.get(shares_col, 0)) if shares_col else 0.0

    # Lấy giá động
    md = get_dynamic_market_data(df, ticker, row=row)
    price = md['price'] if md['price'] else 0.0
    mcap = md['market_cap'] if md['market_cap'] else 0.0

    # Tính EPS, BVPS
    eps = profit / shares if shares > 0 else 0.0
    bvps = equity / shares if shares > 0 and equity > 0 else 0.0
    # Nếu vẫn thiếu BVPS, thử từ P/B
    if bvps <= 0:
        pb_col = find_column_by_keywords(df, ['pb', 'p/b'])
        if pb_col:
            pb = safe_float(row.get(pb_col, 0))
            if pb > 0 and price > 0:
                bvps = price / pb

    # ROE (%)
    roe = (profit / equity * 100) if equity > 0 else 0.0

    # P/E
    pe = price / eps if eps != 0 else 0.0
    if pe <= 0:
        pe_col = find_column_by_keywords(df, ['pe', 'p/e'])
        if pe_col:
            pe = safe_float(row.get(pe_col, 0))

    # P/B
    pb = price / bvps if bvps != 0 else 0.0
    if pb <= 0:
        pb_col = find_column_by_keywords(df, ['pb', 'p/b'])
        if pb_col:
            pb = safe_float(row.get(pb_col, 0))

    # Nguồn BVPS
    if bvps > 0:
        bvps_source = "BCTC gốc" if equity > 0 and shares > 0 else "Tính từ P/B"
        bvps_msg = f"✅ BVPS = {bvps:,.0f}"
    else:
        bvps_source = "Không có dữ liệu"
        bvps_msg = "⚠️ Không đủ dữ liệu BVPS"

    bank_rev = None
    if is_bank(ticker):
        bank_rev, _ = get_bank_revenue(row, df)

    return {
        'eps': eps,
        'bvps': bvps,
        'price': price,
        'market_cap': mcap,
        'roe': roe,
        'pe': pe,
        'pb': pb,
        'profit': profit,
        'equity': equity,
        'shares': shares,
        'bvps_source': bvps_source,
        'bvps_message': bvps_msg,
        'has_eps': eps != 0,
        'has_bvps': bvps != 0,
        'bank_revenue': bank_rev
    }

# =============================================================================
# 7. LOAD DATA CSV
# =============================================================================
@st.cache_data(ttl=3600)
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df = fix_duplicate_columns(df)
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return pd.DataFrame()

# =============================================================================
# 8. TRÍCH XUẤT NHIỀU CHỈ SỐ CHO TAB TỔNG QUAN
# =============================================================================
def extract_all_metrics(row, df, ticker):
    metrics = {}
    kw_groups = {
        'ROE': ['roe', 'return on equity'],
        'ROA': ['roa', 'return on assets'],
        'EPS': ['eps', 'earnings per share'],
        'BVPS': ['bvps', 'book value per share'],
        'Cổ tức (VNĐ/cp)': ['cổ tức', 'dividend'],
        'Tỷ suất cổ tức': ['tỷ suất cổ tức', 'dividend yield'],
        'Nợ/VCSH': ['nợ/vcsh', 'd/e', 'debt to equity'],
        'Biên LN gộp': ['biên lợi nhuận gộp', 'gross margin'],
        'Biên LN ròng': ['biên lợi nhuận ròng', 'net margin'],
        'P/E': ['pe', 'p/e'],
        'P/B': ['pb', 'p/b'],
        'Tổng tài sản': ['tổng tài sản', 'total assets'],
        'Vốn chủ sở hữu': ['vốn chủ', 'equity'],
        'Nợ dài hạn': ['nợ dài hạn', 'long term debt'],
        'Nợ ngắn hạn': ['nợ ngắn hạn', 'current liabilities'],
        'Tăng trưởng doanh thu (%)': ['tăng trưởng doanh thu', 'revenue growth'],
        'Tăng trưởng LN (%)': ['tăng trưởng lợi nhuận', 'profit growth'],
        'Vốn hóa': ['vốn hóa', 'market cap'],
    }
    is_bank = is_bank(ticker)
    for name, keys in kw_groups.items():
        if is_bank and name in ['Biên LN gộp', 'Biên LN ròng']:
            continue
        col = find_column_by_keywords(df, keys)
        if col:
            val = safe_float(row.get(col, 0))
            if val != 0:
                metrics[name] = val
    if is_bank:
        rev, _ = get_bank_revenue(row, df)
        if rev:
            metrics['Thu nhập lãi thuần (NII)'] = rev
    else:
        rev_col = find_column_by_keywords(df, ['doanh thu', 'revenue'])
        if rev_col:
            rev = safe_float(row.get(rev_col, 0))
            if rev:
                metrics['Doanh thu'] = rev
    return metrics

# =============================================================================
# 9. ML MODELS (giữ nguyên, dùng safe_float)
# =============================================================================
def label_rule(roe, roa, margin, de):
    r, a, m, d = safe_float(roe), safe_float(roa), safe_float(margin), safe_float(de)
    if r >= 15 and a >= 8 and m >= 20 and d < 1:
        return 2
    if r >= 8 and a >= 4 and d < 2:
        return 1
    return 0

@st.cache_resource
def train_risk_ensemble(df, ticker_col):
    seed_X = np.array([[22.5,12,35,0.3],[18,9.5,28,0.5],[12,5,15,1.2],[3,1,5,3],[-2,-1,2,4]], dtype=np.float64)
    seed_y = np.array([2,2,1,0,0])
    X_list, y_list = [], []
    if not df.empty:
        roe_col = find_column_by_keywords(df, ['roe'])
        roa_col = find_column_by_keywords(df, ['roa'])
        margin_col = find_column_by_keywords(df, ['biên lợi nhuận gộp', 'gross margin'])
        de_col = find_column_by_keywords(df, ['nợ/vcsh', 'd/e'])
        if all([roe_col, roa_col, margin_col, de_col]):
            for _, row in df.iterrows():
                r = safe_float(row.get(roe_col, 0))
                a = safe_float(row.get(roa_col, 0))
                m = safe_float(row.get(margin_col, 20))
                d = safe_float(row.get(de_col, 0.5))
                X_list.append([r, a, m, d])
                y_list.append(label_rule(r, a, m, d))
    if X_list:
        X = np.vstack([seed_X, np.array(X_list, dtype=np.float64)])
        y = np.concatenate([seed_y, np.array(y_list)])
    else:
        X, y = seed_X, seed_y
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models = {}
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_enc)
    models['rf'] = rf
    mlp = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=42)
    mlp.fit(X_scaled, y_enc)
    models['mlp'] = mlp
    if LGB_AVAILABLE:
        lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        lgbm.fit(X_scaled, y_enc)
        models['lgbm'] = lgbm
    if XGB_AVAILABLE:
        xgbc = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='mlogloss', random_state=42)
        xgbc.fit(X_scaled, y_enc)
        models['xgb'] = xgbc
    models['le'] = le
    return models, scaler

def predict_risk(roe, roa, margin, de):
    r, a, m, d = safe_float(roe), safe_float(roa), safe_float(margin), safe_float(de)
    if r==0 and a==0 and m==0 and d==0:
        return 1, 50.0
    df = st.session_state.get('df', pd.DataFrame())
    ticker_col = st.session_state.get('ticker_col', None)
    models, scaler = train_risk_ensemble(df, ticker_col)
    inp = np.array([[r, a, m, d]], dtype=np.float64)
    inp_scaled = scaler.transform(inp)
    preds, probs = [], []
    for name, model in models.items():
        if name == 'le': continue
        pred = model.predict(inp_scaled)[0]
        prob = model.predict_proba(inp_scaled)[0]
        preds.append(pred)
        probs.append(prob)
    final = Counter(preds).most_common(1)[0][0]
    final_label = models['le'].inverse_transform([final])[0]
    conf = float(np.mean([prob[final] for prob in probs]) * 100)
    return int(final_label), round(conf, 1)

@st.cache_resource
def train_eps_regressor(df):
    # Tương tự như trên, giữ nguyên
    seed_X = np.array([[22.5,12,35,0.3],[18,9.5,28,0.5],[12,5,15,1.2],[3,1,5,3],[-2,-1,2,4]], dtype=np.float64)
    seed_y = np.array([8000,5000,2500,800,-500], dtype=np.float64)
    X_list, y_list = [], []
    if not df.empty:
        eps_col = find_column_by_keywords(df, ['eps'])
        roe_col = find_column_by_keywords(df, ['roe'])
        roa_col = find_column_by_keywords(df, ['roa'])
        margin_col = find_column_by_keywords(df, ['biên lợi nhuận gộp', 'gross margin'])
        de_col = find_column_by_keywords(df, ['nợ/vcsh', 'd/e'])
        if all([eps_col, roe_col, roa_col, margin_col, de_col]):
            for _, row in df.iterrows():
                r = safe_float(row.get(roe_col, 0))
                a = safe_float(row.get(roa_col, 0))
                m = safe_float(row.get(margin_col, 20))
                d = safe_float(row.get(de_col, 0.5))
                e = safe_float(row.get(eps_col, 2000))
                X_list.append([r, a, m, d])
                y_list.append(e)
    if X_list:
        X = np.vstack([seed_X, np.array(X_list, dtype=np.float64)])
        y = np.concatenate([seed_y, np.array(y_list, dtype=np.float64)])
    else:
        X, y = seed_X, seed_y
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models = {}
    rfr = RandomForestRegressor(n_estimators=50, random_state=42)
    rfr.fit(X_scaled, y)
    models['rf'] = rfr
    mlp = MLPRegressor(hidden_layer_sizes=(50,25), max_iter=500, random_state=42)
    mlp.fit(X_scaled, y)
    models['mlp'] = mlp
    if LGB_AVAILABLE:
        lgbm = lgb.LGBMRegressor(n_estimators=50, random_state=42)
        lgbm.fit(X_scaled, y)
        models['lgbm'] = lgbm
    if XGB_AVAILABLE:
        xgbr = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
        xgbr.fit(X_scaled, y)
        models['xgb'] = xgbr
    return models, scaler

def predict_eps(roe, roa, margin, de):
    r, a, m, d = safe_float(roe), safe_float(roa), safe_float(margin), safe_float(de)
    if r==0 and a==0 and m==0 and d==0:
        return 0.0
    df = st.session_state.get('df', pd.DataFrame())
    models, scaler = train_eps_regressor(df)
    inp = np.array([[r, a, m, d]], dtype=np.float64)
    inp_scaled = scaler.transform(inp)
    preds = [model.predict(inp_scaled)[0] for model in models.values()]
    return round_float(np.mean(preds), 0)

def hybrid_valuation(price, eps, bvps, roe, margin, de, sector):
    price, eps, bvps, roe, margin, de = safe_float(price), safe_float(eps), safe_float(bvps), safe_float(roe), safe_float(margin), safe_float(de)
    if price <= 0:
        price = 0.0
    graham = (22.5 * eps * bvps) ** 0.5 if eps > 0 and bvps > 0 else price * 0.8
    roa = safe_divide(roe, (1 + de), roe) if de >= 0 else roe
    ml_eps = predict_eps(roe, roa, margin, de) if roe > 0 and roa > 0 else eps
    ml_bvps = bvps if bvps > 0 else (roe / 10) * price if roe > 0 else price * 0.5
    ml_graham = (22.5 * ml_eps * ml_bvps) ** 0.5 if ml_eps > 0 and ml_bvps > 0 else graham
    if sector == 'banking':
        excess = bvps + (bvps * ((roe / 100) - 0.10)) / (0.10 - 0.08) if roe > 0 else bvps * 1.2
        hybrid = 0.4 * graham + 0.3 * ml_graham + 0.3 * excess
    elif sector == 'securities':
        pe_adj = 8.5 + 2 * (10 if margin > 20 else 7)
        graham_pe = eps * pe_adj
        hybrid = 0.5 * graham + 0.5 * graham_pe
    else:
        hybrid = 0.5 * graham + 0.5 * ml_graham
    result = max(hybrid, price * 0.6) if price > 0 else hybrid
    return round_float(result, 0) if result > 0 else None

def forecast_trend(value, growth, periods=4):
    v, g = safe_float(value), safe_float(growth)
    if v == 0:
        return [0.0] * periods
    return [round_float(v * ((1 + g/100) ** i), 0) for i in range(1, periods+1)]

# =============================================================================
# 10. BỘ LỌC 3 TẦNG VÀ ALTMAN Z
# =============================================================================
def calculate_altman_z(row, df):
    cols = {
        'wc': find_column_by_keywords(df, ['vốn lưu động']),
        'ta': find_column_by_keywords(df, ['tổng tài sản']),
        're': find_column_by_keywords(df, ['lợi nhuận giữ lại']),
        'ebit': find_column_by_keywords(df, ['ebit', 'lợi nhuận trước thuế']),
        'me': find_column_by_keywords(df, ['vốn hóa']),
        'tl': find_column_by_keywords(df, ['tổng nợ']),
        'sales': find_column_by_keywords(df, ['doanh thu'])
    }
    vals = {}
    for k, c in cols.items():
        vals[k] = safe_float(row.get(c, 0)) if c else 0.0
    ta = vals['ta']
    if ta == 0:
        return None
    A = vals['wc'] / ta
    B = vals['re'] / ta
    C = vals['ebit'] / ta
    D = vals['me'] / (vals['tl'] if vals['tl'] != 0 else 1)
    E = vals['sales'] / ta
    return 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

def check_signal(row, df, ticker, sector, intrinsic, price):
    msgs = []
    adtv_col = find_column_by_keywords(df, ['adtv20', 'khối lượng trung bình 20'])
    if adtv_col:
        adtv = safe_float(row.get(adtv_col, 0))
        if adtv < 200000:
            return False, f"❌ Thanh khoản thấp (ADTV20={adtv:,.0f})"
        msgs.append(f"✅ ADTV20={adtv:,.0f}")
    else:
        msgs.append("⚠️ Bỏ qua thanh khoản")
    # EPS, BVPS
    eps_col = find_column_by_keywords(df, ['eps'])
    bvps_col = find_column_by_keywords(df, ['bvps'])
    eps = safe_float(row.get(eps_col, 0)) if eps_col else 0
    bvps = safe_float(row.get(bvps_col, 0)) if bvps_col else 0
    if bvps <= 0:
        equity_col = find_column_by_keywords(df, ['vốn chủ'])
        shares_col = find_column_by_keywords(df, ['cổ phiếu lưu hành'])
        if equity_col and shares_col:
            eq, sh = safe_float(row.get(equity_col, 0)), safe_float(row.get(shares_col, 0))
            if eq > 0 and sh > 0:
                bvps = eq / sh
        if bvps <= 0:
            pb_col = find_column_by_keywords(df, ['pb'])
            if pb_col:
                pb = safe_float(row.get(pb_col, 0))
                if pb > 0 and price > 0:
                    bvps = price / pb
        if bvps <= 0:
            bvps = price / 10 if price > 0 else 2500
        msgs.append(f"ℹ️ BVPS tự tính = {bvps:,.0f}")
    else:
        msgs.append(f"✅ EPS={eps:,.0f}, BVPS={bvps:,.0f}")

    if sector == 'banking':
        npl_col = find_column_by_keywords(df, ['npl', 'nợ xấu'])
        if npl_col:
            npl = safe_float(row.get(npl_col, 100))
            if npl < 2:
                msgs.append(f"✅ NPL={npl:.2f}%")
            else:
                return False, f"❌ NPL={npl:.2f}% >= 2%"
        else:
            msgs.append("⚠️ Bỏ qua NPL")
    else:
        z = calculate_altman_z(row, df)
        if z is not None:
            if z > 1.8:
                msgs.append(f"✅ Altman Z={z:.2f}")
            else:
                return False, f"❌ Altman Z={z:.2f} <= 1.8"

    if intrinsic and intrinsic > 0 and price > 0:
        mos = (intrinsic - price) / intrinsic * 100
        if mos >= 20:
            msgs.append(f"✅ MOS={mos:.1f}%")
        else:
            return False, f"❌ MOS={mos:.1f}% < 20%"
    else:
        return False, "⚠️ Không có định giá để tính MOS"
    return True, " | ".join(msgs)

# =============================================================================
# 11. FETCH REAL-TIME (vnstock3 + fallback CSV)
# =============================================================================
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
]

@st.cache_data(ttl=3600)
def get_stock_data_bulletproof(symbol):
    sources = ['VCI', 'TCBS', 'MSN']
    price, market_cap, pe, pb, roe = 0.0, "N/A", "N/A", "N/A", "N/A"
    df_income, df_balance = pd.DataFrame(), pd.DataFrame()
    active_source = "Không kết nối"
    for src in sources:
        try:
            stock = Vnstock().stock(symbol=symbol, source=src)
            df_p = stock.quote.history(period='1D')
            if df_p is not None and not df_p.empty:
                raw = df_p['close'].iloc[-1]
                price = float(raw * 1000 if raw < 1000 else raw)
            df_r = stock.finance.ratio(period='year', lang='vi')
            if df_r is not None and not df_r.empty:
                latest = df_r.iloc[0]
                for col in df_r.columns:
                    c_low = str(col).lower().strip()
                    val = latest[col]
                    if pd.notna(val) and val != "":
                        if 'p/e' in c_low or 'pe' in c_low:
                            pe = f"{float(val):.2f}"
                        elif 'p/b' in c_low or 'pb' in c_low:
                            pb = f"{float(val):.2f}"
                        elif 'roe' in c_low:
                            rv = float(val)
                            roe = f"{rv*100:.2f}%" if rv < 5 else f"{rv:.2f}%"
                        elif 'vốn hóa' in c_low or 'marketcap' in c_low:
                            vv = float(val)
                            market_cap = f"{vv/1e9:,.1f} Tỷ"
            inc = stock.finance.income_statement(period='year', lang='vi')
            bs = stock.finance.balance_sheet(period='year', lang='vi')
            if inc is not None: df_income = inc
            if bs is not None: df_balance = bs
            if price > 0 or not df_income.empty:
                active_source = src
                break
        except:
            continue
    if price == 0 and df_income.empty:
        fallback = "vn_top598_financial_statements_master_2022_2025.csv"
        if not os.path.exists(fallback):
            fallback = os.path.join("data", fallback)
        if os.path.exists(fallback):
            try:
                df_fb = pd.read_csv(fallback, encoding='utf-8-sig')
                tcol = find_column_by_keywords(df_fb, ['mã cp', 'ticker'])
                if tcol:
                    mask = df_fb[tcol].astype(str).str.upper() == symbol
                    if mask.any():
                        r = df_fb[mask].iloc[0]
                        price = safe_float(r.get(find_column_by_keywords(df_fb, ['giá']), 0))
                        pe_fb = safe_float(r.get(find_column_by_keywords(df_fb, ['pe']), 0))
                        if pe_fb > 0: pe = f"{pe_fb:.2f}"
                        pb_fb = safe_float(r.get(find_column_by_keywords(df_fb, ['pb']), 0))
                        if pb_fb > 0: pb = f"{pb_fb:.2f}"
                        roe_fb = safe_float(r.get(find_column_by_keywords(df_fb, ['roe']), 0))
                        if roe_fb > 0: roe = f"{roe_fb:.2f}%"
                        df_income = pd.DataFrame([{
                            'Revenue': safe_float(r.get('Doanh thu', 0)),
                            'Net Profit': safe_float(r.get('Lợi nhuận sau thuế', 0)),
                            'Total Assets': safe_float(r.get('Tổng tài sản', 0)),
                            'Total Equity': safe_float(r.get('Vốn chủ sở hữu', 0))
                        }])
                        df_balance = pd.DataFrame([{
                            'Total Assets': safe_float(r.get('Tổng tài sản', 0)),
                            'Total Equity': safe_float(r.get('Vốn chủ sở hữu', 0))
                        }])
                        active_source = "Fallback CSV"
                        mc = safe_float(r.get('Vốn hóa', 0))
                        if mc > 0:
                            market_cap = format_vn_currency(mc)
            except:
                pass
    return price, market_cap, pe, pb, roe, df_income, df_balance, active_source, None

def render_real_time(ticker):
    st.subheader(f"📡 DỮ LIỆU REAL-TIME {ticker}")
    with st.spinner("Đang lấy dữ liệu..."):
        price, mcap, pe, pb, roe, inc, bs, src, _ = get_stock_data_bulletproof(ticker)
    if price == 0 and inc.empty:
        st.error("Không lấy được dữ liệu")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Giá", format_price_vnd(price))
    c2.metric("Vốn hóa", mcap)
    c3.metric("P/E", pe)
    c4, c5, c6 = st.columns(3)
    c4.metric("P/B", pb)
    c5.metric("ROE", roe)
    c6.metric("Nguồn", src)
    if not inc.empty:
        st.dataframe(inc, use_container_width=True)
    if not bs.empty:
        st.dataframe(bs, use_container_width=True)

# =============================================================================
# 12. POPUP VIP
# =============================================================================
@st.dialog("🚀 NÂNG CẤP FINEX VN VIP")
def vip_dialog():
    st.markdown("**Nhận trọn bộ Bộ Lọc 3 Tầng & AI Signal** chỉ 99.000đ/tháng.")
    st.info("Chuyển khoản VietQR để kích hoạt trong 30s")
    if st.button("✅ Đăng ký ngay"):
        st.success("Đã ghi nhận! Liên hệ Zalo 0327 625 853.")
        st.balloons()

# =============================================================================
# 13. MAIN
# =============================================================================
def main():
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None

    # HEADER
    st.markdown("""
    <div class="header-box">
        <div><div class="header-title">📊 FINEX VN <span>TERMINAL</span></div>
        <div class="header-sub">Hệ Thống Phân Tích & Định Giá Doanh Nghiệp</div></div>
        <div class="header-right">
            <div class="basic-badge" onclick="document.querySelector('[data-testid=\"stButton\"] button')?.click();">⭐ Gói 99K</div>
            <div class="vip-badge" onclick="alert('Liên hệ Zalo 0327 625 853')">💎 VIP 299K</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Mở popup VIP", key="vip_btn"):
        vip_dialog()

    st.sidebar.header("🔍 CHỌN NGUỒN DỮ LIỆU")
    data_source = st.sidebar.selectbox("Loại:", ["📂 File CSV", "📡 API vnstock3"])

    df = pd.DataFrame()
    sector = "general"
    selected_ticker = None
    df_inc, df_bs = pd.DataFrame(), pd.DataFrame()

    if data_source == "📂 File CSV":
        dataset = st.sidebar.selectbox("Khối dữ liệu:", ["📊 Top 598 Doanh nghiệp tổng hợp"])
        filepath = "vn_top598_financial_statements_master_2022_2025.csv"
        if not os.path.exists(filepath):
            filepath = os.path.join("data", filepath)
        if not os.path.exists(filepath):
            st.error("Không tìm thấy file CSV")
            st.stop()
        df = load_data(filepath)
        if df.empty:
            st.stop()
        ticker_col = find_column_by_keywords(df, ['mã cp', 'ticker'])
        if ticker_col is None:
            st.error("Không tìm thấy cột mã CP")
            st.stop()
        st.session_state.ticker_col = ticker_col
        ticker_list = sorted(df[ticker_col].astype(str).unique().tolist())
        if st.session_state.selected_ticker is None or st.session_state.selected_ticker not in ticker_list:
            st.session_state.selected_ticker = ticker_list[0]
        selected_ticker = st.sidebar.selectbox("Chọn mã:", ticker_list, index=ticker_list.index(st.session_state.selected_ticker))
        if selected_ticker != st.session_state.selected_ticker:
            st.session_state.selected_ticker = selected_ticker
            st.rerun()
        row_data = df[df[ticker_col].astype(str) == selected_ticker].iloc[0]
        sector = detect_sector(filepath, dataset)

    else:  # API
        ticker_api = st.sidebar.text_input("Mã CP:", value="HPG").strip().upper()
        if not ticker_api:
            st.stop()
        price, mcap, pe, pb, roe, df_inc, df_bs, src, _ = get_stock_data_bulletproof(ticker_api)
        if df_inc.empty:
            st.warning(f"Không có BCTC cho {ticker_api}, chỉ hiển thị thông tin cơ bản")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Giá", format_price_vnd(price))
            c2.metric("Vốn hóa", mcap)
            c3.metric("P/E", pe)
            c4.metric("P/B", pb)
            c5.metric("ROE", roe)
            st.stop()
        else:
            recent_inc = df_inc.iloc[0]
            recent_bs = df_bs.iloc[0] if not df_bs.empty else pd.Series()
            data = {
                'Mã CP': ticker_api,
                'Doanh thu': recent_inc.get('Revenue', 0),
                'Lợi nhuận sau thuế': recent_inc.get('Net Profit', 0),
                'Tổng tài sản': recent_bs.get('Total Assets', 0),
                'Vốn chủ sở hữu': recent_bs.get('Total Equity', 0),
                'Giá hiện tại': price,
                'Vốn hóa': 0,
            }
            df = pd.DataFrame([data])
            selected_ticker = ticker_api
            row_data = df.iloc[0]
            st.session_state.ticker_col = 'Mã CP'
            sector = detect_sector(ticker_api, "")

    st.session_state.df = df
    if selected_ticker is None:
        st.warning("Chưa chọn mã")
        st.stop()

    # Lấy chỉ số tài chính
    fin = extract_financial_metrics(row_data, df, selected_ticker)
    eps, bvps = fin['eps'], fin['bvps']
    price = fin['price']
    mcap = fin['market_cap']
    roe, pe, pb = fin['roe'], fin['pe'], fin['pb']
    industry = categorize_industry(df, selected_ticker, row_data)
    if industry == 20:
        sector = 'banking'
    elif industry == 30 and sector not in ('securities','insurance'):
        sector = 'securities'
    margin = 0.0 if industry == 20 else safe_float(row_data.get(find_column_by_keywords(df, ['biên lợi nhuận gộp']), 0))
    de = safe_float(row_data.get(find_column_by_keywords(df, ['nợ/vcsh', 'd/e']), 0))
    intrinsic = hybrid_valuation(price, eps, bvps, roe, margin, de, sector)
    bank_rev = fin.get('bank_revenue', None)

    # Thông tin nhanh (5 cột)
    st.markdown("#### 📊 Thông tin nhanh")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Giá", format_price_vnd(price))
    if mcap and mcap > 0:
        cap_display = format_vn_currency(mcap)
    else:
        cap_display = "N/A"
    c2.metric("Vốn hóa", cap_display)
    c3.metric("P/E", format_pe_pb(pe))
    c4.metric("P/B", format_pe_pb(pb))
    c5.metric("ROE", format_percent(roe))

    # Tín hiệu
    st.markdown("---")
    st.markdown("### 🚦 TÍN HIỆU ĐẦU TƯ - BỘ LỌC 3 TẦNG")
    if fin['has_eps'] or eps != 0:
        passed, msg = check_signal(row_data, df, selected_ticker, sector, intrinsic, price)
        if passed:
            st.markdown(f"""
            <div class="signal-buy"><div class="title">✅ TÍN HIỆU MUA</div>
            <div class="signal-msg">{msg}</div></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="signal-reject"><div class="title">⛔ KHÔNG PHÁT TÍN HIỆU</div>
            <div class="signal-msg">{msg}</div></div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="signal-reject"><div class="title">⚠️ THIẾU DỮ LIỆU</div>
        <div class="signal-msg">Không có EPS hoặc BVPS từ BCTC gốc.</div></div>
        """, unsafe_allow_html=True)

    # TABS
    tabs = st.tabs(["📋 DỮ LIỆU", "🧮 ĐỊNH GIÁ", "📈 DỰ BÁO", "📊 TỔNG QUAN", "🔎 TRA CỨU", "🤖 ML", "🧠 DL", "📡 REAL-TIME"])

    with tabs[0]:  # DỮ LIỆU
        if data_source == "📂 File CSV":
            st.subheader("Dữ liệu từ CSV")
            st.dataframe(df, use_container_width=True)
        else:
            st.subheader("Báo cáo thu nhập")
            if not df_inc.empty:
                st.dataframe(df_inc, use_container_width=True)
            st.subheader("Bảng cân đối")
            if not df_bs.empty:
                st.dataframe(df_bs, use_container_width=True)

    with tabs[1]:  # ĐỊNH GIÁ
        st.subheader(f"🧮 ĐỊNH GIÁ CHI TIẾT: {selected_ticker}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Giá hiện tại", format_price_vnd(price))
        col2.metric("EPS", format_eps(eps))
        col3.metric("BVPS", format_eps(bvps))
        col4.metric("Định giá Hybrid", format_price_vnd(intrinsic) if intrinsic else "N/A")

        st.markdown("#### 📌 Chỉ số quy mô")
        col5, col6, col7 = st.columns(3)
        if is_bank(selected_ticker) and bank_rev:
            rev_display = format_vn_currency(bank_rev, unit='ty')
            rev_label = "NII/TOI"
        else:
            rev_col = find_column_by_keywords(df, ['doanh thu'])
            rev_val = safe_float(row_data.get(rev_col, 0)) if rev_col else 0
            rev_display = format_vn_currency(rev_val, unit='ty') if rev_val else "N/A"
            rev_label = "Doanh thu"
        profit_val = safe_float(row_data.get(find_column_by_keywords(df, ['lợi nhuận sau thuế']), 0)) or 0
        asset_val = safe_float(row_data.get(find_column_by_keywords(df, ['tổng tài sản']), 0)) or 0
        col5.metric(rev_label, rev_display)
        col6.metric("Lợi nhuận", format_vn_currency(profit_val, unit='ty') if profit_val else "N/A")
        col7.metric("Tổng tài sản", format_vn_currency(asset_val, unit='ty') if asset_val else "N/A")

        # Thêm phần đánh giá chất lượng (giữ nguyên)
        st.markdown("---")
        st.markdown("### 🏆 ĐÁNH GIÁ CHẤT LƯỢNG DOANH NGHIỆP")
        div_yield = safe_float(row_data.get(find_column_by_keywords(df, ['tỷ suất cổ tức']), 0))
        de_ratio = safe_float(row_data.get(find_column_by_keywords(df, ['nợ/vcsh', 'd/e']), 0))
        gross_margin = 0.0 if industry == 20 else safe_float(row_data.get(find_column_by_keywords(df, ['biên lợi nhuận gộp']), 0))
        growth_rev = safe_float(row_data.get(find_column_by_keywords(df, ['tăng trưởng doanh thu']), 0))
        growth_profit = safe_float(row_data.get(find_column_by_keywords(df, ['tăng trưởng lợi nhuận']), 0))
        if intrinsic and intrinsic > 0 and price > 0:
            mos = (intrinsic - price) / intrinsic * 100
        else:
            mos = None
        score, total = 0, 0
        checks = []
        if mos is not None:
            total += 1
            if mos >= 20:
                score += 1
                checks.append(f"✅ MOS={mos:.1f}% ≥ 20%")
            else:
                checks.append(f"❌ MOS={mos:.1f}% < 20%")
        if div_yield > 0:
            total += 1
            if div_yield >= 3:
                score += 1
                checks.append(f"✅ Cổ tức {div_yield:.1f}% ≥ 3%")
            else:
                checks.append(f"ℹ️ Cổ tức {div_yield:.1f}%")
        if de_ratio > 0:
            total += 1
            if de_ratio < 1:
                score += 1
                checks.append(f"✅ D/E={de_ratio:.2f} < 1")
            else:
                checks.append(f"⚠️ D/E={de_ratio:.2f} ≥ 1")
        if roe > 0:
            total += 1
            if roe >= 15:
                score += 1
                checks.append(f"✅ ROE={roe:.1f}% ≥ 15%")
            else:
                checks.append(f"⚠️ ROE={roe:.1f}% < 15%")
        if not is_bank(selected_ticker) and gross_margin > 0:
            total += 1
            if gross_margin >= 20:
                score += 1
                checks.append(f"✅ Biên gộp {gross_margin:.1f}% ≥ 20%")
            else:
                checks.append(f"ℹ️ Biên gộp {gross_margin:.1f}% < 20%")
        if pe > 0:
            total += 1
            if pe <= 15:
                score += 1
                checks.append(f"✅ P/E={pe:.1f} ≤ 15")
            else:
                checks.append(f"⚠️ P/E={pe:.1f} > 15")
        if pb > 0:
            total += 1
            if pb <= 1.5:
                score += 1
                checks.append(f"✅ P/B={pb:.2f} ≤ 1.5")
            else:
                checks.append(f"ℹ️ P/B={pb:.2f} > 1.5")
        growth_used = max(growth_rev, growth_profit)
        if growth_used > 0:
            total += 1
            if growth_used >= 10:
                score += 1
                checks.append(f"✅ Tăng trưởng {growth_used:.1f}% ≥ 10%")
            else:
                checks.append(f"ℹ️ Tăng trưởng {growth_used:.1f}%")
        if total == 0:
            st.info("Không đủ dữ liệu đánh giá")
        else:
            st.markdown(f"#### 🎯 ĐÁNH GIÁ: **{score}/{total} SAO** " + "⭐"*score)
            for c in checks:
                st.markdown(f"- {c}")

        # Slider
        st.markdown("---")
        st.markdown("## 🎛️ Mô phỏng Kịch bản Định giá")
        default_g, default_r, default_mos = 12.0, 12.0, 25
        for k, v in [('slider_g', default_g), ('slider_r', default_r), ('slider_mos', default_mos)]:
            if k not in st.session_state:
                st.session_state[k] = v
        col1, col2, col3 = st.columns(3)
        with col1:
            g = st.slider("📈 Tăng trưởng (g)", 2.0, 25.0, st.session_state.slider_g, 0.5, key="g_slider")
        with col2:
            r = st.slider("📊 Chiết khấu (r)", 8.0, 18.0, st.session_state.slider_r, 0.5, key="r_slider")
        with col3:
            mos = st.slider("🛡️ Biên an toàn (MoS)", 10, 50, st.session_state.slider_mos, 5, key="mos_slider")
        st.session_state.slider_g, st.session_state.slider_r, st.session_state.slider_mos = g, r, mos

        if r > g:
            intrinsic_sim = eps * (1 + g/100) / ((r - g) / 100)
        else:
            intrinsic_sim = eps * 15
        safe_price = intrinsic_sim * (1 - mos/100)
        intrinsic_sim = round_float(intrinsic_sim, 0)
        safe_price = round_float(safe_price, 0)

        if price > 0:
            diff = (intrinsic_sim - price) / price * 100
            if price < safe_price:
                rec = "RẤT HẤP DẪN / MUA"
                cls = "rec-buy"
            elif safe_price <= price <= intrinsic_sim:
                rec = "ĐỊNH GIÁ HỢP LÝ / THEO DÕI"
                cls = "rec-hold"
            else:
                rec = "ĐẮT / CÂN NHẮC BÁN"
                cls = "rec-sell"
        else:
            rec, cls, diff = "KHÔNG CÓ DỮ LIỆU GIÁ", "rec-hold", 0.0

        colA, colB, colC, colD = st.columns(4)
        colA.metric("💰 Giá trị thực", f"{intrinsic_sim:,.0f}đ")
        colB.metric("🛡️ Giá mua an toàn", f"{safe_price:,.0f}đ")
        colC.metric("📊 Chênh lệch", f"{diff:+.1f}%")
        colD.markdown(f"<div style='text-align:center;'><span class='recommend-badge {cls}'>{rec}</span></div>", unsafe_allow_html=True)

    with tabs[2]:  # DỰ BÁO
        st.markdown("### 📌 Dự báo tăng trưởng dài hạn")
        if not df.empty:
            row = df.iloc[0]
            rev_col = find_column_by_keywords(df, ['doanh thu'])
            prof_col = find_column_by_keywords(df, ['lợi nhuận sau thuế'])
            div_col = find_column_by_keywords(df, ['cổ tức', 'cash_dividend'])
            col1, col2, col3 = st.columns(3)
            with col1:
                rev_col_name = st.selectbox("Doanh thu", df.columns, index=list(df.columns).index(rev_col) if rev_col else 0)
            with col2:
                prof_col_name = st.selectbox("Lợi nhuận", df.columns, index=list(df.columns).index(prof_col) if prof_col else 0)
            with col3:
                div_col_name = st.selectbox("Cổ tức (VNĐ/cp)", df.columns, index=list(df.columns).index(div_col) if div_col else 0)
            base_rev = safe_float(row[rev_col_name])
            base_prof = safe_float(row[prof_col_name])
            base_div = safe_float(row[div_col_name]) if div_col_name else 0.0
            if base_div < 0: base_div = 0.0

            g_rate = st.number_input("Tăng trưởng (%/năm)", value=12.0, step=1.0) / 100.0
            years = st.slider("Số năm", 3, 10, 5)
            if base_div == 0:
                st.info("Chưa có dữ liệu cổ tức từ BCTC gốc")

            years_lab = [f"Năm {i}" for i in range(years+1)]
            rev_proj = [base_rev * ((1+g_rate)**i) for i in range(years+1)]
            prof_proj = [base_prof * ((1+g_rate)**i) for i in range(years+1)]
            eps = safe_float(row.get(find_column_by_keywords(df, ['eps']), 0))
            shares_out = base_prof / eps if eps > 0 and base_prof > 0 else 1
            if base_div > 0:
                div_proj = [max(base_div * ((1+g_rate)**i), 0.0) for i in range(years+1)]
            else:
                div_proj = [0.0] * (years+1)
            rev_proj = [round_float(v, 0) for v in rev_proj]
            prof_proj = [round_float(v, 0) for v in prof_proj]
            div_proj = [round_float(v, 0) for v in div_proj]

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            fig.add_trace(go.Bar(x=years_lab, y=rev_proj, name=rev_col_name, marker_color="#3B82F6"))
            fig.add_trace(go.Scatter(x=years_lab, y=prof_proj, name=prof_col_name, line=dict(color="#22C55E", width=3), mode="lines+markers"))
            fig.add_trace(go.Bar(x=years_lab, y=div_proj, name="Cổ tức", marker_color="#F59E0B"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            df_forecast = pd.DataFrame({
                "Năm": years_lab,
                rev_col_name: rev_proj,
                prof_col_name: prof_proj,
                "Cổ tức (VNĐ/cp)": div_proj,
            })
            st.dataframe(df_forecast, use_container_width=True)
        else:
            st.info("Không có dữ liệu dự báo")

    with tabs[3]:  # TỔNG QUAN
        st.subheader(f"📊 TỔNG QUAN DOANH NGHIỆP: {selected_ticker}")
        if not df.empty:
            comp_name, ind_name = get_company_and_industry(selected_ticker, row_data, df)
            st.markdown(f"**Tên:** {comp_name}")
            st.markdown(f"**Ngành:** {ind_name}")
            metrics = extract_all_metrics(row_data, df, selected_ticker)
            if metrics:
                groups = {
                    'Hiệu quả': ['ROE', 'ROA', 'Biên LN gộp', 'Biên LN ròng'],
                    'Định giá': ['EPS', 'BVPS', 'P/E', 'P/B'],
                    'An toàn': ['Nợ/VCSH', 'Tỷ suất cổ tức'],
                    'Quy mô': ['Doanh thu', 'Lợi nhuận', 'Tổng tài sản', 'Vốn chủ sở hữu', 'Vốn hóa']
                }
                for gname, keys in groups.items():
                    available = {k: v for k, v in metrics.items() if k in keys}
                    if available:
                        st.markdown(f"**{gname}**")
                        cols = st.columns(min(len(available), 4))
                        for idx, (key, val) in enumerate(available.items()):
                            with cols[idx % 4]:
                                if key in ['ROE', 'ROA', 'Biên LN gộp', 'Biên LN ròng', 'Tỷ suất cổ tức']:
                                    st.metric(key, format_percent(val))
                                elif key in ['P/E', 'P/B']:
                                    st.metric(key, format_pe_pb(val))
                                elif key in ['Doanh thu', 'Lợi nhuận', 'Tổng tài sản', 'Vốn chủ sở hữu', 'Vốn hóa']:
                                    st.metric(key, format_vn_currency(val))
                                else:
                                    st.metric(key, format_eps(val) if 'EPS' in key or 'BVPS' in key else f"{val:,.0f}")
                        st.markdown("---")
            else:
                st.warning("Không có chỉ số nào")
        else:
            st.info("Không có dữ liệu")

    with tabs[4]:  # TRA CỨU
        st.subheader("🔎 Tra cứu nhanh")
        search = st.text_input("Nhập mã CP:").upper()
        if search:
            if data_source == "📂 File CSV" and ticker_col:
                mask = df[ticker_col].astype(str).str.upper() == search
                if mask.any():
                    r = df[mask].iloc[0]
                    st.success(f"Tìm thấy {search}")
                    st.json({col: r[col] for col in df.columns[:10]})
                else:
                    st.error("Không tìm thấy")
            else:
                price_api, mcap, pe_api, pb_api, roe_api, inc, bs, src, _ = get_stock_data_bulletproof(search)
                if not inc.empty:
                    st.dataframe(inc, use_container_width=True)
                else:
                    st.error("Không tìm thấy qua API")

    with tabs[5]:  # ML
        st.subheader(f"🤖 PHÂN TÍCH SỨC KHỎE TÀI CHÍNH BẰNG AI")
        if not df.empty:
            roe_ml = safe_float(row_data.get(find_column_by_keywords(df, ['roe']), 0))
            roa_ml = safe_float(row_data.get(find_column_by_keywords(df, ['roa']), 0))
            margin_ml = safe_float(row_data.get(find_column_by_keywords(df, ['biên lợi nhuận gộp']), 0))
            de_ml = safe_float(row_data.get(find_column_by_keywords(df, ['nợ/vcsh', 'd/e']), 0))
            eps_ml = fin['eps']
            bvps_ml = fin['bvps']
            price_ml = fin['price']

            risk_label, conf = predict_risk(roe_ml, roa_ml, margin_ml, de_ml)
            status_map = {2: ("💎 Xuất sắc", "#22C55E"), 1: ("⚖️ An toàn", "#3B82F6"), 0: ("⚠️ Rủi ro", "#EF4444")}
            label, color = status_map.get(risk_label, ("Không xác định", "#94A3B8"))
            st.markdown(f"""
            <div style="background:#1E293B; padding:16px; border-left:5px solid {color}; border-radius:8px;">
                <h4 style="color:{color};">{label}</h4>
                <p style="color:#94A3B8;">Độ tin cậy: {conf}%</p>
            </div>
            """, unsafe_allow_html=True)

            ml_eps = predict_eps(roe_ml, roa_ml, margin_ml, de_ml)
            st.metric("EPS dự báo (Ensemble)", format_eps(ml_eps))

            hybrid_val = hybrid_valuation(price_ml, eps_ml, bvps_ml, roe_ml, margin_ml, de_ml, sector)
            col1, col2, col3 = st.columns(3)
            col1.metric("Giá hiện tại", format_price_vnd(price_ml))
            col2.metric("Graham cơ bản", format_price_vnd((22.5*eps_ml*bvps_ml)**0.5 if eps_ml>0 and bvps_ml>0 else 0))
            col3.metric("Hybrid + Ensemble", format_price_vnd(hybrid_val) if hybrid_val else "N/A")
            if hybrid_val and price_ml > 0:
                mos_hybrid = (hybrid_val - price_ml) / hybrid_val * 100
                st.metric("Biên an toàn Hybrid", f"{mos_hybrid:.1f}%")
                if mos_hybrid >= 20:
                    st.success("✅ Vùng giá hợp lý")
                elif mos_hybrid >= 10:
                    st.info("ℹ️ Trung bình")
                else:
                    st.warning("⚠️ Biên an toàn thấp")
        else:
            st.info("Không có dữ liệu")

    with tabs[6]:  # DL
        st.subheader(f"🧠 DEEP LEARNING & ENSEMBLE - DỰ BÁO XU HƯỚNG")
        if not df.empty:
            row = df.iloc[0]
            roe_dl = safe_float(row.get(find_column_by_keywords(df, ['roe']), 0))
            roa_dl = safe_float(row.get(find_column_by_keywords(df, ['roa']), 0))
            margin_dl = safe_float(row.get(find_column_by_keywords(df, ['biên lợi nhuận gộp']), 0))
            de_dl = safe_float(row.get(find_column_by_keywords(df, ['nợ/vcsh', 'd/e']), 0))
            eps_dl = fin['eps']
            revenue_dl = safe_float(row.get(find_column_by_keywords(df, ['doanh thu']), 0))
            profit_dl = safe_float(row.get(find_column_by_keywords(df, ['lợi nhuận sau thuế']), 0))

            growth_rev = safe_float(row.get(find_column_by_keywords(df, ['tăng trưởng doanh thu']), 0))
            growth_profit = safe_float(row.get(find_column_by_keywords(df, ['tăng trưởng lợi nhuận']), 0))
            expected_g = (growth_rev + roe_dl/5) / 2 if growth_rev > 0 else max(roe_dl/5, 5)

            quarters = [f"Q{i+1}" for i in range(4)]
            rev_forecast = forecast_trend(revenue_dl, expected_g, 4)
            prof_forecast = forecast_trend(profit_dl, expected_g, 4)
            eps_forecast = forecast_trend(eps_dl, expected_g, 4)

            df_trend = pd.DataFrame({
                "Quý": quarters,
                "Doanh thu": [format_vn_currency(v) for v in rev_forecast],
                "Lợi nhuận": [format_vn_currency(v) for v in prof_forecast],
                "EPS": [format_eps(v) for v in eps_forecast]
            })
            st.dataframe(df_trend, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=quarters, y=rev_forecast, name="Doanh thu", line=dict(color="#3B82F6")))
            fig.add_trace(go.Scatter(x=quarters, y=prof_forecast, name="Lợi nhuận", line=dict(color="#22C55E")))
            fig.add_trace(go.Scatter(x=quarters, y=eps_forecast, name="EPS", line=dict(color="#F59E0B"), yaxis="y2"))
            fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu")

    with tabs[7]:  # REAL-TIME
        render_real_time(selected_ticker)

    # ===== FOOTER & DISCLAIMER =====
    st.markdown("---")
    st.markdown(
        """
    <div class="zalo-contact-bar">
        <div class="label">📱 Kết nối với chúng tôi</div>
        <div>
            <span>💬 Zalo</span>
            <a href="https://zalo.me/0327625853" target="_blank" class="number">0327 625 853</a>
        </div>
    </div>
    """,
        unsafe_allow_html=True
    )

    with st.expander("ℹ️ GIỚI THIỆU CÔNG NGHỆ, GÓI VIP & TUYÊN BỐ PHÁP LÝ"):
        st.markdown("""
        **🚀 Công nghệ:** FINEX VN Terminal – Big Data Processing, Graham & Fisher valuation, Ensemble ML/DL.
        **👨‍💻 Nhà sáng lập:** Trần Anh Quân.
        **💎 VIP 299K/tháng:** Mở khóa full BCTC làm sạch, API realtime, hỗ trợ 1-1.
        """)
        st.markdown("""
        <div style="background:rgba(239,68,68,0.1); border-left:4px solid #EF4444; padding:12px 16px; border-radius:4px; font-size:0.85em; color:#CBD5E1;">
        <strong>⚠️ TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM:</strong> Công cụ chỉ hỗ trợ phân tích dữ liệu công khai, không đưa ra khuyến nghị đầu tư. Mọi quyết định giao dịch là trách nhiệm cá nhân.
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# CHẠY APP
# =============================================================================
if __name__ == '__main__':
    main()
