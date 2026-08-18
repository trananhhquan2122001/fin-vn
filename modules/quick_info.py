import streamlit as st
from utils import safe_float, format_vn_currency

def get_col_value(df_row, possible_names):
    """Quét tự động tên cột trong CSV"""
    if isinstance(df_row, dict):
        for name in possible_names:
            if name in df_row and df_row[name] is not None:
                val = safe_float(df_row[name])
                if val != 0:
                    return val
    return 0.0

def render_quick_info(df_row):
    """Giao diện Thông tin nhanh - Chữ trắng sáng nổi bật trên nền tối"""
    
    # 1. Ép kiểu dữ liệu an toàn
    row_data = dict(df_row) if df_row is not None else {}
    
    # 2. Bóc tách dữ liệu
    price = get_col_value(row_data, ['close_price', 'price', 'gia_dong_cua', 'Gia', 'close', 'match_price'])
    market_cap = get_col_value(row_data, ['market_cap', 'von_hoa', 'VonHoa', 'marketCap'])
    shares = get_col_value(row_data, ['shares_outstanding', 'so_luong_cp', 'so_cp_luu_hanh', 'shares'])
    net_profit = get_col_value(row_data, ['net_profit', 'loi_nhuan_sau_thue', 'LNST', 'profit'])
    equity = get_col_value(row_data, ['equity', 'von_chu_so_huu', 'VCSH', 'total_equity'])
    
    # 3. Tính toán linh hoạt
    if price == 0 and market_cap > 0 and shares > 0:
        price = market_cap / shares
        
    pe = get_col_value(row_data, ['pe', 'P/E', 'PE'])
    pb = get_col_value(row_data, ['pb', 'P/B', 'PB'])
    roe = get_col_value(row_data, ['roe', 'ROE'])
    
    if roe == 0 and equity > 0 and net_profit != 0:
        roe = (net_profit / equity) * 100
    if pe == 0 and price > 0 and net_profit > 0 and shares > 0:
        pe = price / (net_profit / shares)
    if pb == 0 and price > 0 and equity > 0 and shares > 0:
        pb = price / (equity / shares)

    # 4. ÉP MÀU CHỮ SÁNG TRẮNG / VÀNG NỔI BẬT NỀN TỐI (CSS INLINE)
    st.markdown("""
    <style>
        .metric-card {
            background-color: #1e293b;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #334155;
            text-align: center;
        }
        .metric-label {
            color: #94a3b8 !important;
            font-size: 14px;
            font-weight: 500;
        }
        .metric-value {
            color: #ffffff !important;
            font-size: 20px;
            font-weight: bold;
            margin-top: 4px;
        }
        .metric-highlight {
            color: #38bdf8 !important;
            font-size: 20px;
            font-weight: bold;
            margin-top: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Định dạng chuỗi hiển thị
    str_price = f"{price:,.0f} VNĐ" if price > 0 else "N/A"
    str_mcap = format_vn_currency(market_cap) if market_cap > 0 else "N/A"
    str_pe = f"{pe:.2f}" if pe > 0 else "N/A"
    str_pb = f"{pb:.2f}" if pb > 0 else "N/A"
    str_roe = f"{roe:.2f}%" if roe != 0 else "N/A"

    # 5. HIỂN THỊ THẺ SÁNG RÕ
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Giá</div><div class="metric-highlight">{str_price}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Vốn hóa</div><div class="metric-value">{str_mcap}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">P/E</div><div class="metric-value">{str_pe}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">P/B</div><div class="metric-value">{str_pb}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">ROE</div><div class="metric-value">{str_roe}</div></div>', unsafe_allow_html=True)
