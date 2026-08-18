import streamlit as st
from utils import safe_float, format_vn_currency

def get_col_value(df_row, possible_names):
    """Tìm giá trị bằng cách kiểm tra cả chữ hoa, chữ thường và dấu gạch dưới"""
    if df_row is None:
        return 0.0
        
    # Chuyển đổi Series hoặc Dict thành dict có key viết thường để so sánh
    if hasattr(df_row, 'to_dict'):
        data = df_row.to_dict()
    elif isinstance(df_row, dict):
        data = df_row
    else:
        return 0.0

    # Tạo map key viết thường
    lower_data = {str(k).lower().strip(): v for k, v in data.items()}

    for name in possible_names:
        clean_name = str(name).lower().strip()
        if clean_name in lower_data and lower_data[clean_name] is not None:
            val = safe_float(lower_data[clean_name])
            if val != 0:
                return val
    return 0.0

def render_quick_info(df_row):
    """Giao diện Thông tin nhanh - Quét đa dạng tên cột CSV"""
    
    # 1. Quét các tên cột phổ biến trong dữ liệu chứng khoán VN
    price = get_col_value(df_row, ['close_price', 'price', 'gia_dong_cua', 'gia', 'close', 'match_price', 'giadongcua', 'gia_khop_lenh'])
    market_cap = get_col_value(df_row, ['market_cap', 'von_hoa', 'vonhoa', 'marketcap', 'von_hoa_thi_truong'])
    shares = get_col_value(df_row, ['shares_outstanding', 'so_luong_cp', 'so_cp_luu_hanh', 'shares', 'khoi_luong_cp_luu_hanh', 'klcp'])
    net_profit = get_col_value(df_row, ['net_profit', 'loi_nhuan_sau_thue', 'lnst', 'profit', 'loinhuansauthue'])
    equity = get_col_value(df_row, ['equity', 'von_chu_so_huu', 'vcsh', 'total_equity', 'vonchusohuu'])
    
    # 2. Tự động tính toán nếu CSV chỉ có 1 vài chỉ số cơ bản
    if price == 0 and market_cap > 0 and shares > 0:
        price = market_cap / shares
        
    pe = get_col_value(df_row, ['pe', 'p/e', 'p_e'])
    pb = get_col_value(df_row, ['pb', 'p/b', 'p_b'])
    roe = get_col_value(df_row, ['roe', 'r_o_e'])
    
    if roe == 0 and equity > 0 and net_profit != 0:
        roe = (net_profit / equity) * 100
    if pe == 0 and price > 0 and net_profit > 0 and shares > 0:
        pe = price / (net_profit / shares)
    if pb == 0 and price > 0 and equity > 0 and shares > 0:
        pb = price / (equity / shares)

    # 3. CSS GIAO DIỆN SÁNG NỔI BẬT
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

    # Đóng gói hiển thị
    str_price = f"{price:,.0f} VNĐ" if price > 0 else "N/A"
    str_mcap = format_vn_currency(market_cap) if market_cap > 0 else "N/A"
    str_pe = f"{pe:.2f}" if pe > 0 else "N/A"
    str_pb = f"{pb:.2f}" if pb > 0 else "N/A"
    str_roe = f"{roe:.2f}%" if roe != 0 else "N/A"

    # 4. HIỂN THỊ
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
