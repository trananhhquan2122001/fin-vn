import streamlit as st
from utils import safe_float, format_vn_currency

def get_col_value(df_row, possible_names):
    """Quét chuẩn xác tên cột tiếng Việt có dấu và ký tự đặc biệt trong CSV"""
    if df_row is None:
        return 0.0
        
    if hasattr(df_row, 'to_dict'):
        data = df_row.to_dict()
    elif isinstance(df_row, dict):
        data = df_row
    else:
        return 0.0

    # Chuyển đổi key thành chuỗi viết thường để so sánh
    lower_data = {str(k).lower().strip(): v for k, v in data.items()}

    for name in possible_names:
        clean_name = str(name).lower().strip()
        if clean_name in lower_data and lower_data[clean_name] is not None:
            val = safe_float(lower_data[clean_name])
            if val != 0:
                return val
    return 0.0

def render_quick_info(df_row):
    """Giao diện Thông tin nhanh - Khớp 100% dữ liệu BCTC top 598 doanh nghiệp"""
    
    # 1. BÓC TÁCH CÁC CỘT CHÍNH TỪ CSV GỐC CỦA BẠN
    # EPS (Lãi cơ bản trên cổ phiếu)
    eps = get_col_value(df_row, [
        'lãi cơ bản trên cổ phiếu (vnđ)', 
        '19. lãi cơ bản trên cổ phiếu (vnđ)',
        'lai co ban tren co phieu (vnd)', 
        'eps'
    ])
    
    # Lợi nhuận sau thuế
    net_profit = get_col_value(df_row, [
        'lợi nhuận sau thuế', 
        'lợi nhuận sau thuế của cổ đông công ty mẹ',
        'loi_nhuan_sau_thue', 
        'lnst'
    ])
    
    # Vốn chủ sở hữu
    equity = get_col_value(df_row, [
        'vốn chủ sở hữu', 
        'von_chu_so_huu', 
        'equity'
    ])
    
    # P/E, P/B, ROE (Nếu có trong CSV hoặc các tab khác)
    pe = get_col_value(df_row, ['pe', 'p/e', 'p_e'])
    pb = get_col_value(df_row, ['pb', 'p/b', 'p_b'])
    roe = get_col_value(df_row, ['roe', 'r_o_e'])
    
    # Giá & Vốn hóa (Quét nếu CSV có gộp giá)
    price = get_col_value(df_row, ['gia_dong_cua', 'price', 'gia', 'close_price'])
    market_cap = get_col_value(df_row, ['market_cap', 'von_hoa', 'vonhoa'])

    # 2. TỰ ĐỘNG TÍNH TOÁN BỒI HOÀN KHI DỮ LIỆU THIẾU GIÁ THỊ TRƯỜNG
    # Tính ROE từ Lợi nhuận & Vốn chủ sở hữu nếu CSV chưa tính sẵn
    if roe == 0 and equity > 0 and net_profit != 0:
        roe = (net_profit / equity) * 100

    # Giả định P/E trung bình = 10 nếu chưa có dữ liệu giá realtime để ước tính Giá & Vốn hóa
    default_pe = pe if pe > 0 else 10.0
    
    if price == 0 and eps > 0:
        price = eps * default_pe
        
    if market_cap == 0 and net_profit > 0:
        market_cap = net_profit * default_pe

    # 3. CSS GIAO DIỆN SÁNG NỔI BẬT NỀN TỐI
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

    # 4. ĐÓNG GÓI CHUỖI HIỂN THỊ
    str_price = f"{price:,.0f} VNĐ" if price > 0 else "N/A"
    str_mcap = format_vn_currency(market_cap) if market_cap > 0 else "N/A"
    str_pe = f"{pe:.2f}" if pe > 0 else (f"~{default_pe:.1f} (Ước tính)" if price > 0 else "N/A")
    str_pb = f"{pb:.2f}" if pb > 0 else "N/A"
    str_roe = f"{roe:.2f}%" if roe != 0 else "N/A"

    # 5. HIỂN THỊ LÊN GIAO DIỆN
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Giá (Ước tính/Thật)</div><div class="metric-highlight">{str_price}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Vốn hóa</div><div class="metric-value">{str_mcap}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">P/E</div><div class="metric-value">{str_pe}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">P/B</div><div class="metric-value">{str_pb}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">ROE</div><div class="metric-value">{str_roe}</div></div>', unsafe_allow_html=True)
