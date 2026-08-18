import streamlit as st
from utils import safe_float, format_vn_currency

def get_col_value(df_row, possible_names):
    """Hàm quét tự động tất cả các tên cột có thể có trong file CSV"""
    for name in possible_names:
        if name in df_row and df_row[name] is not None:
            val = safe_float(df_row[name])
            if val != 0:
                return val
    return 0.0

def render_quick_info(df_row):
    """Hiển thị Tab Thông tin nhanh - Tự động quét cột & Tính toán chỉ số"""
    
    # 1. Trích xuất các dữ liệu cơ bản từ nhiều tên cột khác nhau
    price = get_col_value(df_row, ['close_price', 'price', 'gia_dong_cua', 'Gia', 'close'])
    market_cap = get_col_value(df_row, ['market_cap', 'von_hoa', 'VonHoa', 'marketCap'])
    shares = get_col_value(df_row, ['shares_outstanding', 'so_luong_cp', 'so_cp_luu_hanh', 'shares'])
    
    net_profit = get_col_value(df_row, ['net_profit', 'loi_nhuan_sau_thue', 'LNST', 'profit'])
    equity = get_col_value(df_row, ['equity', 'von_chu_so_huu', 'VCSH', 'total_equity'])
    
    # 2. TỰ ĐỘNG TÍNH GIÁ nếu thiếu Giá nhưng có Vốn hóa & Số lượng cổ phiếu
    if price == 0 and market_cap > 0 and shares > 0:
        price = market_cap / shares
        
    # 3. TỰ ĐỘNG TÍNH EPS, BVPS, P/E, P/B, ROE nếu CSV không có sẵn
    pe = get_col_value(df_row, ['pe', 'P/E', 'PE'])
    pb = get_col_value(df_row, ['pb', 'P/B', 'PB'])
    roe = get_col_value(df_row, ['roe', 'ROE'])
    
    # Tính toán động nếu CSV thiếu các chỉ số P/E, P/B, ROE
    if roe == 0 and equity > 0 and net_profit != 0:
        roe = (net_profit / equity) * 100
        
    if pe == 0 and price > 0 and net_profit > 0 and shares > 0:
        eps = net_profit / shares
        pe = price / eps if eps > 0 else 0
        
    if pb == 0 and price > 0 and equity > 0 and shares > 0:
        bvps = equity / shares
        pb = price / bvps if bvps > 0 else 0

    # 4. HIỂN THỊ LÊN GIAO DIỆN
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Giá", f"{price:,.0f} VNĐ" if price > 0 else "N/A")
    with col2:
        st.metric("Vốn hóa", format_vn_currency(market_cap) if market_cap > 0 else "N/A")
    with col3:
        st.metric("P/E", f"{pe:.2f}" if pe > 0 else "N/A")
    with col4:
        st.metric("P/B", f"{pb:.2f}" if pb > 0 else "N/A")
    with col5:
        st.metric("ROE", f"{roe:.2f}%" if roe != 0 else "N/A")
