import streamlit as st
from utils import safe_float, format_vn_currency

def render_quick_info(df_row):
    """Hiển thị Tab Thông tin nhanh - Không tràn chữ, Không cố định giá 25k"""
    # Tính giá động từ Vốn hóa / Số lượng cổ phiếu nếu thiếu cột giá
    price = safe_float(df_row.get('close_price', df_row.get('price', 0)))
    market_cap = safe_float(df_row.get('market_cap', 0))
    shares = safe_float(df_row.get('shares_outstanding', 0))
    
    if price == 0 and market_cap > 0 and shares > 0:
        price = market_cap / shares

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Giá", f"{price:,.0f} VNĐ" if price > 0 else "N/A")
    with col2:
        st.metric("Vốn hóa", format_vn_currency(market_cap) if market_cap > 0 else "N/A")
    with col3:
        pe = safe_float(df_row.get('pe', 'N/A'))
        st.metric("P/E", f"{pe:.2f}" if pe > 0 else "N/A")
    with col4:
        pb = safe_float(df_row.get('pb', 'N/A'))
        st.metric("P/B", f"{pb:.2f}" if pb > 0 else "N/A")
    with col5:
        roe = safe_float(df_row.get('roe', 'N/A'))
        st.metric("ROE", f"{roe:.2f}%" if roe != 0 else "N/A")
