import numpy as np

def safe_float(val, default=0.0):
    """Tránh lỗi TypeError/Crash khi dữ liệu là N/A, None hoặc String"""
    if val is None or val == "N/A":
        return default
    try:
        if isinstance(val, str):
            val = val.replace(',', '').replace('VNĐ', '').replace('đ', '').strip()
        cleaned_val = float(val)
        return cleaned_val if not (np.isnan(cleaned_val) or np.isinf(cleaned_val)) else default
    except (ValueError, TypeError):
        return default

def format_vn_currency(val_in_vnd):
    """Quy đổi đơn vị linh hoạt ra Tỷ đồng hoặc Nghìn tỷ đồng"""
    val = safe_float(val_in_vnd)
    if abs(val) >= 1e12:
        return f"{val / 1e12:,.2f} Nghìn tỷ"
    elif abs(val) >= 1e9:
        return f"{val / 1e9:,.2f} Tỷ"
    return f"{val:,.0f} VNĐ"
