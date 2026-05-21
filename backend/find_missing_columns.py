import pandas as pd
import numpy as np

df = pd.read_csv('uploads/BCTC_by_year.csv')

print("="*100)
print("TÌM KIẾM CỘT THIẾU CHO X8, X9, X11, X12")
print("="*100)

# Keywords để tìm
search_keywords = {
    'X8 (Interest Expense / Lãi vay)': [
        'chi phí lãi vay',
        'chi phí lãi',
        'lãi vay',
        'interest expense',
        'chi phí tài chính',
        'chi phí lãi vay'
    ],
    'X9 (Depreciation / Khấu hao)': [
        'khấu hao',
        'depreciation',
        'amortization',
        'hao mòn',
        'giá trị hao mòn'
    ],
    'X11 (Inventory / Hàng tồn kho)': [
        'hàng tồn kho',
        'tồn kho',
        'inventory',
        'giá vốn hàng bán'
    ],
    'X12 (Receivables / Phải thu)': [
        'phải thu khách hàng',
        'phải thu',
        'receivables',
        'accounts receivable'
    ]
}

# Tìm kiếm
for category, keywords in search_keywords.items():
    print(f"\n🔍 {category}:")
    print("-" * 100)
    found_any = False
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                non_null = df[col].notna().sum()
                non_zero = (df[col] != 0).sum() if df[col].dtype in [int, float, np.int64, np.float64] else 0
                print(f"  ✓ {col[:70]:70s} | Non-null: {non_null:2d}/34 | Non-zero: {non_zero:2d}/34")
                found_any = True
    
    if not found_any:
        print(f"  ✗ Không tìm thấy")

print("\n" + "="*100)
print("TẤT CẢ CỘT CÓ DỮ LIỆU (Non-null > 0):")
print("="*100)

# List all columns with data
cols_with_data = []
for col in df.columns:
    if col not in ['STT', 'Mã CP', 'Năm']:
        non_null = df[col].notna().sum()
        if non_null > 0:
            non_zero = (df[col] != 0).sum() if df[col].dtype in [int, float, np.int64, np.float64] else 0
            cols_with_data.append((col, non_null, non_zero))

# Sort by non-null count
cols_with_data.sort(key=lambda x: x[1], reverse=True)

print(f"\nTổng cột có data: {len(cols_with_data)}/386")
print("\nTop 50 cột:")
for i, (col, non_null, non_zero) in enumerate(cols_with_data[:50]):
    print(f"{i+1:2d}. {col[:70]:70s} | Non-null: {non_null:2d}/34 | Non-zero: {non_zero:2d}/34")
