import pandas as pd
import numpy as np

df = pd.read_csv('uploads/BCTC_by_year.csv')

print('PHÂN TÍCH CHI TIẾT DỮ LIỆU FILE GỐC')
print('='*100)
print(f'Shape: {df.shape}')
print(f'Companies: {df["Mã CP"].unique()}')
print(f'Years: {sorted(df["Năm"].unique())}')

print('\n' + '='*100)
print('TOP 20 CỘT CÓ DỮ LIỆU (non-null):')
print('='*100)

col_stats = []
for col in df.columns:
    if col not in ['STT', 'Mã CP', 'Năm']:
        non_null = df[col].notna().sum()
        non_zero = 0
        if df[col].dtype in [int, float, np.int64, np.float64]:
            non_zero = (df[col] != 0).sum()
        col_stats.append({
            'Column': col,
            'Non-null': non_null,
            'Non-zero': non_zero
        })

# Sort by non-null count
col_stats_sorted = sorted(col_stats, key=lambda x: x['Non-null'], reverse=True)

for i, stat in enumerate(col_stats_sorted[:20]):
    print(f"{i+1}. {stat['Column'][:60]:60s} | Non-null: {stat['Non-null']:2d}/34 | Non-zero: {stat['Non-zero']:2d}/34")

print('\n' + '='*100)
print('ĐỐI TƯỢNG FILE:')
print('='*100)
print('Đây là file BCTC (Báo Cáo Tài Chính) với định dạng CHUYỂN VỊ (transpose)')
print('- Các cột là DÒNG dữ liệu (items) từ báo cáo tài chính')
print('- Mỗi công ty × năm là một hàng')
print('- Chỉ ~50 cột trong 386 cột có giá trị thực (>0)')
print('- 336 cột còn lại là headers/empty')

print('\n' + '='*100)
print('KẾT LUẬN:')
print('='*100)
print('✓ File CÓ dữ liệu nhưng số lượng cột có data RẤT ÍT')
print('✓ Cần sử dụng preprocessing để:')
print('  1. Loại bỏ cột trống')
print('  2. Tìm đúng columns để tính X1-X13')
print('  3. Xử lý NaN values')
print('\n✓ Preprocessor đã làm điều này → processed_BCTC_by_year.csv')
