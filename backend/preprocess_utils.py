import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import os


class FinancialIndicatorCalculator:
    """Calculate X1-X13 financial indicators from BCTC data"""
    
    @staticmethod
    def calculate_indicators(df):
        """
        Calculate all financial indicators X1-X13
        
        Args:
            df: DataFrame with financial columns
            
        Returns:
            DataFrame with X1-X13 columns added
        """
        df = df.copy()
        
        # Initialize X columns
        for i in range(1, 14):
            df[f'X{i}'] = 0.0
        
        # Find financial columns by keyword matching
        fin_cols = FinancialIndicatorCalculator._find_columns(df)
        
        print("\n" + "="*80)
        print("📊 TÌM KIẾM CÁC CỘT TÀI CHÍNH:")
        print("="*80)
        for key, col in fin_cols.items():
            status = "✓" if col else "✗"
            print(f"  {status} {key:25s}: {col}")
        print("="*80 + "\n")
        
        # DEBUG: Print sample values from found columns
        print("📍 SAMPLE VALUES FROM FOUND COLUMNS (Row 0):")
        print("="*80)
        if fin_cols['revenue']:
            val = df.iloc[0][fin_cols['revenue']]
            print(f"  revenue ({fin_cols['revenue']}): {val}")
        if fin_cols['current_assets']:
            val = df.iloc[0][fin_cols['current_assets']]
            print(f"  current_assets ({fin_cols['current_assets']}): {val}")
        if fin_cols['current_liabilities']:
            val = df.iloc[0][fin_cols['current_liabilities']]
            print(f"  current_liabilities ({fin_cols['current_liabilities']}): {val}")
        if fin_cols['total_assets']:
            val = df.iloc[0][fin_cols['total_assets']]
            print(f"  total_assets ({fin_cols['total_assets']}): {val}")
        if fin_cols['inventory']:
            val = df.iloc[0][fin_cols['inventory']]
            print(f"  inventory ({fin_cols['inventory']}): {val}")
        print("="*80 + "\n")
        
        # Process each row
        for idx in df.index:
            try:
                # Extract values
                revenue = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['revenue'], 1)
                cogs = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['cogs'], 0)
                gross_profit = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['gross_profit'], 0)
                
                if gross_profit == 0 and revenue != 0:
                    gross_profit = revenue - cogs
                
                pretax_income = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['pretax_income'], 0)
                total_assets = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['total_assets'], 1)
                equity = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['equity'], 1)
                total_liabilities = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['total_liabilities'], 0)
                current_assets = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['current_assets'], 0)
                current_liabilities = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['current_liabilities'], 1)
                inventory = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['inventory'], 0)
                cash = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['cash'], 0)
                receivables = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['receivables'], 0)
                interest_expense = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['interest_expense'], 0)
                depreciation = FinancialIndicatorCalculator._get_value(df, idx, fin_cols['depreciation'], 0)
                
                # Fallback
                if total_liabilities == 0:
                    total_liabilities = max(0, total_assets - equity)
                
                # DEBUG: Print first row values
                if idx == 0:
                    print("\n📍 DEBUG ROW 0 VALUES:")
                    print(f"  revenue={revenue}, cogs={cogs}, gross_profit={gross_profit}")
                    print(f"  pretax_income={pretax_income}, total_assets={total_assets}, equity={equity}")
                    print(f"  total_liabilities={total_liabilities}, current_assets={current_assets}")
                    print(f"  current_liabilities={current_liabilities}, inventory={inventory}, cash={cash}")
                
                # Calculate X1-X13 based on the formula table
                # X1 = Lợi nhuận gộp / Doanh thu thuần
                df.at[idx, 'X1'] = gross_profit / revenue if revenue != 0 else 0
                
                # X2 = Thu nhập trước thuế / Doanh thu thuần
                df.at[idx, 'X2'] = pretax_income / revenue if revenue != 0 else 0
                
                # X3 = Thu nhập trước thuế / Tổng tài sản
                df.at[idx, 'X3'] = pretax_income / total_assets if total_assets != 0 else 0
                
                # X4 = Thu nhập trước thuế / Vốn chủ sở hữu
                df.at[idx, 'X4'] = pretax_income / equity if equity != 0 else 0
                
                # X5 = Tổng nợ phải trả / Tổng tài sản
                df.at[idx, 'X5'] = total_liabilities / total_assets if total_assets != 0 else 0
                
                # X6 = Tài sản ngắn hạn / Nợ ngắn hạn
                df.at[idx, 'X6'] = current_assets / current_liabilities if current_liabilities != 0 else 0
                
                # X7 = (Tài sản ngắn hạn - Hàng tồn kho) / Nợ ngắn hạn
                df.at[idx, 'X7'] = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else 0
                
                # X8 = (Pre-tax Income + Interest Expense) / Interest Expense
                # Interest Coverage Ratio
                if interest_expense != 0:
                    ebit = pretax_income + interest_expense
                    df.at[idx, 'X8'] = ebit / interest_expense
                else:
                    df.at[idx, 'X8'] = 0
                
                # X9 = (Pre-tax Income + Interest Expense + Depreciation) / Total Debt
                # Debt Service Coverage Ratio
                if total_liabilities != 0:
                    debt_service = pretax_income + interest_expense + depreciation
                    df.at[idx, 'X9'] = debt_service / total_liabilities
                else:
                    df.at[idx, 'X9'] = 0
                
                # X10 = Tiền và các khoản tương đương tiền / Nợ ngắn hạn
                df.at[idx, 'X10'] = cash / current_liabilities if current_liabilities != 0 else 0
                
                # X11 = Giá vốn hàng bán / Hàng tồn kho bình quân
                # Inventory Turnover Ratio
                if fin_cols['inventory'] and inventory != 0:
                    df.at[idx, 'X11'] = cogs / inventory if inventory != 0 else 0
                else:
                    df.at[idx, 'X11'] = 0
                
                # X12 = Phải thu / Doanh thu bình quân (receivables turnover)
                # Accounts Receivable Turnover
                if fin_cols['receivables'] and receivables != 0:
                    df.at[idx, 'X12'] = revenue / receivables if receivables != 0 else 0
                else:
                    df.at[idx, 'X12'] = 0
                
                # X13 = Tổng doanh thu / Tổng tài sản
                df.at[idx, 'X13'] = revenue / total_assets if total_assets != 0 else 0
                
            except Exception as e:
                print(f"⚠️  Error at row {idx}: {str(e)}")
                for i in range(1, 14):
                    df.at[idx, f'X{i}'] = 0.0
        
        return df
    
    @staticmethod
    def _find_columns(df):
        """Find financial columns by keyword matching - for companies (banking or retail/manufacturing)"""
        cols = {}
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Revenue - Try both banking and retail/manufacturing patterns
            if not cols.get('revenue'):
                if 'thu nhập lãi thuần' in col_lower:  # Banking: interest income
                    cols['revenue'] = col
                elif 'doanh thu thuần' in col_lower or 'tổng doanh thu' in col_lower:  # Retail: net revenue
                    cols['revenue'] = col
            
            # COGS - Chi phí lãi (interest) OR Giá vốn hàng bán (COGS)
            if not cols.get('cogs'):
                if 'giá vốn hàng bán' in col_lower:  # Retail: COGS
                    cols['cogs'] = col
                elif 'chi phí lãi' in col_lower and 'tương tự' in col_lower:  # Banking: interest expense
                    cols['cogs'] = col
            
            # Gross Profit - "Lợi nhuận từ HDKD trước chi phí dự phòng"
            if not cols.get('gross_profit') and 'lợi nhuận từ hdkd' in col_lower and 'trước chi phí dự phòng' in col_lower:
                cols['gross_profit'] = col
            
            # Pre-tax Income - "Tổng lợi nhuận trước thuế"
            if not cols.get('pretax_income') and 'tổng lợi nhuận trước thuế' in col_lower:
                cols['pretax_income'] = col
            
            # Total Assets - "TỔNG CỘNG TÀI SẢN"
            if not cols.get('total_assets') and 'tổng cộng tài sản' in col_lower:
                cols['total_assets'] = col
            
            # Equity - "VIII. Vốn chủ sở hữu"
            if not cols.get('equity') and col.strip().startswith('VIII.') and 'vốn chủ sở hữu' in col_lower:
                cols['equity'] = col
            
            # Total Liabilities - "TỔNG NỢ PHẢI TRẢ" (excluding vốn chủ)
            if not cols.get('total_liabilities') and 'tổng nợ phải trả' in col_lower and 'vốn chủ' not in col_lower:
                cols['total_liabilities'] = col
            
            # Current Assets - Try multiple patterns
            if not cols.get('current_assets'):
                if col.strip().startswith('III.') and 'tiền gửi khách hàng' in col_lower:  # Banking
                    cols['current_assets'] = col
                elif 'a. tài sản ngắn hạn' in col_lower or 'tài sản lưu động' in col_lower:  # Retail
                    cols['current_assets'] = col
            
            # Current Liabilities - Try multiple patterns
            if not cols.get('current_liabilities'):
                if col.strip().startswith('VII.') and 'các khoản nợ' in col_lower:  # Banking
                    cols['current_liabilities'] = col
                elif col.strip().startswith('I.') and 'nợ phải trả ngắn hạn' in col_lower:  # Retail/Banking alternative
                    cols['current_liabilities'] = col
            
            # Inventory - "1. Hàng tồn kho" (for retail/manufacturing companies)
            if not cols.get('inventory') and '1. hàng tồn kho' in col_lower:
                cols['inventory'] = col
            
            # Receivables - "1. Các khoản phải thu" (primary for banking) or "1. Phải thu ngắn hạn của khách hàng" (for retail)
            if not cols.get('receivables'):
                if col.strip() == '1. Các khoản phải thu':  # Banking receivables
                    cols['receivables'] = col
                elif 'phải thu' in col_lower and 'khách hàng' in col_lower and 'ngắn hạn' in col_lower:
                    cols['receivables'] = col
            
            # Cash - "I. Tiền mặt..." or alternatives
            if not cols.get('cash') and col.strip().startswith('I.') and 'tiền mặt' in col_lower:
                cols['cash'] = col
            
            # Interest Expense - "Chi phí lãi và các chi phí tương tự" (primary) or "Chi phí lãi vay"
            if not cols.get('interest_expense'):
                if 'chi phí lãi và các chi phí tương tự' in col_lower:  # Banking interest expense (primary)
                    cols['interest_expense'] = col
                elif 'chi phí lãi vay' in col_lower or 'trong đó: chi phí lãi vay' in col_lower:
                    cols['interest_expense'] = col
            
            # Depreciation - "Giá trị hao mòn lũy kế" (accumulated depreciation)
            if not cols.get('depreciation'):
                if 'giá trị hao mòn lũy kế' in col_lower or 'hao mòn' in col_lower:
                    cols['depreciation'] = col
        
        # Fill missing with None
        for key in ['revenue', 'cogs', 'gross_profit', 'pretax_income', 'total_assets', 'equity', 'total_liabilities', 'current_assets', 'current_liabilities', 'inventory', 'cash', 'receivables', 'interest_expense', 'depreciation']:
            if key not in cols:
                cols[key] = None
        
        return cols
    
    @staticmethod
    def _get_value(df, idx, col_name, default=0):
        """Get value from DataFrame safely"""
        if col_name is None or col_name not in df.columns:
            return default
        try:
            val = df.at[idx, col_name]
            if pd.isna(val):
                return default
            return float(val)
        except:
            return default


class DataPreprocessor:
    """Preprocessing pipeline: imputation, outlier handling, scaling"""
    
    @staticmethod
    def preprocess(df):
        """
        Complete preprocessing pipeline
        
        Args:
            df: DataFrame with X1-X13 columns
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        # Only get X1-X13 columns (not other X* columns from original file)
        x_cols = [col for col in df.columns if col in [f'X{i}' for i in range(1, 14)]]
        
        if not x_cols:
            return df
        
        # Step 1: Handle missing values with KNN imputation
        print("🔧 Step 1: Handling missing values with KNN imputation...")
        df = DataPreprocessor._impute_missing(df, x_cols)
        
        # Step 2: Handle outliers with IQR method
        print("🔧 Step 2: Handling outliers with IQR method...")
        df = DataPreprocessor._handle_outliers(df, x_cols)
        
        # Step 3: Winsorize
        print("🔧 Step 3: Winsorizing outliers...")
        df = DataPreprocessor._winsorize(df, x_cols)
        
        # Step 4: Scale features
        print("🔧 Step 4: Scaling features with StandardScaler...")
        df = DataPreprocessor._scale_features(df, x_cols)
        
        return df
    
    @staticmethod
    def _impute_missing(df, x_cols):
        """Handle missing values"""
        df = df.copy()
        
        # Check if there are any missing values
        if df[x_cols].isnull().sum().sum() == 0:
            print("   No missing values found")
            return df
        
        print(f"   Found {df[x_cols].isnull().sum().sum()} missing values")
        print(f"   X columns before impute: {len(x_cols)} -> {x_cols}")
        
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        imputed_values = imputer.fit_transform(df[x_cols])
        
        print(f"   Imputed array shape: {imputed_values.shape}")
        print(f"   X columns to assign: {len(x_cols)}")
        
        # Assign back safely - ensure we only assign to available columns
        for i, col in enumerate(x_cols):
            if i < imputed_values.shape[1]:
                df[col] = pd.Series(imputed_values[:, i], index=df.index)
            else:
                print(f"   WARNING: Column {col} index {i} out of bounds!")
        
        return df
    
    @staticmethod
    def _handle_outliers(df, x_cols):
        """Handle outliers using IQR method"""
        df = df.copy()
        
        for col in x_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
    
    @staticmethod
    def _winsorize(df, x_cols):
        """Winsorize outliers at 5th and 95th percentiles"""
        df = df.copy()
        
        for col in x_cols:
            p5 = df[col].quantile(0.05)
            p95 = df[col].quantile(0.95)
            df[col] = df[col].clip(lower=p5, upper=p95)
        
        return df
    
    @staticmethod
    def _scale_features(df, x_cols):
        """Scale features with StandardScaler"""
        df = df.copy()
        
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df[x_cols])
        
        # Assign back as DataFrame to preserve index and column names
        scaled_df = pd.DataFrame(scaled_array, columns=x_cols, index=df.index)
        df[x_cols] = scaled_df
        
        return df


def process_raw_data(file_path, output_path=None):
    """
    Process raw BCTC data and create preprocessed dataset
    
    Args:
        file_path: Path to raw data file (.xlsx or .csv)
        output_path: Path to save processed file (optional)
        
    Returns:
        Processed DataFrame
    """
    
    # Load file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    print(f"\n📂 Loaded file: {file_path}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Print all columns
    print("\n" + "="*80)
    print(f"📋 CÁC CỘT TRONG FILE GỐC ({len(df.columns)} cột):")
    print("="*80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")
    print("="*80 + "\n")
    
    # Rename metadata columns
    rename_map = {}
    if 'Mã CP' in df.columns:
        rename_map['Mã CP'] = 'MaCP'
    if 'Năm' in df.columns:
        rename_map['Năm'] = 'Year'
    
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        for old, new in rename_map.items():
            print(f"✓ Renamed '{old}' → '{new}'")
    
    # Create STT if not exists
    if 'STT' not in df.columns:
        df.insert(0, 'STT', range(1, len(df) + 1))
    
    # Calculate indicators
    print("🔢 Calculating X1-X13 financial indicators...")
    df = FinancialIndicatorCalculator.calculate_indicators(df)
    
    # Preprocess
    print("\n📊 Applying preprocessing pipeline...")
    df = DataPreprocessor.preprocess(df)
    
    # Keep only important columns
    important_cols = ['STT', 'MaCP', 'Year'] + [f'X{i}' for i in range(1, 14)]
    available_cols = [col for col in important_cols if col in df.columns]
    df = df[available_cols]
    
    print(f"\n✅ Processing complete!")
    print(f"   Output shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {', '.join(df.columns.tolist())}")
    
    # Save
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"   💾 Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Test
    test_file = 'uploads/BCTC_by_year.csv'
    if os.path.exists(test_file):
        result = process_raw_data(test_file, 'test_output.csv')
        print("\n📊 Sample data (first 5 rows):")
        print(result.head())
    else:
        print(f"❌ File not found: {test_file}")
