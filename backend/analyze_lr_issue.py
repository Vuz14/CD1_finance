import pandas as pd
import os

print('='*70)
print('ANALYZING WHY LR PREDICTS 0% FOR SOME COMPANIES')
print('='*70)

# Check training data
if os.path.exists('uploads/Dataset_CD1_preprocessed.csv'):
    train = pd.read_csv('uploads/Dataset_CD1_preprocessed.csv')
    print(f'\nTraining data shape: {train.shape}')
    print(f'Columns: {list(train.columns)}')
    
    if 'default_final' in train.columns:
        print('\n📊 DEFAULT DISTRIBUTION BY COMPANY:')
        print('-'*70)
        for company in sorted(train['MaCP'].unique()):
            company_train = train[train['MaCP'] == company]
            total = len(company_train)
            defaults = company_train['default_final'].sum()
            non_defaults = total - defaults
            pct = (defaults / total * 100) if total > 0 else 0
            
            print(f'{company:8s}: Total={total:3d}, Defaults={int(defaults):2d}, Non-default={non_defaults:2d} ({pct:5.1f}%)')
        
        print('\n⚠️  ISSUE: LR learns from data imbalance')
        print('If a company has NO defaults (0%) in training data,')
        print('LR will likely predict 0% probability of default.')
        print('\nSolution needed:')
        print('1. Check if test companies have actual default cases')
        print('2. Consider class balancing or threshold adjustment')
