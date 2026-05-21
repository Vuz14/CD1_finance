"""
Train wrapper script - routes to appropriate training function
"""
import sys
import json
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'Model type not specified'}))
        sys.exit(1)
    
    model_type = sys.argv[1]
    params_json = sys.argv[2] if len(sys.argv) > 2 else '{}'
    
    try:
        params = json.loads(params_json)
    except:
        params = {}
    
    try:
        # Import and call appropriate training function
        if model_type == 'ann':
            from train_ann import train_ann
            # Only override if explicitly provided
            kwargs = {'save_dir': 'results/ann'}
            if 'dropout' in params:
                kwargs['dropout'] = params['dropout']
            if 'batchnorm' in params:
                kwargs['batchnorm'] = params['batchnorm']
            if 'lr' in params:
                kwargs['lr'] = params['lr']
            if 'weight_decay' in params:
                kwargs['weight_decay'] = params['weight_decay']
            if 'n_trials' in params:
                kwargs['n_trials'] = params['n_trials']
            if 'timeout' in params:
                kwargs['timeout'] = params['timeout']
            if 'use_optuna' in params:
                kwargs['use_optuna'] = params['use_optuna']
            result = train_ann(**kwargs)
        elif model_type == 'lr':
            from train_lr import train_lr
            result = train_lr(
                save_dir='results/lr',
                C=params.get('C'),
                penalty=params.get('penalty')
            )
        elif model_type == 'xgb':
            from train_xgb import train_xgb
            result = train_xgb(
                save_dir='results/xgb',
                n_estimators=params.get('n_estimators'),
                max_depth=params.get('max_depth'),
                learning_rate=params.get('learning_rate'),
                subsample=params.get('subsample'),
                colsample_bytree=params.get('colsample_bytree'),
                use_scale_pos_weight=params.get('use_scale_pos_weight', True)
            )
        elif model_type == 'tree':
            from train_tree import train_tree
            result = train_tree(
                save_dir='results/tree',
                max_depth=params.get('max_depth'),
                min_samples_leaf=params.get('min_samples_leaf'),
                criterion=params.get('criterion')
            )
        else:
            print(json.dumps({'success': False, 'error': f'Unknown model type: {model_type}'}))
            sys.exit(1)
        
        # Success
        print(json.dumps({'success': True, 'model_type': model_type, 'metrics': result if result else {}}))
    
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()
