"""
Flask backend for ML model training and evaluation
"""
import os
import sys
import io

# Fix UTF-8 encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import subprocess
import joblib

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and parse CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and validate CSV
        df = pd.read_csv(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'shape': list(df.shape),
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'first_rows': df.head().to_dict('records')
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train_model():
    """Train ML model"""
    try:
        data = request.json
        model_type = data.get('model_type')
        params = data.get('params', {})
        
        if model_type not in ['ann', 'lr', 'xgb', 'tree']:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Run training script via wrapper
        import json
        params_json = json.dumps(params)
        
        # Set environment for UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Use the current Python executable (from venv)
        python_exe = sys.executable
        
        result = subprocess.run(
            [python_exe, 'train_wrapper.py', model_type, params_json],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            return jsonify({'error': result.stderr or 'Training failed'}), 500
        
        try:
            # Try to parse the last line as JSON (in case there's other output)
            stdout = result.stdout or ""
            lines = stdout.strip().split('\n')
            train_result = None
            for line in reversed(lines):
                line = line.strip()
                if line and line.startswith('{'):
                    try:
                        train_result = json.loads(line)
                        break
                    except:
                        continue
            
            if not train_result:
                return jsonify({'error': f'Invalid training output: {stdout[:200]}'}), 500
            
            if not train_result.get('success'):
                return jsonify({'error': train_result.get('error', 'Unknown error')}), 500
        except Exception as e:
            return jsonify({'error': f'Invalid training output: {str(e)}'}), 500
        
        # Read results
        results_dir = os.path.join(RESULTS_FOLDER, model_type)
        metrics = load_metrics(results_dir, model_type)
        features = load_feature_importance(results_dir, model_type)
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'metrics': metrics,
            'features': features,
            'history': {  # Empty history for now - could be enhanced later
                'epoch': [],
                'train_loss': [],
                'val_loss': []
            }
        }), 200
    
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Training timeout (took more than 1 hour)'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def load_metrics(results_dir, model_type):
    """Load metrics from results directory"""
    metrics_file = os.path.join(results_dir, 'metrics_test.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}


def load_feature_importance(results_dir, model_type):
    """Load feature importance from results"""
    # Try with model_type suffix first (old format)
    feature_file = os.path.join(results_dir, f'feature_influence_analysis_{model_type}.csv')
    if os.path.exists(feature_file):
        df = pd.read_csv(feature_file)
        return df.head(5).to_dict('records')
    
    # Fallback to generic filename (new format)
    feature_file = os.path.join(results_dir, 'feature_influence_analysis.csv')
    if os.path.exists(feature_file):
        df = pd.read_csv(feature_file)
        return df.head(5).to_dict('records')
    
    return []


def calculate_default_rate_by_company_and_year(model_type=None):
    """Calculate average probability of default for each company and year"""
    try:
        # Load predictions with probability from results - use specified model or try all
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If model_type specified, use only that model
        if model_type:
            models = [model_type]
        else:
            models = ['xgb', 'ann', 'lr', 'tree']  # Order by preference
        
        predictions_path = None
        for model in models:
            # Try bankruptcy_probability_all_years.csv first (XGB format)
            path = os.path.join(backend_dir, f"results/{model}/bankruptcy_probability_all_years.csv")
            if os.path.exists(path):
                predictions_path = path
                break
            # Fallback to predictions_with_probability.csv (ANN, LR, TREE format)
            path = os.path.join(backend_dir, f"results/{model}/predictions_with_probability.csv")
            if os.path.exists(path):
                predictions_path = path
                break
        
        if not predictions_path:
            return None
        
        df = pd.read_csv(predictions_path)
        
        # Handle both file formats (Probability vs Risk_Percent)
        prob_column = 'Probability' if 'Probability' in df.columns else 'Risk_Percent'
        if prob_column not in df.columns:
            return None
        
        # Group by MaCP (company code) and Year, then calculate mean of Probability
        company_year_prob = df.groupby(['MaCP', 'Year'])[prob_column].mean().reset_index()
        company_year_prob.columns = ['company', 'year', 'default_rate']
        company_year_prob['default_rate'] = company_year_prob['default_rate'].round(2)
        
        # Calculate average probability per company across all years (for selection)
        company_avg_prob = df.groupby('MaCP')[prob_column].mean().reset_index()
        company_avg_prob.columns = ['company', 'avg_prob']
        company_avg_prob['avg_prob'] = company_avg_prob['avg_prob'].round(2)
        company_avg_prob = company_avg_prob.sort_values('avg_prob')
        
        return {}
    except Exception as e:
        print(f"Error calculating default rates: {e}")
        return None


@app.route('/results/<model_type>', methods=['GET'])
def get_results(model_type):
    """Get training results for specific model"""
    try:
        results_dir = os.path.join(RESULTS_FOLDER, model_type)
        
        if not os.path.exists(results_dir):
            return jsonify({'error': 'Results not found'}), 404
        
        # Load metrics
        metrics_file = os.path.join(results_dir, 'metrics_test.json')
        metrics = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        
        # Load feature importance
        feature_files = [
            f'feature_influence_analysis_{model_type}.csv',
            'feature_influence_analysis.csv',  # Generic filename (LR, Tree, etc)
            f'feature_importance_{model_type}.csv',
            'feature_importance.csv'  # Generic filename
        ]
        
        features = []
        for feature_file in feature_files:
            filepath = os.path.join(results_dir, feature_file)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                features = df.head(5).to_dict('records')
                break
        
        # Load epoch history if available
        history_file = os.path.join(results_dir, 'training_history.json')
        history = {}
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'metrics': metrics,
            'features': features,
            'history': history
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models = [
        {'id': 'ann', 'name': 'Artificial Neural Network', 'params': ['dropout', 'batchnorm', 'lr', 'weight_decay']},
        {'id': 'lr', 'name': 'Logistic Regression', 'params': ['C', 'penalty']},
        {'id': 'xgb', 'name': 'XGBoost', 'params': ['max_depth', 'learning_rate', 'n_estimators']},
        {'id': 'tree', 'name': 'Decision Tree', 'params': ['max_depth', 'min_samples_leaf', 'criterion']}
    ]
    return jsonify({'models': models}), 200


@app.route('/compare', methods=['POST'])
def compare_models():
    """Get comparison data for all models"""
    try:
        results = {}
        models = ['ann', 'lr', 'xgb', 'tree']
        
        for model_type in models:
            results_dir = os.path.join(RESULTS_FOLDER, model_type)
            
            # Load metrics
            metrics_file = os.path.join(results_dir, 'metrics_test.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    results[model_type] = {
                        'metrics': json.load(f),
                        'features': [],
                        'status': 'trained'
                    }
                    
                    # Load features
                    feature_file = os.path.join(results_dir, f'feature_influence_analysis_{model_type}.csv')
                    if os.path.exists(feature_file):
                        df = pd.read_csv(feature_file)
                        results[model_type]['features'] = df.head(5).to_dict('records')
            else:
                # Model not trained yet
                results[model_type] = {
                    'metrics': {},
                    'features': [],
                    'status': 'not_trained',
                    'message': 'Model has not been trained yet'
                }
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train-all', methods=['POST'])
def train_all_models():
    """Train all models automatically"""
    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        results = {}
        models = ['ann', 'lr', 'xgb', 'tree']
        
        for model_type in models:
            results_dir = os.path.join(RESULTS_FOLDER, model_type)
            
            # Check if model already trained
            metrics_file = os.path.join(results_dir, 'metrics_test.json')
            if os.path.exists(metrics_file):
                # Model already trained, skip
                results[model_type] = {
                    'status': 'already_trained',
                    'message': f'{model_type.upper()} already trained'
                }
                continue
            
            # Train model
            try:
                print(f"Training {model_type.upper()}...")
                cmd = [
                    sys.executable,
                    'train_wrapper.py',
                    model_type,
                    json.dumps({})  # Empty params for default training
                ]
                
                result = subprocess.run(
                    cmd,
                    cwd=backend_dir,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode == 0:
                    output = json.loads(result.stdout)
                    results[model_type] = {
                        'status': 'trained',
                        'message': f'{model_type.upper()} trained successfully'
                    }
                else:
                    results[model_type] = {
                        'status': 'failed',
                        'message': f'Failed to train {model_type.upper()}: {result.stderr}'
                    }
            except Exception as e:
                results[model_type] = {
                    'status': 'failed',
                    'message': str(e)
                }
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/default-rate', methods=['GET'])
def get_default_rate():
    """Get default rate data for highest and lowest companies"""
    try:
        model_type = request.args.get('model_type', 'xgb')  # Default to xgb if not specified
        data = calculate_default_rate_by_company_and_year(model_type=model_type)
        if data is None:
            return jsonify({'error': 'Could not calculate default rates'}), 500
        
        return jsonify({
            'success': True,
            'data': data,
            'model': model_type
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/company-probability', methods=['GET'])
def get_company_probability():
    """Get probability trends for a random or specific company across all models and years"""
    try:
        company_code = request.args.get('company')
        
        # If no company specified, pick a random one
        if not company_code:
            # Load all companies from any model's predictions
            dataset_path = os.path.join(RESULTS_FOLDER, '../uploads/Dataset_CD1_preprocessed.csv')
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                companies = df['MaCP'].unique()
                company_code = np.random.choice(companies)
            else:
                return jsonify({'error': 'Dataset not found'}), 404
        
        result = {
            'company': company_code,
            'models': {}
        }
        
        # Get data from all models
        models = ['ann', 'lr', 'xgb', 'tree']
        for model_type in models:
            results_dir = os.path.join(RESULTS_FOLDER, model_type)
            
            # Try both prediction file formats
            pred_files = [
                os.path.join(results_dir, 'bankruptcy_probability_all_years.csv'),
                os.path.join(results_dir, 'predictions_with_probability.csv')
            ]
            
            for pred_file in pred_files:
                if os.path.exists(pred_file):
                    df = pd.read_csv(pred_file)
                    
                    # Filter by company
                    company_data = df[df['MaCP'] == company_code]
                    
                    if len(company_data) > 0:
                        # Group by year and get average probability
                        prob_col = 'Probability' if 'Probability' in df.columns else 'Risk_Percent'
                        yearly_data = company_data.groupby('Year')[prob_col].mean().reset_index()
                        yearly_data.columns = ['year', 'probability']
                        yearly_data = yearly_data.sort_values('year')
                        yearly_data['probability'] = yearly_data['probability'].round(2)
                        
                        result['models'][model_type] = yearly_data.to_dict('records')
                    break
        
        if not result['models']:
            return jsonify({'error': f'No data found for company {company_code}'}), 404
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict default probability for given features using all models"""
    try:
        data = request.json
        features_dict = data.get('features', {})
        
        # Expected features (X1-X13)
        feature_names = [f'X{i}' for i in range(1, 14)]
        
        # Extract features, skip missing ones
        features = {}
        for name in feature_names:
            if name in features_dict and features_dict[name] is not None:
                try:
                    features[name] = float(features_dict[name])
                except (ValueError, TypeError):
                    pass  # Skip invalid values
        
        if not features:
            return jsonify({'error': 'No valid features provided'}), 400
        
        results = {}
        models = ['ann', 'lr', 'tree', 'xgb']
        
        for model_type in models:
            try:
                model_path = os.path.join(RESULTS_FOLDER, model_type, 'final_model')
                pkl_file = os.path.join(model_path, f'pd_{model_type}_model.pkl')
                feature_file = os.path.join(model_path, 'feature_schema.json')
                
                if not os.path.exists(pkl_file):
                    results[model_type] = {'error': 'Model not found'}
                    continue
                
                # Load model
                import joblib
                model = joblib.load(pkl_file)
                
                # Load feature schema
                if os.path.exists(feature_file):
                    with open(feature_file, 'r') as f:
                        schema = json.load(f)
                        expected_features = schema.get('features', feature_names)
                else:
                    expected_features = feature_names
                
                # Prepare feature array with correct order
                feature_values = []
                for fname in expected_features:
                    if fname in features:
                        feature_values.append(features[fname])
                    else:
                        feature_values.append(0)  # Use 0 for missing features
                
                # Make prediction
                X = np.array(feature_values).reshape(1, -1)
                y_prob = model.predict_proba(X)[0, 1]
                
                results[model_type] = {
                    'probability': float(round(y_prob * 100, 2)),
                    'features_used': len(features),
                    'features_expected': len(expected_features)
                }
            except Exception as e:
                results[model_type] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'data': results,
            'input_features': features
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/debug-models', methods=['GET'])
def debug_models():
    """Debug endpoint to inspect all models"""
    try:
        import torch
        models_info = {}
        models = ['ann', 'lr', 'tree', 'xgb']
        
        for model_type in models:
            model_path = os.path.join(RESULTS_FOLDER, model_type, 'final_model')
            info = {
                'path': model_path,
                'exists': os.path.exists(model_path),
                'files': [],
                'model_loaded': False,
                'model_type': None,
                'has_predict_proba': False,
                'error': None
            }
            
            if os.path.exists(model_path):
                info['files'] = os.listdir(model_path)
                
                # Try to find and load model
                pkl_file = None
                for pattern in ['pd_model.pkl', 'pd_tree_model.pkl', 'ann_final_model.pth.joblib', 'pd_ann_model.pkl']:
                    candidate = os.path.join(model_path, pattern)
                    if os.path.exists(candidate):
                        pkl_file = candidate
                        break
                
                if pkl_file:
                    try:
                        model = joblib.load(pkl_file)
                        info['model_loaded'] = True
                        info['model_type'] = type(model).__name__
                        info['has_predict_proba'] = hasattr(model, 'predict_proba')
                        
                        # Check if it's torch model
                        if isinstance(model, torch.nn.Module):
                            info['model_type'] = 'PyTorch ANN'
                        elif hasattr(model, 'named_steps'):
                            info['model_type'] = 'Pipeline'
                            info['pipeline_steps'] = list(model.named_steps.keys())
                    except Exception as e:
                        info['error'] = str(e)
                else:
                    info['error'] = 'No model file found'
            
            models_info[model_type] = info
        
        return jsonify(models_info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/predict-from-file', methods=['POST'])
def predict_from_file():
    """Predict for next year based on preprocessed CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Read CSV
        df = pd.read_csv(file.stream)
        
        # Validate required columns
        required_cols = ['STT', 'MaCP', 'Year', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400
        
        # Find max year
        max_year = int(df['Year'].max())
        next_year = max_year + 1
        
        # Get data for max year
        max_year_data = df[df['Year'] == max_year]
        
        if len(max_year_data) == 0:
            return jsonify({'error': f'No data found for year {max_year}'}), 400
        
        # Get list of available companies
        available_companies = df['MaCP'].unique().tolist()
        
        # Get company from request or use first company from max year
        selected_company = request.form.get('company', max_year_data.iloc[0]['MaCP'])
        
        # Find data for selected company in max year
        company_row = max_year_data[max_year_data['MaCP'] == selected_company]
        
        if len(company_row) == 0:
            # If not found, use first available
            company_row = max_year_data.iloc[[0]]
            selected_company = company_row.iloc[0]['MaCP']
        else:
            company_row = company_row.iloc[[0]]
        
        company_code = company_row.iloc[0]['MaCP']
        
        # Extract features X1-X13
        features = {}
        for i in range(1, 14):
            col_name = f'X{i}'
            val = company_row.iloc[0][col_name]
            if pd.notna(val):
                features[col_name] = float(val)
            else:
                features[col_name] = 0
        
        # Make predictions with all models
        results = {}
        models = ['ann', 'lr', 'tree', 'xgb']
        
        for model_type in models:
            try:
                model_path = os.path.join(RESULTS_FOLDER, model_type, 'final_model')
                pkl_file = os.path.join(model_path, f'pd_{model_type}_model.pkl')
                feature_file = os.path.join(model_path, 'feature_schema.json')
                
                if not os.path.exists(pkl_file):
                    results[model_type] = {'error': 'Model not found'}
                    continue
                
                # Load model
                model = joblib.load(pkl_file)
                
                # Load feature schema
                feature_names = [f'X{i}' for i in range(1, 14)]
                if os.path.exists(feature_file):
                    with open(feature_file, 'r') as f:
                        schema = json.load(f)
                        feature_names = schema.get('features', feature_names)
                
                # Prepare feature array
                feature_values = []
                for fname in feature_names:
                    if fname in features:
                        feature_values.append(features[fname])
                    else:
                        feature_values.append(0)
                
                # Make prediction
                X = np.array(feature_values).reshape(1, -1)
                y_prob = model.predict_proba(X)[0, 1]
                
                results[model_type] = {
                    'probability': float(round(y_prob * 100, 2))
                }
            except Exception as e:
                results[model_type] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'data': {
                'maxYear': max_year,
                'nextYear': next_year,
                'companyCode': company_code,
                'features': features,
                'predictions': results
            },
            'companies': available_companies
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-from-file-by-company', methods=['POST'])
def predict_from_file_by_company():
    """Predict for a specific company from the uploaded file"""
    try:
        data = request.json
        company_code = data.get('company')
        
        if not company_code:
            return jsonify({'error': 'Company code required'}), 400
        
        # Load the processed file
        file_path = os.path.join('uploads', 'processed_BCTC_by_year.csv')
        if not os.path.exists(file_path):
            return jsonify({'error': 'Processed file not found. Please upload a file first.'}), 400
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ['STT', 'MaCP', 'Year', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400
        
        # Find max year for this company
        company_data = df[df['MaCP'] == company_code]
        
        if len(company_data) == 0:
            return jsonify({'error': f'Company {company_code} not found in file'}), 400
        
        max_year = int(company_data['Year'].max())
        next_year = max_year + 1
        
        # Get data for max year
        company_row = company_data[company_data['Year'] == max_year]
        
        if len(company_row) == 0:
            return jsonify({'error': f'No data found for company {company_code} in year {max_year}'}), 400
        
        company_row = company_row.iloc[0]
        
        # Extract features X1-X13
        features = {}
        for i in range(1, 14):
            col_name = f'X{i}'
            val = company_row[col_name]
            if pd.notna(val):
                features[col_name] = float(val)
            else:
                features[col_name] = 0
        
        # Make predictions with all models
        results = {}
        models = ['ann', 'lr', 'tree', 'xgb']
        
        for model_type in models:
            try:
                model_path = os.path.join(RESULTS_FOLDER, model_type, 'final_model')
                pkl_file = os.path.join(model_path, f'pd_{model_type}_model.pkl')
                feature_file = os.path.join(model_path, 'feature_schema.json')
                
                if not os.path.exists(pkl_file):
                    results[model_type] = {'error': 'Model not found'}
                    continue
                
                # Load model
                model = joblib.load(pkl_file)
                
                # Load feature schema
                feature_names = [f'X{i}' for i in range(1, 14)]
                if os.path.exists(feature_file):
                    with open(feature_file, 'r') as f:
                        schema = json.load(f)
                        feature_names = schema.get('features', feature_names)
                
                # Prepare feature array
                feature_values = []
                for fname in feature_names:
                    if fname in features:
                        feature_values.append(features[fname])
                    else:
                        feature_values.append(0)
                
                # Make prediction
                X = np.array(feature_values).reshape(1, -1)
                y_prob = model.predict_proba(X)[0, 1]
                
                results[model_type] = {
                    'probability': float(round(y_prob * 100, 2))
                }
            except Exception as e:
                results[model_type] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'data': {
                'maxYear': max_year,
                'nextYear': next_year,
                'companyCode': company_code,
                'features': features,
                'predictions': results
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/test-single-predict', methods=['POST'])
def test_single_predict():
    """Test single model prediction with detailed error info"""
    try:
        data = request.json
        model_type = data.get('model_type', 'lr')
        test_values = {
            'X1': 0.5, 'X2': 0.5, 'X3': 0.5, 'X4': 0.5,
            'X5': 0.5, 'X6': 0.5, 'X7': 0.5, 'X8': 0.5,
            'X9': 0.5, 'X10': 0.5, 'X11': 0.5, 'X12': 0.5, 'X13': 0.5
        }
        
        result = {'model': model_type, 'steps': []}
        
        model_path = os.path.join(RESULTS_FOLDER, model_type, 'final_model')
        result['steps'].append(f"Model path: {model_path}")
        
        # Find model file
        pkl_file = None
        for pattern in ['pd_model.pkl', 'pd_tree_model.pkl', 'ann_final_model.pth.joblib']:
            candidate = os.path.join(model_path, pattern)
            if os.path.exists(candidate):
                pkl_file = candidate
                break
        
        if not pkl_file:
            result['error'] = 'Model file not found'
            return jsonify(result), 400
        
        result['steps'].append(f"Found model file: {os.path.basename(pkl_file)}")
        
        # Load model
        model = joblib.load(pkl_file)
        result['steps'].append(f"Model loaded: {type(model).__name__}")
        
        # Load feature schema
        feature_file_json = os.path.join(model_path, 'feature_schema.json')
        feature_file_csv = os.path.join(model_path, 'feature_schema.csv')
        expected_features = None
        
        if os.path.exists(feature_file_json):
            with open(feature_file_json, 'r') as f:
                schema = json.load(f)
                expected_features = schema.get('features')
            result['steps'].append(f"Features from JSON: {expected_features}")
        elif os.path.exists(feature_file_csv):
            df_feat = pd.read_csv(feature_file_csv)
            expected_features = df_feat['feature'].tolist()
            result['steps'].append(f"Features from CSV: {expected_features}")
        else:
            expected_features = [f'X{i}' for i in range(1, 14)]
            result['steps'].append(f"Using default features: {expected_features}")
        
        # Prepare data
        feature_data = {}
        for fname in expected_features:
            if fname == 'Year':
                import datetime
                feature_data[fname] = float(datetime.datetime.now().year)
            else:
                feature_data[fname] = test_values.get(fname, 0.0)
        
        # Create DataFrame
        X = pd.DataFrame([feature_data])
        result['steps'].append(f"Input shape: {X.shape}")
        result['steps'].append(f"Input columns: {list(X.columns)}")
        
        # Try prediction
        try:
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X)[0, 1]
                result['success'] = True
                result['probability'] = float(y_prob * 100)
                result['steps'].append(f"Prediction successful: {y_prob*100:.2f}%")
            else:
                result['steps'].append(f"Model has no predict_proba")
                result['error'] = 'Model does not have predict_proba method'
        except Exception as e:
            result['error'] = str(e)
            result['steps'].append(f"Prediction error: {str(e)}")
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500


@app.route('/best-params', methods=['GET'])
def get_best_params():
    """Get optimal parameters for all models"""
    try:
        best_params_all = {}
        model_names = {
            'ann': 'ANN',
            'lr': 'LR',
            'xgb': 'XGB',
            'tree': 'TREE'
        }
        
        for model_type, model_name in model_names.items():
            params_file = os.path.join(RESULTS_FOLDER, model_type, 'final_model', 'best_params.json')
            if os.path.exists(params_file):
                try:
                    with open(params_file, 'r') as f:
                        best_params_all[model_type] = json.load(f)
                except Exception as e:
                    best_params_all[model_type] = None
            else:
                best_params_all[model_type] = None
        
        return jsonify({
            'success': True,
            'best_params': best_params_all,
            'models': ['ann', 'lr', 'xgb', 'tree']
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    """Upload raw BCTC file, calculate indicators X1-X13, preprocess, and return processed CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        allowed_ext = {'xlsx', 'xls', 'csv'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_ext:
            return jsonify({'error': 'Only Excel (.xlsx, .xls) or CSV files are allowed'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}')
        file.save(temp_path)
        
        try:
            # Import preprocessing utilities
            from preprocess_utils import process_raw_data
            
            # Process the data
            processed_df = process_raw_data(temp_path)
            
            # Save processed file
            processed_filename = f'processed_{filename.rsplit(".", 1)[0]}.csv'
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            processed_df.to_csv(processed_path, index=False, encoding='utf-8-sig')
            
            # Return download info and preview
            return jsonify({
                'success': True,
                'filename': processed_filename,
                'download_url': f'/download/{processed_filename}',
                'shape': list(processed_df.shape),
                'columns': list(processed_df.columns),
                'preview': processed_df.head(10).to_dict('records'),
                'statistics': {
                    'rows': len(processed_df),
                    'columns': len(processed_df.columns),
                    'companies': int(processed_df['MaCP'].nunique()) if 'MaCP' in processed_df.columns else 0,
                    'years': int(processed_df['Year'].nunique()) if 'Year' in processed_df.columns else 0
                }
            }), 200
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed CSV file"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Security check - ensure file is in upload folder
        if not os.path.abspath(filepath).startswith(os.path.abspath(app.config['UPLOAD_FOLDER'])):
            return jsonify({'error': 'Invalid file path'}), 400
        
        from flask import send_file
        return send_file(
            filepath,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)