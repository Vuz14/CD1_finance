"""
Test script to verify the application setup
Run this to ensure everything is configured correctly
"""
import sys
import subprocess
import requests
from pathlib import Path

def test_backend_setup():
    """Test if backend dependencies are installed"""
    print("\n🔍 Testing Backend Setup...")
    try:
        import flask
        print("✅ Flask installed")
        import pandas
        print("✅ Pandas installed")
        import numpy
        print("✅ NumPy installed")
        import sklearn
        print("✅ Scikit-learn installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_backend_health():
    """Test if backend server is running"""
    print("\n🔍 Testing Backend Health...")
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend (is it running on port 5000?)")
        return False
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def test_python_scripts():
    """Test if training scripts exist"""
    print("\n🔍 Testing Python Training Scripts...")
    backend_path = Path('backend')
    required_files = [
        'train_ann.py',
        'train_lr.py',
        'train_xgb.py',
        'train_tree.py',
        'spilit.py'
    ]
    
    all_present = True
    for file in required_files:
        file_path = backend_path / file
        if file_path.exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} NOT found")
            all_present = False
    
    if not all_present:
        print("\n⚠️  Some training scripts are missing!")
        print("Please copy them from your old project to the backend/ directory")
    
    return all_present

def test_frontend_setup():
    """Test if frontend dependencies are installed"""
    print("\n🔍 Testing Frontend Setup...")
    try:
        frontend_path = Path('frontend/node_modules')
        if frontend_path.exists():
            print("✅ Frontend node_modules exists")
            return True
        else:
            print("⚠️  Frontend dependencies not installed")
            print("Run: cd frontend && npm install")
            return False
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def test_file_structure():
    """Test if project structure is correct"""
    print("\n🔍 Testing Project Structure...")
    required_dirs = [
        'backend',
        'frontend/src',
        'frontend/src/components',
        'frontend/src/pages',
        'frontend/src/types'
    ]
    
    all_present = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"❌ {dir_path}/ NOT found")
            all_present = False
    
    return all_present

def print_header():
    """Print welcome header"""
    print("\n" + "="*50)
    print("   ML Model Trainer - Setup Verification")
    print("="*50)

def print_summary(results):
    """Print test summary"""
    print("\n" + "="*50)
    print("   Summary")
    print("="*50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. cd backend && python app.py")
        print("2. cd frontend && npm start")
        print("3. Open http://localhost:3000 in your browser")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    return passed == total

def main():
    print_header()
    
    results = {
        "File Structure": test_file_structure(),
        "Backend Dependencies": test_backend_setup(),
        "Python Training Scripts": test_python_scripts(),
        "Frontend Setup": test_frontend_setup(),
    }
    
    # Only test health if backend seems configured
    if results["Backend Dependencies"]:
        results["Backend Health Check"] = test_backend_health()
    
    success = print_summary(results)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
