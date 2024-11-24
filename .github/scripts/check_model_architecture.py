import os
import ast
import sys
from pathlib import Path
import torch
from torchsummary import summary

def count_parameters(node):
    """Count potential parameters in layer definitions"""
    param_count = 0
    if isinstance(node, ast.Call):
        # Common layer types that have parameters
        param_layers = ['Conv2d', 'Linear', 'LSTM', 'GRU', 'Dense', 'ConvTranspose2d', 'BatchNorm2d']
        if hasattr(node.func, 'id') and node.func.id in param_layers:
            # Basic parameter estimation based on common args
            for arg in node.args:
                if isinstance(arg, ast.Num):
                    param_count += arg.n
    return param_count

def check_model_architecture(file_content):
    tree = ast.parse(file_content)
    
    findings = {
        'has_batchnorm': False,
        'has_dropout': False,
        'has_fc_or_gap': False,
        'total_params': 0,
        'imports_ok': True
    }
    
    # Check imports first
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            try:
                if isinstance(node, ast.Import):
                    module_name = node.names[0].name
                else:
                    module_name = node.module
                
                if module_name in ['torch', 'torch.nn']:
                    findings['imports_ok'] &= True
            except ImportError as e:
                findings['imports_ok'] = False
                print(f"Warning: Import error - {e}")
    
    for node in ast.walk(tree):
        # Check for BatchNorm
        if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
            if 'BatchNorm' in node.func.id:
                findings['has_batchnorm'] = True
            
            # Check for Dropout
            elif 'Dropout' in node.func.id:
                findings['has_dropout'] = True
            
            # Check for FC/GAP
            elif 'Linear' in node.func.id or 'AdaptiveAvgPool' in node.func.id:
                findings['has_fc_or_gap'] = True
        
        # Count parameters
        findings['total_params'] += count_parameters(node)
    
    return findings

def main():
    # Verify torch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}\n")
    
    # Find all Python files in the project
    model_files = list(Path('.').rglob('*model*.py'))
    
    if not model_files:
        print("❌ No model files found!")
        sys.exit(1)
    
    all_checks_passed = True
    
    for model_file in model_files:
        print(f"\nChecking {model_file}...")
        
        with open(model_file, 'r') as f:
            content = f.read()
        
        findings = check_model_architecture(content)
        
        # Print findings
        print("\nArchitecture Check Results:")
        print(f"{'='*30}")
        
        # Check imports
        if not findings['imports_ok']:
            print("❌ Required imports are missing")
            all_checks_passed = False
        
        # Check BatchNorm
        print(f"✓ BatchNorm: {findings['has_batchnorm']}")
        if not findings['has_batchnorm']:
            print("⚠️  Warning: No Batch Normalization found")
            all_checks_passed = False
        
        # Check Dropout
        print(f"✓ Dropout: {findings['has_dropout']}")
        if not findings['has_dropout']:
            print("⚠️  Warning: No Dropout layers found")
            all_checks_passed = False
        
        # Check FC/GAP
        print(f"✓ FC/GAP: {findings['has_fc_or_gap']}")
        if not findings['has_fc_or_gap']:
            print("⚠️  Warning: No Fully Connected or Global Average Pooling layers found")
            all_checks_passed = False
        
        # Check parameter count
        print(f"✓ Estimated Parameters: {findings['total_params']}")
        if findings['total_params'] > 1e8:  # 100M parameters
            print("⚠️  Warning: Model might be too large (>100M parameters)")
            all_checks_passed = False
    
    if not all_checks_passed:
        sys.exit(1)

if __name__ == "__main__":
    main() 