import os
import ast
import sys
from pathlib import Path
import torch
from torchsummary import summary

def calculate_conv2d_params(in_channels, out_channels, kernel_size):
    """Calculate parameters for Conv2d layer"""
    if isinstance(kernel_size, (int, float)):
        kernel_size = (kernel_size, kernel_size)
    return (in_channels * out_channels * kernel_size[0] * kernel_size[1]) + out_channels

def count_parameters(node):
    """Count potential parameters in layer definitions"""
    param_count = 0
    if isinstance(node, ast.Call):
        if hasattr(node.func, 'attr'):
            # Conv2d layers
            if node.func.attr == 'Conv2d' and len(node.args) >= 3:
                in_channels = ast.literal_eval(node.args[0])
                out_channels = ast.literal_eval(node.args[1])
                kernel_size = ast.literal_eval(node.args[2])
                param_count += calculate_conv2d_params(in_channels, out_channels, kernel_size)
            
            # BatchNorm2d layers
            elif node.func.attr == 'BatchNorm2d' and len(node.args) >= 1:
                num_features = ast.literal_eval(node.args[0])
                param_count += 2 * num_features  # gamma and beta parameters
            
            # Linear layers
            elif node.func.attr == 'Linear' and len(node.args) >= 2:
                in_features = ast.literal_eval(node.args[0])
                out_features = ast.literal_eval(node.args[1])
                param_count += (in_features * out_features) + out_features
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
        # Check for layer assignments in __init__
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(node.value, ast.Call):
                    # Check for BatchNorm
                    if hasattr(node.value.func, 'attr') and 'BatchNorm' in node.value.func.attr:
                        findings['has_batchnorm'] = True
                    
                    # Check for Dropout
                    elif hasattr(node.value.func, 'attr') and 'Dropout' in node.value.func.attr:
                        findings['has_dropout'] = True
                    
                    # Check for FC/GAP
                    elif hasattr(node.value.func, 'attr') and ('Linear' in node.value.func.attr or 'AdaptiveAvgPool' in node.value.func.attr):
                        findings['has_fc_or_gap'] = True
                    
                    # Count parameters for this layer
                    findings['total_params'] += count_parameters(node.value)
    
    return findings

def main():
    # Verify torch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}\n")
    
    # Find all Python files in the project
    model_files = list(Path('.').rglob('*model*.py'))
    test_files = list(Path('.').rglob('*test*.py'))
    check_files = list(Path('.').rglob('*check*.py'))
    
    # Exclude test and check files
    model_files = [f for f in model_files if f not in test_files and f not in check_files]
    
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
        print(f"✓ Estimated Parameters: {findings['total_params']:,}")
        if findings['total_params'] > 20000:  # Changed threshold to 20,000
            print("⚠️  Warning: Model has more than 20,000 parameters")
            all_checks_passed = False
    
    if not all_checks_passed:
        sys.exit(1)

if __name__ == "__main__":
    main() 