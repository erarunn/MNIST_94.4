# ML Model Architecture Checker

This repository contains GitHub Actions workflows to automatically check ML model architectures for best practices and common requirements. It specifically analyzes MNIST model implementations for proper architecture components.

## Installation

1. Clone the repository:

## What it checks

The checker analyzes Python files containing model definitions and verifies:

1. **Total Parameter Count**: Estimates the total number of parameters in the model and warns if it exceeds 100M parameters
2. **Batch Normalization**: Checks for the presence of BatchNorm layers
3. **Dropout**: Verifies that Dropout layers are included for regularization
4. **Architecture Components**: Confirms the use of either Fully Connected layers or Global Average Pooling

## How it works

The checker automatically runs on:
- Every push to the `main` branch
- All pull requests targeting the `main` branch

### Check Results

The workflow will show:
- ✓ Pass/fail status for each check
- ⚠️ Warnings for missing components
- Estimated parameter count

Example output: 