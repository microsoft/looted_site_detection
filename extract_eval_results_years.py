#!/usr/bin/env python3
"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
"""
Extract eval_results.json from training logs for CNN experiments across years (2017-2023)
"""

import json
import os
import re
from pathlib import Path

def extract_json_from_log(log_path):
    """Extract the JSON results block from a training log file."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Find the last occurrence of { that starts a JSON object with "feature_type"
        # Look backwards from the end to find the JSON block
        last_brace_idx = content.rfind('{')
        if last_brace_idx == -1:
            return None
        
        # Search backwards for a line containing "feature_type" 
        feature_type_idx = content.rfind('"feature_type"')
        if feature_type_idx == -1:
            return None
        
        # Find the opening brace before "feature_type"
        start_idx = content.rfind('{', 0, feature_type_idx + 1)
        if start_idx == -1:
            return None
        
        # Now find the matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            try:
                result = json.loads(json_str)
                # Verify it has the expected structure
                if 'metrics' in result and 'f1' in result['metrics']:
                    return result
            except json.JSONDecodeError as e:
                print(f"    JSON decode error: {e}")
                return None
        
        return None
    except Exception as e:
        print(f"    Error reading {log_path}: {e}")
        return None

def main():
    import os
    base_dir = Path(os.environ.get('RESULTS_DIR', 'results'))
    
    # Years to process (2017-2023)
    years = range(2017, 2024)
    
    # Model configuration
    model = "efficientnet_b1"
    pretrained = True
    num_folds = 5
    
    total_extracted = 0
    total_failed = 0
    total_skipped = 0
    
    print("Extracting eval_results.json for years 2017-2023")
    print("=" * 70)
    
    for year in years:
        year_dir = base_dir / f"model_runs_cnn_{year}"
        
        if not year_dir.exists():
            print(f"\nYear {year}: Directory not found - {year_dir}")
            continue
        
        print(f"\nProcessing year {year}:")
        print("-" * 50)
        
        year_extracted = 0
        year_failed = 0
        year_skipped = 0
        
        for fold in range(num_folds):
            fold_dir = year_dir / f"{model}_pretrained_{pretrained}_fold_{fold}"
            eval_results_path = fold_dir / "eval_results.json"
            training_log_path = fold_dir / "training_log.txt"
            
            if not fold_dir.exists():
                print(f"  Fold {fold}: Directory not found")
                year_failed += 1
                continue
            
            # Check if eval_results.json already exists
            if eval_results_path.exists():
                print(f"  Fold {fold}: eval_results.json already exists - skipping")
                year_skipped += 1
                continue
            
            # Check if training log exists
            if not training_log_path.exists():
                print(f"  Fold {fold}: training_log.txt not found")
                year_failed += 1
                continue
            
            # Extract JSON from log
            result = extract_json_from_log(training_log_path)
            
            if result:
                # Write to eval_results.json
                with open(eval_results_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Fold {fold}: ✓ Extracted (F1: {result['metrics']['f1']:.4f})")
                year_extracted += 1
            else:
                print(f"  Fold {fold}: ✗ Failed to extract JSON from log")
                year_failed += 1
        
        print(f"Year {year} summary: {year_extracted} extracted, {year_skipped} skipped, {year_failed} failed")
        total_extracted += year_extracted
        total_failed += year_failed
        total_skipped += year_skipped
    
    print("\n" + "=" * 70)
    print(f"Total: {total_extracted} extracted, {total_skipped} skipped, {total_failed} failed")
    print("=" * 70)

if __name__ == "__main__":
    main()
