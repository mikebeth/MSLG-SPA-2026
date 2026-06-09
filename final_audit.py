import os

def audit_submission(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # 1. Total Line Count (IberLEF rule: one per instance)
    # Instances are 245 in the test set.
    lines = content.splitlines()
    total_lines = len(lines)
    
    # 2. Check for Carriage Return (\r) - Should be FALSE for Linux format
    has_cr = b'\r' in content
    
    # 3. Verify exactly \n as terminator
    ends_with_newline = content.endswith(b'\n')
    
    # 4. Check for drift (search for internal quotes and tabs)
    # Each line must start with " and have exactly one \t
    corrupted_lines = []
    for i, line in enumerate(lines):
        # We expect "ID"\t"Prediction"
        parts = line.decode('utf-8', errors='ignore').split('\t')
        if len(parts) != 2:
            corrupted_lines.append(i+1)
        elif not parts[0].startswith('"') or not parts[0].endswith('"'):
            corrupted_lines.append(i+1)
            
    print(f"AUDIT REPORT: {os.path.basename(file_path)}")
    print(f"  - Instance Count: {total_lines} (Expected: 245)")
    print(f"  - Linux Format (No \\r): {'SUCCESS' if not has_cr else 'FAILED (Contains \\r)'}")
    print(f"  - Ends with \\n: {'SUCCESS' if ends_with_newline else 'FAILED'}")
    if corrupted_lines:
        print(f"  - Drift/Format Alignment Errors: FAILED at lines {corrupted_lines[:5]}")
    else:
        print(f"  - Drift/Format Alignment: SUCCESS (Exactly one tab per line)")
    print("-" * 30)

audit_submission('f:/!Projects/MSLG-SPA 2026/MBM_Solution 1_MSLG2SPA.txt')
audit_submission('f:/!Projects/MSLG-SPA 2026/MBM_Solution 1_SPA2MSLG.txt')
