import pandas as pd
import os

def verify_id_sequence(test_file, submission_file):
    test_df = pd.read_csv(test_file, sep='\t')
    test_ids = test_df['ID'].astype(str).tolist()
    
    sub_ids = []
    with open(submission_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # Extract ID from "ID"\t"Output"
                parts = line.split('\t')
                if parts:
                    sub_id = parts[0].strip('"')
                    sub_ids.append(sub_id)
    
    print(f"File: {os.path.basename(submission_file)}")
    print(f"  Test IDs: {len(test_ids)}")
    print(f"  Sub IDs:  {len(sub_ids)}")
    
    if len(test_ids) != len(sub_ids):
        print(f"  ERROR: Line count mismatch! {len(test_ids)} vs {len(sub_ids)}")
        return
    
    mismatches = []
    for i, (tid, sid) in enumerate(zip(test_ids, sub_ids)):
        if tid != sid:
            mismatches.append((i+1, tid, sid))
    
    if mismatches:
        print(f"  ERROR: ID mismatch found at {len(mismatches)} lines.")
        for line_no, tid, sid in mismatches[:5]:
            print(f"    Line {line_no}: Expected {tid}, found {sid}")
    else:
        print("  SUCCESS: All IDs match the test set order exactly.")

print("--- VERIFYING ID SEQUENCES ---")
verify_id_sequence('f:/!Projects/MSLG-SPA 2026/MSLG2SPA_test.txt', 'f:/!Projects/MSLG-SPA 2026/MBM_Solution 1_MSLG2SPA.txt')
verify_id_sequence('f:/!Projects/MSLG-SPA 2026/SPA2MSLG_test.txt', 'f:/!Projects/MSLG-SPA 2026/MBM_Solution 1_SPA2MSLG.txt')
