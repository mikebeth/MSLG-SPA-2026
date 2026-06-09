import pandas as pd
import os
import re

print("--- INSPECTING TEST FILE ---")
test_path = 'f:/!Projects/MSLG-SPA 2026/SPA2MSLG_test.txt'
if os.path.exists(test_path):
    df_test = pd.read_csv(test_path, sep='\t')
    target_ids = [35, 396, 644]
    for i in target_ids:
        row = df_test[df_test['ID'] == i]
        if not row.empty:
            print(f"ID {i} (TEST): {repr(row['SPA'].values[0])}")

print("\n--- INSPECTING CSV PREDICTIONS ---")
csv_path = 'f:/!Projects/MSLG-SPA 2026/predictions_subtask_B.csv'
if os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
    for i in target_ids:
        row = df_csv[df_csv['ID'] == i]
        if not row.empty:
            print(f"ID {i} (CSV): {repr(row['prediction'].values[0])}")

print("\n--- INSPECTING FINAL TXT ---")
txt_path = 'f:/!Projects/MSLG-SPA 2026/MBM_Solution 1_SPA2MSLG.txt'
if os.path.exists(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Total lines in TXT: {len(lines)}")
        for i in target_ids:
            matching = [l for l in lines if l.startswith(f'"{i}"')]
            if matching:
                print(f"ID {i} (TXT): {repr(matching[0])}")
            else:
                print(f"ID {i} NOT FOUND IN TXT")

print("\n--- SEARCHING FOR TAGS GLOBALLY IN TXT ---")
if os.path.exists(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        tags = re.findall(r'<[^>]*>', content)
        if tags:
            print(f"Found tags: {set(tags)}")
        else:
            print("No <tags> found in TXT.")
