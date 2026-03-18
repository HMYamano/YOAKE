# アノテーションの構造確認
import json

with open(r"C:\Users\hayam\Desktop\YOAKE_tryal/data\val\annotations.json") as f:
    ann = json.load(f)

if isinstance(ann, dict):
    print("Top-level keys:", list(ann.keys()))
    for k, v in ann.items():
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)} items")
            if v and isinstance(v[0], dict):
                print(f"    first item keys: {list(v[0].keys())}")
                print(f"    first item: {v[0]}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
elif isinstance(ann, list):
    print(f"Top-level: list of {len(ann)} items")
    if ann:
        first = ann[0]
        print(f"First item type: {type(first).__name__}")
        if isinstance(first, dict):
            print(f"First item keys: {list(first.keys())}")
            print(f"First item: {first}")
else:
    print(f"Unexpected type: {type(ann)}")