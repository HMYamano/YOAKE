import json
for split in ['train', 'val']:
    with open(r'C:\Users\utopi\Desktop\YOAKE_tryal\data\\' + split + r'\annotations.json') as f:
        d = json.load(f)
    vids = len(d['videos'])
    frames = sum(len(v['frames']) for v in d['videos'])
    print(f'{split}: {vids} sequences, {frames} frames')