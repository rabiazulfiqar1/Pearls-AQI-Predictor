from pathlib import Path

for p in Path(".").rglob("karachi_training_data.csv"):
    print(p.resolve())