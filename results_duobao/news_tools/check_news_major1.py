import os
import re
import json

DIR = r"D:\iquant_data\data_v2\news_major1"

files = [f for f in os.listdir(DIR) if f.startswith("analysis_") and f.endswith(".json")]
dates = []

for f in files:
    match = re.search(r'analysis_(\d{4}-\d{2}-\d{2})\.json', f)
    if match:
        dates.append(match.group(1))

dates_sorted = sorted(dates)

print(f"Total files: {len(files)}")
print(f"First date: {dates_sorted[0]}")
print(f"Last date:  {dates_sorted[-1]}")
print(f"\nLast 20 dates:")
for d in dates_sorted[-20:]:
    print(f"  {d}")

print(f"\nChecking latest file...")
latest_file = os.path.join(DIR, f"analysis_{dates_sorted[-1]}.json")
with open(latest_file, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"  Filename: {latest_file}")
print(f"  Article date: {data.get('article_date')}")
print(f"  Title: {data.get('article_title')}")
print(f"  Market impact: {data.get('market_impact')}")
