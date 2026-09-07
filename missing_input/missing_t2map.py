import re
from datetime import datetime

log_path = r"E:\Project_DCE\DCE_PRG_NEW\bids\derivatives\2026_07_02\logs\PraguePipeline_141153.log"          # <- your log file
out_path = "missing_mp2rage.txt"  # <- output

marker = "ERROR : [Z score] MP2RAGE file not found"
pattern = re.compile(r"sub-([^\\/]+)[\\/]ses-([^\\/\s]+)")

rows = []
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if marker in line:
            m = pattern.search(line)
            if m:
                sub, ses = m.group(1), m.group(2)
                # convert YYYYMMDD -> dd.mm.YYYY
                date = datetime.strptime(ses, "%Y%m%d").strftime("%d.%m.%Y")
                rows.append((sub, date))

# drop duplicates, keep order
seen = set()
rows = [r for r in rows if not (r in seen or seen.add(r))]

with open(out_path, "w", encoding="utf-8") as f:
    for sub, date in rows:
        f.write(f"{sub}\t{date}\n")

print(f"Found {len(rows)} entries, written to {out_path}")