"""Word counts per speaker block of a talk script, at 140 and 150 wpm.

  python3 code/wordcount.py update-talk-script.md
"""
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
body = text.split("\n---\n")[1] if "\n---\n" in text else text
blocks = re.split(r"^## ", body, flags=re.M)[1:]
total, per = 0, {}
for b in blocks:
    head, _, rest = b.partition("\n")
    m = re.search(r"\((\w+), (\d+) s\)", head)
    if not m:
        continue
    words = len(re.findall(r"[A-Za-z0-9'’\-]+", rest))
    total += words
    per[m.group(1)] = per.get(m.group(1), 0) + words
    print(f"{head[:44]:<46}{words:5d} words  {words / 140 * 60:5.0f}s at 140  (budget {m.group(2)}s)")
print(f"\ntotal {total} words: {total / 140:.2f} min at 140 wpm, {total / 150:.2f} at 150")
print("  " + "  ".join(f"{k} {v}w={v / 140 * 60:.0f}s" for k, v in per.items()))
