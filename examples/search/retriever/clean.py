#!/usr/bin/env python3
"""Drop malformed lines from a JSONL corpus.

Usage:
    python clean_corpus.py <src.jsonl> [<dst.jsonl>]

If <dst.jsonl> is omitted, writes alongside <src> as <stem>.clean.jsonl.

Reports good / bad line counts. Prints the first few bad lines (truncated) so
you can tell whether corruption is sparse (a few bad rows → safe to skip) or
catastrophic (file truncated mid-stream → must re-download).
"""

import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        print("usage: clean_corpus.py <src.jsonl> [<dst.jsonl>]", file=sys.stderr)
        sys.exit(2)

    src = sys.argv[1]
    if len(sys.argv) >= 3:
        dst = sys.argv[2]
    else:
        root, ext = os.path.splitext(src)
        dst = root + ".clean" + ext

    good = 0
    bad = 0
    sample_bad = []

    with open(src, encoding="utf-8", errors="replace") as f, open(dst, "w", encoding="utf-8") as o:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
            except Exception as e:
                bad += 1
                if len(sample_bad) < 10:
                    sample_bad.append((i, str(e)[:80], line[:120].rstrip()))
                continue
            o.write(json.dumps(obj, ensure_ascii=False) + "\n")
            good += 1
            if good % 1_000_000 == 0:
                print(f"  ... {good:,} good lines so far", flush=True)

    print()
    print(f"src : {src}")
    print(f"dst : {dst}")
    print(f"good: {good:,}")
    print(f"bad : {bad:,}")
    if sample_bad:
        print("\nfirst bad lines:")
        for i, err, snippet in sample_bad:
            print(f"  line {i}: {err}\n           {snippet!r}")

    if good == 0:
        print("\nFATAL: zero good lines, the file looks completely broken.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
