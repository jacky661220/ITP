import csv, re, sys, os, unicodedata
from urllib.parse import urlparse, unquote, parse_qs, urlsplit

URL_RE = re.compile(r'((?:https?://|http://|https://|www\.)[^\s,<>()\[\]{}"\'|]+)', re.IGNORECASE)

# 常見包裝/污染清理：增加 urldefense / sendgrid 的多種鍵位
WRAPPER_PATTERNS = [
    # Proofpoint/URLDefense V3/v2 等
    re.compile(r'https?://urldefense\.com/v3/__([^;]+);', re.IGNORECASE),
    # 也有直接 urldefense 把原始 URL 放在 __https://xxxx__; 形式
    re.compile(r'__https?://([^;]+);', re.IGNORECASE),

    # Sendgrid 典型
    re.compile(r'https?://u\d+\.ct\.sendgrid\.net/wf/click[^?]*\?([^#]+)', re.IGNORECASE),
    re.compile(r'https?://.*?/ls/click[^?]*\?([^#]+)', re.IGNORECASE),
    re.compile(r'https?://.*?/wf/click[^?]*\?([^#]+)', re.IGNORECASE),

    # 其他 redirect 包裝
    re.compile(r'https?://[^/\s]+/redirect[^?]*\?([^#]+)', re.IGNORECASE),
]

BLACKLIST_TOKENS = {
    "fraudulent mobile application",
}

def strip_control_chars(s: str) -> str:
    if s is None:
        return ""
    return "".join(ch for ch in s if ch == "\t" or (unicodedata.category(ch)[0] != "C"))

def normalize_text(s: str) -> str:
    s = strip_control_chars(str(s))
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

def soft_fix_scheme(u: str) -> str:
    u = u.replace('hxxp://', 'http://').replace('hxxps://', 'https://')
    u = re.sub(r'^https?\s*]\s*//', lambda m: m.group(0).replace(']', ''), u, flags=re.IGNORECASE)
    u = u.replace('http[:]//', 'http://').replace('https[:]//', 'https://')
    u = u.replace('http:///', 'http://').replace('https:///', 'https://')
    u = u.replace('xxp://', 'http://').replace('xp://', 'http://')
    return u

def extract_inner_wrapped(u: str) -> str:
    s = u
    for pat in WRAPPER_PATTERNS:
        m = pat.search(s)
        if not m: 
            continue
        try:
            grp = m.group(1)
        except:
            continue
        # urldefense 的 __https://xxx__; 直接取出即可
        if s.startswith("https://urldefense.com") or s.startswith("http://urldefense.com") or grp.startswith("http"):
            try:
                inner = grp
                if inner.startswith("http%3A") or inner.startswith("https%3A"):
                    inner = unquote(inner)
                return inner
            except:
                pass
        # 對 sendgrid/query 的情形，解析查找可能的鍵
        try:
            q = parse_qs(grp)
            cand_keys = ["u","upn","url","_u","target","r","redirect","q"]
            for k in cand_keys:
                if k in q and len(q[k])>0:
                    inner = q[k][0]
                    inner = unquote(inner)
                    return inner
        except:
            pass
    return u

def is_http_url(u: str) -> bool:
    try:
        if not u.lower().startswith(("http://","https://")):
            u2 = "http://" + u
        else:
            u2 = u
        p = urlparse(u2)
        return bool(p.scheme in ("http","https") and p.netloc and "." in p.netloc)
    except:
        return False

def normalize_url(u: str) -> str:
    u = normalize_text(u)
    if not u or any(tok in u.lower() for tok in BLACKLIST_TOKENS):
        return ""
    u = extract_inner_wrapped(u)
    u = soft_fix_scheme(u)
    if (not u.lower().startswith(("http://","https://"))) and u.lower().startswith("www."):
        u = "http://" + u
    if not u.lower().startswith(("http://","https://")):
        return ""
    u = u.rstrip("?,;)")
    # 去掉兩端多餘括號
    u = u.strip("()[]{}<>")
    return u

def split_cell_urls(s: str):
    s = normalize_text(s)
    pieces = set()
    # 先用正規擷取一輪
    for m in URL_RE.findall(s):
        pieces.add(m)
    # 再用常見分隔切割一輪
    for tok in re.split(r'[\s,;|]+', s):
        if tok.startswith(("http", "www.")):
            pieces.add(tok)
    out = []
    seen = set()
    for cand in pieces:
        u = normalize_url(cand)
        if not u: 
            continue
        if not is_http_url(u): 
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def maybe_fix_tail_label(url: str, label: str):
    # 舊資料可能有 ?0 或 ?1 汙染
    m = re.search(r'\?+([01])$', url)
    if (label is None or label == "") and m:
        lab = m.group(1)
        fixed_url = re.sub(r'\?+[01]$', "", url)
        return fixed_url, lab
    if label in ("0","1"):
        fixed_url = re.sub(r'\?+$', "", url)
        return fixed_url, label
    return url, label

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("IN_CSV", "url data.csv")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUT_CSV", "all.csv")

    if not os.path.exists(in_path):
        print("用法: python clean_csv.py <輸入CSV> <輸出CSV>")
        print(f"找不到輸入檔: {in_path}")
        sys.exit(1)

    out_rows = []
    bad_rows = 0

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        idx_url, idx_label = 0, 1
        if header:
            h = [normalize_text(x).lower() for x in header]
            if "url" in h: idx_url = h.index("url")
            if "label" in h: idx_label = h.index("label")

        for row in r:
            if not row:
                continue
            row = [normalize_text(x) for x in row]
            url_field = row[idx_url] if idx_url < len(row) else ""
            label_field = row[idx_label] if idx_label < len(row) else ""

            lab = None
            try:
                lv = int(label_field)
                if lv in (0,1):
                    lab = str(lv)
            except:
                lab = None

            candidates = split_cell_urls(url_field)
            # 掃描其餘欄位
            for k, cell in enumerate(row):
                if k in (idx_url, idx_label): 
                    continue
                if "http" in cell.lower() or "www." in cell.lower():
                    candidates += split_cell_urls(cell)

            if not candidates:
                continue

            for u in candidates:
                fixed_u, fixed_lab = maybe_fix_tail_label(u, lab)
                if not is_http_url(fixed_u):
                    continue
                if fixed_lab not in ("0","1"):
                    bad_rows += 1
                    continue
                out_rows.append((fixed_u, int(fixed_lab)))

    # 去重保持順序
    out_rows = list(dict.fromkeys(out_rows))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url","label"])
        for u, lab in out_rows:
            w.writerow([u, lab])

    print(f"wrote {len(out_rows)} rows to {out_path}")
    if bad_rows:
        print(f"[INFO] skipped rows due to invalid/missing label or malformed url tail: {bad_rows}")

if __name__ == "__main__":
    main()