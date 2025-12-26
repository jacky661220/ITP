import csv, os, sys, random
from urllib.parse import urlparse
from collections import defaultdict, Counter
import tldextract

random.seed(42)

def get_keys(u: str):
    try:
        p = urlparse(u)
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        ext = tldextract.extract(netloc)
        etld1 = ".".join([ext.domain, ext.suffix]) if ext.suffix else netloc
        return netloc, etld1
    except:
        return "", ""

def read_all(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        assert "url" in r.fieldnames and "label" in r.fieldnames, "CSV must have url,label"
        for row in r:
            u = row["url"].strip()
            try:
                lab = int(row["label"])
            except:
                continue
            if lab not in (0,1): continue
            netloc, etld1 = get_keys(u)
            if not netloc or not etld1: continue
            rows.append((u, lab, netloc, etld1))
    return rows

def group_by_etld1(rows):
    groups = defaultdict(list)
    for u, lab, netloc, etld1 in rows:
        groups[etld1].append((netloc, u, lab))
    return groups

def split_groups(groups, ratios=(0.8,0.1,0.1), min_quota=None):
    # min_quota: {"train": {0: min_phish, 1: min_good}, "val": {...}, "test": {...}}
    etld1s = list(groups.keys())
    random.shuffle(etld1s)

    # 先計算總數
    total = sum(len(groups[k]) for k in etld1s)
    tgt_train = int(total * ratios[0])
    tgt_val   = int(total * ratios[1])

    # 依照組大小排序（大組優先分配）
    def score(k):
        labs = [lab for _,_,lab in groups[k]]
        c = Counter(labs)
        return -(len(labs)), -abs(c.get(1,0)-c.get(0,0))
    etld1s.sort(key=score)

    splits = {"train": [], "val": [], "test": []}
    cnt = {"train":0, "val":0, "test":0}

    for k in etld1s:
        bucket = groups[k]
        n = len(bucket)
        if cnt["train"] + n <= tgt_train:
            splits["train"].extend([(k, netloc, u, lab) for netloc,u,lab in bucket])
            cnt["train"] += n
        elif cnt["val"] + n <= tgt_val:
            splits["val"].extend([(k, netloc, u, lab) for netloc,u,lab in bucket])
            cnt["val"] += n
        else:
            splits["test"].extend([(k, netloc, u, lab) for netloc,u,lab in bucket])
            cnt["test"] += n

    # 強制最小配額（如設定）
    if min_quota:
        for name in ("train", "val", "test"):
            labs = [lab for _,_,_,lab in splits[name]]
            c = Counter(labs)
            need0 = max(0, min_quota[name].get(0,0) - c.get(0,0))
            need1 = max(0, min_quota[name].get(1,0) - c.get(1,0))
            if need0==0 and need1==0:
                continue
            # 從其他 split 借：簡單策略，從最多的 split 拿相應類型補到當前 split
            for cls, need in ((0,need0),(1,need1)):
                if need<=0: continue
                # 在其他 splits 找該類並搬移
                for other in ("train","val","test"):
                    if other==name: continue
                    if need<=0: break
                    pick_idx = [i for i,(_,_,_,lab) in enumerate(splits[other]) if lab==cls]
                    random.shuffle(pick_idx)
                    while pick_idx and need>0:
                        i = pick_idx.pop()
                        item = splits[other].pop(i)
                        splits[name].append(item)
                        need -= 1

    return splits

def postprocess_train_balance(train_rows, upsample_minority=False, downsample_majority=False, target_ratio=0.2):
    # 對訓練集可選擇上采樣少數類或下采樣多數類
    from math import ceil
    labs = [lab for _,_,_,lab in train_rows]
    c = Counter(labs)
    n0, n1 = c.get(0,0), c.get(1,0)
    if upsample_minority and n1>0 and n0>0:
        ratio = n1 / (n0 + n1)
        if ratio < target_ratio:
            # 上采樣 1 類
            need = int(target_ratio*(n0+n1)) - n1
            pos_items = [x for x in train_rows if x[3]==1]
            for _ in range(max(0, need)):
                train_rows.append(random.choice(pos_items))
    if downsample_majority and n0>0 and n1>0:
        ratio = n1 / len(train_rows)
        if ratio < target_ratio:
            # 下采樣 0 類
            desired_n1 = int(target_ratio * len(train_rows))
            desired_n0 = int((1-target_ratio) * len(train_rows))
            neg_items = [x for x in train_rows if x[3]==0]
            pos_items = [x for x in train_rows if x[3]==1]
            random.shuffle(neg_items)
            new_rows = pos_items + neg_items[:desired_n0]
            train_rows[:] = new_rows
    return train_rows

def write_split(splits, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for name in ("train","val","test"):
        path = os.path.join(out_dir, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url","label"])
            labs = []
            for _, _, u, lab in splits[name]:
                w.writerow([u, lab])
                labs.append(lab)
        c = Counter(labs)
        stats[name] = {"total": len(labs), "phish(0)": c.get(0,0), "good(1)": c.get(1,0), "ratio_good": round(c.get(1,0)/max(1,len(labs)),4)}
    return stats

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    ratios = (0.8, 0.1, 0.1)

    rows = read_all(in_path)
    all_counts = Counter([lab for _,lab,_,_ in rows])
    print(f"Loaded {len(rows)} rows from {in_path}")
    print(f"Global label counts: {dict(all_counts)}")

    groups = group_by_etld1(rows)
    print(f"Unique eTLD+1 groups: {len(groups)}")

    # 設定最小配額（可依你資料量調整）
    min_quota = {
        "train": {0: 1000, 1: 400},   # 至少這麼多
        "val":   {0: 200,  1: 50},
        "test":  {0: 200,  1: 50},
    }
    splits = split_groups(groups, ratios=ratios, min_quota=min_quota)

    # 選擇是否對訓練集做上/下采樣
    splits["train"] = postprocess_train_balance(
        splits["train"], 
        upsample_minority=True, 
        downsample_majority=False, 
        target_ratio=0.2  # 希望 train 至少 20% good
    )

    stats = write_split(splits, out_dir)
    for k,v in stats.items():
        print(k, v)