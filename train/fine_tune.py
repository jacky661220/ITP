import os, sys, random, csv, numpy as np, torch
from dataclasses import dataclass
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AutoConfig
import torch.optim as optim
from collections import Counter

random.seed(42); np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID2LABEL = {0: "phish", 1: "good"}
LABEL2ID = {"phish": 0, "good": 1}

@dataclass
class URLItem:
    text: str
    label: int

class URLDataset(Dataset):
    def __init__(self, path):
        self.items: List[URLItem] = []
        with open(path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            assert "url" in r.fieldnames and "label" in r.fieldnames
            for row in r:
                u = row["url"].strip()
                try:
                    lab = int(row["label"])
                except:
                    continue
                if lab in (0,1):
                    self.items.append(URLItem(u, lab))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def describe_dataset(name, ds: URLDataset):
    labels = [it.label for it in ds.items]
    c = Counter(labels)
    tot = len(labels)
    print(f"[INFO] {name}: total={tot}, phish(0)={c.get(0,0)}, good(1)={c.get(1,0)}, good_ratio={c.get(1,0)/max(1,tot):.4f}")

def build_sampler(dataset: URLDataset) -> Tuple[WeightedRandomSampler, torch.Tensor]:
    labels = np.array([it.label for it in dataset.items])
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(labels), replacement=True)
    return sampler, torch.tensor(class_weights, dtype=torch.float)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss

def collate_fn(batch: List[URLItem], tokenizer, max_len: int):
    texts = [b.text for b in batch]
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc, labels

def train_epoch(model, loss_fct, loader, optimizer, scheduler, log_every=200):
    model.train()
    total = 0.0
    for step, (enc, labels) in enumerate(loader, 1):
        enc = {k:v.to(DEVICE) for k,v in enc.items()}
        labels = labels.to(DEVICE)
        out = model(**enc)
        logits = out.logits
        if step == 1:
            with torch.no_grad():
                m = logits.mean().item()
                s = logits.std().item()
            print(f"[DEBUG] logits mean={m:.4f}, std={s:.4f}")
        loss = loss_fct(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        if step % log_every == 0:
            print(f"[TRAIN] step={step}, loss={loss.item():.4f}")
    return total / len(loader)

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    probs_list, labels_list = [], []
    for enc, labels in loader:
        enc = {k:v.to(DEVICE) for k,v in enc.items()}
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        probs_list.append(probs)
        labels_list.append(labels.numpy())
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return probs, labels

def scan_threshold(probs, labels, start=0.05, end=0.95, steps=91):
    scores = probs[:, LABEL2ID["phish"]]
    best = {"thr": 0.5, "f1": -1}
    print("threshold, f1_phish, prec_phish, rec_phish, f1_good, prec_good, rec_good, macro_f1")
    for t in np.linspace(start, end, steps):
        pred = np.where(scores >= t, LABEL2ID["phish"], LABEL2ID["good"])
        p0, r0, f10, _ = precision_recall_fscore_support(labels, pred, labels=[LABEL2ID["phish"]], average=None, zero_division=0)
        p1, r1, f11, _ = precision_recall_fscore_support(labels, pred, labels=[LABEL2ID["good"]], average=None, zero_division=0)
        macro = f1_score(labels, pred, average="macro", zero_division=0)
        f0 = f10[0]; f1v = f11[0]
        print(f"{t:.2f}, {f0:.4f}, {p0[0]:.4f}, {r0[0]:.4f}, {f1v:.4f}, {p1[0]:.4f}, {r1[0]:.4f}, {macro:.4f}")
        if f0 > best["f1"]:
            best = {"thr": float(t), "f1": float(f0), "detail": (float(f0), float(p0[0]), float(r0[0]), float(f1v), float(p1[0]), float(r1[0]), float(macro))}
    return best

def make_loader(ds, tokenizer, max_len, batch_size, shuffle=False, sampler=None, mix_neg_upsampling=False):
    if mix_neg_upsampling:
        # 簡單上采樣：把少數類倍增到近似目標比例
        labels = [it.label for it in ds.items]
        c = Counter(labels); n0, n1 = c.get(0,0), c.get(1,0)
        if n1>0 and n1/(n0+n1) < 0.2:
            factor = max(1, int((0.2*(n0+n1))/max(1,n1)))
            items = ds.items + [it for it in ds.items if it.label==1 for _ in range(factor-1)]
            class Tmp(Dataset):
                def __init__(self, xs): self.xs = xs
                def __len__(self): return len(self.xs)
                def __getitem__(self, i): return self.xs[i]
            ds = Tmp(items)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_len)
    )

def main():
    data_dir = "."
    train_path = os.path.join(data_dir, "train.csv")
    val_path   = os.path.join(data_dir, "val.csv")
    test_path  = os.path.join(data_dir, "test.csv")

    train_ds = URLDataset(train_path)
    val_ds   = URLDataset(val_path)
    test_ds  = URLDataset(test_path)
    describe_dataset("train", train_ds)
    describe_dataset("val", val_ds)
    describe_dataset("test", test_ds)

    model_name = os.environ.get("HF_MODEL", "CrabInHoney/urlbert-tiny-v4-phishing-classifier")
    print(f"[INFO] Loading tokenizer/model from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(f"[INFO] model.config.id2label: {model.config.id2label}")
    print(f"[INFO] model.config.label2id: {model.config.label2id}")
    model.to(DEVICE)

    max_len_env = os.environ.get("MAX_LEN")
    if max_len_env:
        max_len = int(max_len_env)
    else:
        max_len = int(getattr(cfg, "max_position_embeddings", 128))
    max_len = int(min(max_len, 512))
    tokenizer.model_max_length = max_len
    print(f"[INFO] Using max_len={max_len}")

    # 選項
    loss_type = os.environ.get("LOSS", "ce").lower()  # ce 或 focal
    use_class_weight = int(os.environ.get("USE_CLASS_WEIGHT", "0")) == 1
    use_sampler = int(os.environ.get("USE_SAMPLER", "0")) == 1
    mix_neg_upsampling = int(os.environ.get("MIX_NEG_UPSAMPLING", "0")) == 1

    # 權重與採樣器
    if use_sampler:
        train_sampler, class_weights = build_sampler(train_ds)
    else:
        train_sampler, class_weights = None, torch.tensor([1.0, 1.0], dtype=torch.float)
    alpha = class_weights.to(DEVICE).float() if use_class_weight else None

    if loss_type == "focal":
        gamma = float(os.environ.get("FOCAL_GAMMA", 2.0))
        loss_fct = FocalLoss(alpha=alpha, gamma=gamma)
        print(f"[INFO] Using FocalLoss gamma={gamma}, class_weight={'on' if use_class_weight else 'off'}")
    else:
        if use_class_weight and alpha is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=alpha)
            print("[INFO] Using CrossEntropyLoss with class weights")
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            print("[INFO] Using CrossEntropyLoss (no class weight)")

    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    epochs = int(os.environ.get("EPOCHS", 3))
    lr = float(os.environ.get("LR", 5e-5))

    train_loader = make_loader(train_ds, tokenizer, max_len, batch_size, shuffle=not use_sampler, sampler=train_sampler, mix_neg_upsampling=mix_neg_upsampling)
    val_loader = make_loader(val_ds, tokenizer, max_len, batch_size, shuffle=False)
    test_loader = make_loader(test_ds, tokenizer, max_len, batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    warmup = max(1, int(0.1*total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    best_state = None
    best_metric = -1.0

    for ep in range(1, epochs+1):
        tr_loss = train_epoch(model, loss_fct, train_loader, optimizer, scheduler)
        val_probs, val_labels = eval_model(model, val_loader)
        val_pred = np.argmax(val_probs, axis=1)
        report = classification_report(val_labels, val_pred, labels=[0,1], target_names=["phish","good"], digits=4, zero_division=0)
        print(f"\nEpoch {ep} | train_loss={tr_loss:.4f}\nVal report:\n{report}")
        # 以 phish 類別 F1 作為早停指標
        p0, r0, f10, _ = precision_recall_fscore_support(val_labels, val_pred, labels=[0], average=None, zero_division=0)
        f1_phish = f10[0]
        if f1_phish > best_metric:
            best_metric = f1_phish
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # 閾值掃描（更細）
    val_probs, val_labels = eval_model(model, val_loader)
    best = scan_threshold(val_probs, val_labels, start=0.05, end=0.95, steps=91)
    thr = best["thr"]
    print(f"\nBest threshold on val for phish-F1: {thr:.2f}, details={best['detail']}")

    # 測試集報告（用閾值）
    test_probs, test_labels = eval_model(model, test_loader)
    phish_scores = test_probs[:, LABEL2ID["phish"]]
    test_pred = np.where(phish_scores >= thr, LABEL2ID["phish"], LABEL2ID["good"])
    print("\nTest report with tuned threshold:")
    print(classification_report(test_labels, test_pred, labels=[0,1], target_names=["phish","good"], digits=4, zero_division=0))

    # 白名單回歸測試（關鍵樣本必須判 phish）
    whitelist = [
        "http://www-livibank-com.meizuangan.cn/en/",
        "https://bochk.p4fdg.top/chinaBank1.html",
        "https://carousell-hk.ebuydirect.shop/login/338JE417OM625Z3630661",
    ]
    print("\n[Sanity] Whitelist check:")
    model.eval()
    for u in whitelist:
        enc = tokenizer([u], truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
        out = model(**enc)
        prob = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()[0]
        p_phish = float(prob[LABEL2ID["phish"]])
        pred = "phish" if p_phish >= thr else "good"
        print(f"  {u} => phish={p_phish:.4f}, pred={pred}")

    out_dir = os.environ.get("OUT_DIR", "checkpoints/urlclf-v1")
    os.makedirs(out_dir, exist_ok=True)
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "threshold.txt"), "w", encoding="utf-8") as f:
        f.write(str(thr))
    print(f"Saved model to {out_dir} and threshold={thr:.2f}")

if __name__ == "__main__":
    main()