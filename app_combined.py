import html
import logging
import re
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    pipeline as hf_pipeline,
)

# On Windows, Playwright needs the Proactor event loop to support subprocess
import sys, asyncio
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# -----------------------
# Global basic settings
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("combined-url-msg-mvp")

# =======================
# Section 1: URL classification (from appv2.py)
# =======================

URL_HF_MODEL = "train/checkpoints/urlclf-v1"
URL_LABEL_MAP = {"LABEL_0": "good", "LABEL_1": "phish"}

URL_REGEX = re.compile(r'^(?:https?://)?[^\s/$.?#].[^\s]*$', re.IGNORECASE)

def normalize_url(u: str) -> str:
    u = (u or "").strip()
    u = html.unescape(u)
    u = unquote(u)
    if not u.lower().startswith(("http://", "https://")):
        u = "http://" + u
    if not URL_REGEX.match(u):
        return u
    try:
        parts = urlparse(u)
        scheme = parts.scheme or "http"
        host = parts.netloc or parts.path
        path = parts.path if parts.netloc else ""
        query = f"?{parts.query}" if parts.query else ""

        # Prevent returning just "http://": if host is still empty, return original u (avoid invalid URL)
        if not host:
            return u

        return f"{scheme}://{host}{path}{query}"
    except Exception:
        return u

# ---- Short URL expansion related ----
SHORTENER_DOMAINS = {
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "buff.ly", "rebrand.ly",
    "is.gd", "t.ly", "cutt.ly", "mcaf.ee", "s.id", "lnkd.in", "fb.me", "yhoo.it",
    "trib.al", "ift.tt", "dlvr.it", "supr.link", "shorte.st", "adf.ly", "rb.gy",
    "shorturl.at", "bit.do", "v.gd", "qr.ae", "amzn.to", "m.tb.cn",
    "vk.cc", "x.co", "u.to", "clck.ru", "2.gp", "spr.ly", "snip.ly",
    "t.cn", "t2m.io", "urlz.fr", "short.cm", "kutt.it",
}

def is_shortener(host: str) -> bool:
    host = (host or "").lower().strip()
    if not host:
        return False
    # Subdomains are also treated as shorteners
    for dom in SHORTENER_DOMAINS:
        if host == dom or host.endswith("." + dom):
            return True
    return False

async def expand_short_url(raw_url: str, max_redirects: int = 5, timeout_ms: int = 4000) -> str:
    try:
        import httpx
    except Exception:
        logger.warning("[SHORT] httpx not installed, skipping expansion")
        return raw_url

    from urllib.parse import urljoin

    # Playwright-based expansion enabled only for shorturl.at (sync API + asyncio.to_thread)
    async def _browser_expand_if_needed(current_url: str, timeout_ms: int) -> str:
        try:
            host = urlparse(current_url).netloc.lower()
        except Exception:
            host = ""
        if "shorturl.at" not in host:
            return current_url
        try:
            def _run_sync_pw(url: str, nav_timeout: int) -> str:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                    ctx = browser.new_context(user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ))
                    page = ctx.new_page()
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout)
                    except Exception:
                        try:
                            page.goto(url, wait_until="load", timeout=nav_timeout)
                        except Exception:
                            pass
                    # Wait for possible JS/Meta refresh
                    try:
                        page.wait_for_timeout(400)
                    except Exception:
                        pass
                    final_url = page.url or url
                    ctx.close()
                    browser.close()
                    return final_url

            nav_timeout = max(2500, min(timeout_ms, 8000))
            final_url = await asyncio.to_thread(_run_sync_pw, current_url, nav_timeout)

            # Strictly validate final URL: must be valid http/https with netloc
            try:
                pu = urlparse(final_url)
                scheme_ok = pu.scheme in ("http", "https")
                host_ok = bool(pu.netloc)
                invalid_scheme = pu.scheme in ("about", "chrome", "data", "blob")
            except Exception:
                scheme_ok = host_ok = False
                invalid_scheme = True

            if not final_url or invalid_scheme or not (scheme_ok and host_ok):
                logger.info(f"[SHORT] Playwright returned invalid URL (fallback applied): {final_url!r}")
                return current_url

            if final_url != current_url:
                logger.info(f"[SHORT] Playwright expanded: {current_url} -> {final_url}")
            return final_url
        except Exception as e:
            logger.info(f"[SHORT] Playwright expansion failed: {repr(e)}")
            return current_url

    def _browser_headers():
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-HK,zh-TW;q=0.9,zh-CN;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Referer": "https://www.google.com/",
            "Origin": "https://www.google.com",
            "DNT": "1",
            "sec-ch-ua": '"Chromium";v="120", "Not A(Brand";v="24", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }

    def _abs_location(resp: "httpx.Response") -> Optional[str]:
        loc = resp.headers.get("location") or resp.headers.get("Location")
        if not loc:
            return None
        try:
            base_url = str(resp.request.url)
            return urljoin(base_url, loc)
        except Exception:
            return loc

    def _follow_once(client: "httpx.Client", url: str) -> tuple[str, int]:
        # Try HEAD first
        try:
            resp = client.head(url, headers=_browser_headers())
        except httpx.HTTPError:
            resp = None

        if resp is not None and 300 <= resp.status_code < 400:
            nxt = _abs_location(resp)
            if nxt:
                return nxt, resp.status_code

        # Then GET
        try:
            resp = client.get(url, headers=_browser_headers())
        except httpx.HTTPError:
            return url, 0

        if 300 <= resp.status_code < 400:
            nxt = _abs_location(resp)
            if nxt:
                return nxt, resp.status_code

        if resp.status_code == 403:
            # Read body -> wait a bit -> GET a few more times (total extra delay < 700ms)
            _ = resp.text
            import time as _t
            for delay in (0.25, 0.35):
                _t.sleep(delay)
                try:
                    resp2 = client.get(url, headers=_browser_headers())
                    if 300 <= resp2.status_code < 400:
                        nxt = _abs_location(resp2)
                        if nxt:
                            return nxt, resp2.status_code
                    if resp2.status_code != 403:
                        return url, resp2.status_code
                except httpx.HTTPError:
                    pass
            return url, 403

        return url, resp.status_code

    try:
        start = normalize_url(raw_url)
        timeout = httpx.Timeout(timeout_ms / 1000.0)
        current = start
        visited = set([current])
        last_status = 0

        with httpx.Client(follow_redirects=False, timeout=timeout) as client:
            # Multi-step manual redirects
            for _ in range(max_redirects):
                nxt, code = _follow_once(client, current)
                last_status = code
                if nxt != current:
                    if nxt in visited:
                        logger.info(f"[SHORT] Detected redirect loop: {nxt}")
                        break
                    visited.add(nxt)
                    current = nxt
                    continue
                else:
                    break

            # Final attempt with auto-follow (preserving cookies)
            try:
                r = client.get(current, headers=_browser_headers(), follow_redirects=True)
                current = str(r.url)
                last_status = r.status_code
            except httpx.HTTPError as he:
                logger.info(f"[SHORT] auto-follow expansion error: {he}")

        # If still at a shortener and likely blocked (403/no progress), try browser path for shorturl.at
        try:
            final_host = urlparse(current).netloc.lower()
        except Exception:
            final_host = ""
        if is_shortener(final_host) and (last_status in (0, 403) or "shorturl.at" in final_host):
            current2 = await _browser_expand_if_needed(current, timeout_ms)
            if current2 and current2 != current:
                current = current2

        return current
    except Exception as e:
        logger.exception(f"[SHORT] Failed to expand short URL: {e}")
        return raw_url

logger.info("Loading URL model and tokenizer...")
url_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"[URL] Using device: {url_device}")

url_tokenizer = BertTokenizerFast.from_pretrained(URL_HF_MODEL)
url_model = BertForSequenceClassification.from_pretrained(URL_HF_MODEL)
url_model.to(url_device)

url_classifier = hf_pipeline(
    "text-classification",
    model=url_model,
    tokenizer=url_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True
)
logger.info("[URL] Model loaded.")

# =======================
# Section 2: Message tone/intent classification
# =======================

MSG_HF_MODEL = "joeddav/xlm-roberta-large-xnli"
MSG_HF_TOKENIZER = "xlm-roberta-large"
MSG_MULTI_LABEL = False  # mutually exclusive classification
TONE_LABELS = ["urgent", "promotion", "informational", "transactional", "question"]

# Dependency checks (sentencepiece / protobuf)
msg_deps = {"sentencepiece": False, "protobuf": False}
try:
    import sentencepiece  # noqa: F401
    msg_deps["sentencepiece"] = True
except Exception:
    pass
try:
    import google.protobuf  # noqa: F401
    msg_deps["protobuf"] = True
except Exception:
    pass

logger.info("Loading zero-shot classification pipeline for messages...")
msg_device = 0 if torch.cuda.is_available() else -1

msg_classifier: Optional[object] = None
if not all(msg_deps.values()):
    missing = [k for k, ok in msg_deps.items() if not ok]
    logger.error(f"[MSG] Missing dependencies: {missing}. Please pip install {' '.join(missing)} first")
else:
    try:
        msg_classifier = hf_pipeline(
            "zero-shot-classification",
            model=MSG_HF_MODEL,
            tokenizer=MSG_HF_TOKENIZER,
            device=msg_device,
        )
        logger.info(f"[MSG] Model loaded. Using device: {'cuda' if msg_device == 0 else 'cpu'}")
    except Exception:
        logger.exception("[MSG] Failed to load model or tokenizer")
        raise

# -----------------------
# Pydantic models
# -----------------------
class ClassifyIn(BaseModel):
    url: Optional[str] = Field(None, description="URL to check, e.g., https://example.com/login")
    message: Optional[str] = Field(None, description="Message content to check (supports Chinese/English/multilingual)")

class UnifiedOut(BaseModel):
    input_url: Optional[str] = None
    normalized_url: Optional[str] = None
    was_input_url: Optional[bool] = None
    top_label: Optional[str] = None
    top_score: Optional[float] = None
    tone: Optional[str] = None

# -----------------------
# Utility functions
# -----------------------
async def classify_url_inner(
    raw_url: str,
    expand_short: bool = True,
    max_redirects: int = 5,
    timeout_ms: int = 4000,
) -> Tuple[str, str, float, bool]:
    """
    Returns:
    - target_url: URL used for classification (after expansion/normalization)
    - top_label, top_score: classification result
    - was_input_url: True if target_url equals input (no expansion); False if expanded
    """
    if not raw_url or not isinstance(raw_url, str):
        raise HTTPException(status_code=400, detail="The 'url' field must be a non-empty string")

    prelim = normalize_url(raw_url)  # normalized version of the input
    was_input_url: bool = True  # default: not expanded
    target_url = prelim

    # If it's a shortener and expansion is allowed, try expansion
    try:
        host = urlparse(prelim).netloc.lower()
    except Exception:
        host = ""
    if expand_short and is_shortener(host):
        expanded = await expand_short_url(prelim, max_redirects=max_redirects, timeout_ms=timeout_ms)

        # Only adopt expanded if it's a valid http/https with netloc; otherwise fall back to prelim
        try:
            pu = urlparse(expanded)
            if pu.scheme in ("http", "https") and pu.netloc:
                candidate = normalize_url(expanded)
                if candidate != prelim:
                    logger.info(f"[SHORT] Expanded short URL: {prelim} -> {candidate}")
                    target_url = candidate
                    was_input_url = False
                else:
                    logger.info(f"[SHORT] Could not further expand, using original: {prelim}")
                    target_url = prelim
                    was_input_url = True
            else:
                logger.info(f"[SHORT] Expansion result invalid (keeping original): {expanded!r}")
                target_url = prelim
                was_input_url = True
        except Exception:
            target_url = prelim
            was_input_url = True

    # Classify with target_url
    try:
        results = url_classifier(target_url)
        scores = results[0]
        pred_map = {}
        for item in scores:
            label = URL_LABEL_MAP.get(item["label"], item["label"])
            pred_map[label] = float(item["score"])
        top_label = max(pred_map, key=pred_map.get)
        top_score = round(pred_map[top_label], 4)
        return target_url, top_label, top_score, was_input_url
    except Exception as e:
        logger.exception("[URL] classification failed")
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

def classify_message_inner(text: str) -> str:
    if msg_classifier is None:
        missing = [k for k, ok in msg_deps.items() if not ok]
        raise HTTPException(
            status_code=500,
            detail=f"Missing dependencies: {missing}. Please install in current environment: pip install {' '.join(missing)}"
        )
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="The 'message' field must be a non-empty string")

    try:
        res = msg_classifier(text, candidate_labels=TONE_LABELS, multi_label=MSG_MULTI_LABEL)
        labels_ranked = res["labels"]
        scores_ranked = res["scores"]
        pred_map = {lbl: float(score) for lbl, score in zip(labels_ranked, scores_ranked)}
        top_label = max(pred_map, key=pred_map.get)
        return top_label
    except Exception as e:
        logger.exception("[MSG] classification failed")
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

# -----------------------
# Launch FastAPI (combined)
# -----------------------
app = FastAPI(
    title="Combined URL + Message Classifier",
    version="1.6.0",
    description="Automatically expands short URLs before classification; returns input_url, normalized_url, was_input_url, top_label, top_score, tone.",
)

@app.get("/")
def root():
    return {"status": "ok", "endpoint": "/classify"}

@app.get("/health")
def health():
    return {
        "status": "working",
        "cuda": torch.cuda.is_available(),
        "url": {"model": URL_HF_MODEL, "device": str(url_device)},
        "message": {
            "model": MSG_HF_MODEL,
            "tokenizer": MSG_HF_TOKENIZER,
            "device": "cuda" if msg_device == 0 else "cpu",
            "deps": msg_deps,
            "tone_labels": TONE_LABELS,
            "multi_label": MSG_MULTI_LABEL,
        },
        "shortener_domains_count": len(SHORTENER_DOMAINS),
    }

@app.get("/health_url")
def health_url():
    return {
        "status": "working",
        "model": URL_HF_MODEL,
        "device": str(url_device),
        "shortener_domains_count": len(SHORTENER_DOMAINS),
    }

@app.get("/health_msg")
def health_msg():
    return {
        "status": "working",
        "model": MSG_HF_MODEL,
        "tokenizer": MSG_HF_TOKENIZER,
        "device": "cuda" if msg_device == 0 else "cpu",
        "deps": msg_deps,
        "tone_labels": TONE_LABELS,
        "multi_label": MSG_MULTI_LABEL,
    }

# New unified endpoint
@app.post("/classify", response_model=UnifiedOut)
async def classify(
    payload: ClassifyIn,
    expand_short: bool = Query(True, description="Whether to auto-expand short URLs"),
    max_redirects: int = Query(5, ge=0, le=10, description="Maximum redirect hops for expansion"),
    timeout_ms: int = Query(4000, ge=500, le=15000, description="Per-step timeout (milliseconds) for expansion"),
):
    if not payload.url and not payload.message:
        raise HTTPException(status_code=400, detail="Provide at least one of: url or message")

    input_url: Optional[str] = None
    normalized_url = top_label = tone = None
    was_input_url: Optional[bool] = None
    top_score: Optional[float] = None

    if payload.url:
        input_url = payload.url
        normalized_url, top_label, top_score, was_input_url = await classify_url_inner(
            payload.url,
            expand_short=expand_short,
            max_redirects=max_redirects,
            timeout_ms=timeout_ms,
        )

    if payload.message:
        tone = classify_message_inner(payload.message)

    return UnifiedOut(
        input_url=input_url,
        normalized_url=normalized_url,
        was_input_url=was_input_url,
        top_label=top_label,
        top_score=top_score,
        tone=tone,
    )

# Run command (local development):
# uvicorn app_combined:app --host 0.0.0.0 --port 8000 --reload