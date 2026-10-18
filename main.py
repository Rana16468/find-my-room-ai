"""
ফাইন্ড মাই রুম - দ্রুত সেল্ফ-লার্নিং রেকমেন্ডেশন ইঞ্জিন (v5, সম্পূর্ণ বাংলা, গোপন তথ্য .env ফাইলে)

১,০০,০০০+ রুমের জন্য ডিজাইন:
  * অফলাইন (স্টার্টআপে + প্রতি ৩০ মিনিটে, ব্যাকগ্রাউন্ডে):
      - TF-IDF  -> TruncatedSVD (৬৪ ডাইমেনশন) = আনসুপারভাইজড "প্যাটার্ন" ভেক্টর
      - MiniBatchKMeans                        = রুমের প্যাটার্ন গ্রুপ (ক্লাস্টার)
      - সবকিছু RAM-এ numpy অ্যারে হিসেবে থাকে
  * অনলাইন (প্রতি রিকুয়েস্টে, কয়েক দশ মিলিসেকেন্ড):
      - একটা ম্যাট্রিক্স গুণ + কয়েকটা বুলিয়ান মাস্ক (সব রুমের উপর একসাথে)
      - রুম ধরে ধরে Python লুপ নেই, পুরো কালেকশন স্ক্যান নেই, TF-IDF ফিটিং নেই
      - শুধু যে ১২টা রুম দেখানো হবে সেগুলোই MongoDB থেকে আনা হয়
  * লার্নিং (প্রতিটি ক্লিক/স্কিপে):
      - অনলাইন লজিস্টিক রিগ্রেশন ফিচারের ওজন (weight) ঠিক করে
      - প্রতি রুমের CTR + এক্সপ্লোরেশন বোনাস, যাতে নতুন রুম সুযোগ পায়
"""
import importlib.util
import sys

# ---- নির্ভরশীল প্যাকেজ চেক: কোনোটা না থাকলে ঠিক কোন কমান্ড চালাতে হবে তা বলে দেবে ----
_REQUIRED = {
    "fastapi": "fastapi", "uvicorn": "uvicorn", "pymongo": "pymongo",
    "numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn",
    "joblib": "joblib", "dotenv": "python-dotenv",
}
_missing = [pip for mod, pip in _REQUIRED.items() if importlib.util.find_spec(mod) is None]
if _missing:
    raise SystemExit(
        "\n\nপ্যাকেজ পাওয়া যায়নি: " + ", ".join(_missing) +
        "\nএই Python এনভায়রনমেন্টে ইনস্টল করতে নিচের কমান্ডটি চালান:\n\n    "
        f'"{sys.executable}" -m pip install ' + " ".join(_missing) + "\n"
    )

import html
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from pathlib import Path
from decimal import Decimal
from typing import Any, List, Optional

import joblib
import numpy as np
from bson import ObjectId
from bson.decimal128 import Decimal128
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient, UpdateOne
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("recommender")

# .env ফাইলটি main.py এর পাশের ফোল্ডার থেকেই পড়া হবে (যেখান থেকেই uvicorn চালাও না কেন)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)


def require_env(name: str) -> str:
    """জরুরি সেটিং .env বা সার্ভারের এনভায়রনমেন্ট থেকে নেয়। না পেলে পরিষ্কার বার্তা দিয়ে থেমে যায়।"""
    value = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not value:
        raise SystemExit(
            f"\n\n{name} পাওয়া যায়নি!\n"
            f"প্রজেক্ট ফোল্ডারে '.env' ফাইল বানিয়ে ভেতরে {name}=... লিখুন\n"
            f"(ফাইলের পথ: {ENV_PATH})\n"
            f"হোস্টিংয়ে থাকলে হোস্টিংয়ের Environment Variables-এ {name} যোগ করুন।\n"
        )
    return value


MONGO_URI = require_env("MONGO_URI")        # ডাটাবেস কানেকশন লিংক (গোপন, .env থেকে আসে)
ADMIN_KEY = require_env("ADMIN_KEY")        # /api/v1/admin/reindex এর গোপন কী (.env থেকে আসে)
# কোন কোন ওয়েবসাইট API ব্যবহার করতে পারবে (কমা দিয়ে আলাদা)। ডিফল্ট * = সবাই
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# --- ইনডেক্স / প্যাটার্ন চেনা ---
MAX_TFIDF_FEATURES = 20000
SVD_DIMS = 64
N_CLUSTERS_MAX = 40
MAX_RESULTS = 240                            # সর্বোচ্চ ২৪০টি রুম র‍্যাংক হবে (১২টা করে ২০ পেজ)
INDEX_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_cache.joblib")
INDEX_CACHE_MAX_AGE = 6 * 3600               # রিস্টার্টে ডিস্ক ক্যাশ এর চেয়ে নতুন হলে সেটাই ব্যবহার হবে
REINDEX_EVERY_SEC = 30 * 60                  # নতুন রুম এলে প্রতি ৩০ মিনিটে আবার শেখা
STATS_SYNC_SEC = 5 * 60
WEIGHTS_TTL = 5                              # শেখা ওজন RAM-এ কত সেকেন্ড ক্যাশ থাকবে

# --- অনলাইন লার্নিং ---
FEATURES = [
    "bias", "text", "location", "city", "type", "keyword",
    "session_type", "session_loc", "cluster", "popularity", "ctr",
]
DEFAULT_WEIGHTS = {
    "bias": -2.5, "text": 2.0, "location": 1.5, "city": 1.5, "type": 1.5,
    "keyword": 1.0, "session_type": 0.8, "session_loc": 1.0, "cluster": 0.8,
    "popularity": 0.5, "ctr": 1.0,
}
LEARNING_RATE = 0.15
W_CLIP = 6.0
EXPLORE_C = 0.6                              # কম ইমপ্রেশনের রুমের জন্য বোনাস
EVENT_TARGET = {"click": 1.0, "save": 1.0, "contact": 1.0, "skip": 0.0}
EVENT_STRENGTH = {"click": 1.0, "save": 1.5, "contact": 2.0, "skip": 0.5}

# ===============================================================
# ২. ডাটাবেস
# ===============================================================
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client.get_default_database(default="rishabhbhard")   # URI-তে ডাটাবেসের নাম না থাকলেও চলবে
rooms_collection = db["list_rooms"]
model_col = db["rec_model"]
impressions_col = db["rec_impressions"]
stats_col = db["rec_room_stats"]
users_col = db["users"]


def setup_db():
    """স্টার্টআপে প্রয়োজনীয় ইনডেক্স ও গ্লোবাল মডেল ডকুমেন্ট তৈরি করে।"""
    try:
        impressions_col.create_index("created_at", expireAfterSeconds=7 * 24 * 3600)
        impressions_col.create_index([("request_id", 1), ("room_id", 1)])
        stats_col.create_index("room_id", unique=True)
        model_col.update_one(
            {"_id": "global"},
            {"$setOnInsert": {"weights": DEFAULT_WEIGHTS, "updates": 0,
                              "impressions": 0, "positives": 0}},
            upsert=True,
        )
        logger.info("MongoDB ইনডেক্স ও মডেল ডকুমেন্ট প্রস্তুত।")
    except Exception as e:
        logger.warning(f"স্টার্টআপে ডাটাবেস সেটআপে সমস্যা: {e}")


# ===============================================================
# ৩. স্কিমা (রিকুয়েস্টের ডাটার গঠন)
# ===============================================================
class RecommendationRequest(BaseModel):
    viewed_room_ids: Optional[List[str]] = []
    search_location: Optional[str] = ""
    search_type: Optional[str] = ""
    keywords: Optional[List[str]] = []
    user_detected_city: Optional[str] = ""


class FeedbackRequest(BaseModel):
    request_id: str
    room_id: str
    event: str  # click | save | contact | skip


# ===============================================================
# ৪. ছোট ছোট সহায়ক ফাংশন
# ===============================================================
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def safe_str(val) -> str:
    """যেকোনো মানকে নিরাপদে স্ট্রিং বানায় (None হলে খালি, লিস্ট হলে জোড়া লাগায়)।"""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(v) for v in val if v is not None)
    return str(val)


def strip_html(raw) -> str:
    """দ্রুত regex দিয়ে HTML ট্যাগ মুছে ফেলে (১ লাখ রুমের জন্য BeautifulSoup খুব ধীর)।"""
    s = safe_str(raw)
    if not s:
        return ""
    return WS_RE.sub(" ", html.unescape(TAG_RE.sub(" ", s))).strip().lower()


def serialize(obj):
    """MongoDB-র বিশেষ টাইপগুলোকে JSON-এ পাঠানোর উপযোগী করে।"""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal128):
        return float(obj.to_decimal())
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [serialize(v) for v in obj]
    return obj


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def factorize(values: List[str]):
    """স্ট্রিংয়ের তালিকাকে সংখ্যার কোডে রূপান্তর করে (দ্রুত তুলনার জন্য)।"""
    vocab: dict = {}
    codes = np.empty(len(values), dtype=np.int32)
    for i, v in enumerate(values):
        codes[i] = vocab.setdefault(v, len(vocab))
    return codes, list(vocab.keys())


# ===============================================================
# ৫. ইন-মেমরি ইনডেক্স (প্যাটার্ন চেনা এখানেই হয়)
# ===============================================================
@dataclass
class Index:
    n: int
    ids: List[str]
    raw_ids: List[Any]
    id_to_row: dict
    type_code: np.ndarray
    type_vocab: List[str]
    loc_code: np.ndarray
    loc_vocab: List[str]
    cluster: np.ndarray
    popularity: np.ndarray
    vecs: np.ndarray                 # (n, d) নরমালাইজড SVD "প্যাটার্ন" ভেক্টর
    tfidf_csc: Any                   # কীওয়ার্ড মেলানোর জন্য স্পার্স টার্ম ম্যাট্রিক্স
    vocab: dict                      # শব্দ -> tfidf_csc-এর কলাম
    vectorizer: Any
    svd: Any
    n_clusters: int
    imps: np.ndarray = field(default=None)
    clicks: np.ndarray = field(default=None)
    built_at: float = 0.0


def build_index(docs: List[dict]) -> Optional[Index]:
    """সব রুম থেকে প্যাটার্ন শিখে নতুন ইনডেক্স তৈরি করে।"""
    n = len(docs)
    if n == 0:
        return None
    t0 = time.time()

    ids, raw_ids, id_to_row = [], [], {}
    types, locs, texts = [], [], []
    popularity = np.zeros(n, dtype=np.float32)

    for i, d in enumerate(docs):
        sid = str(d["_id"])
        ids.append(sid)
        raw_ids.append(d["_id"])
        id_to_row[sid] = i
        if d.get("id") is not None:
            id_to_row[str(d["id"])] = i

        t = safe_str(d.get("types")).lower().strip()
        area = safe_str(d.get("area")).lower().strip()
        nb = safe_str(d.get("neighborhood")).lower().strip()
        addr = safe_str(d.get("address")).lower().strip()
        types.append(t)
        locs.append(f"{area} {nb}".strip())
        texts.append(f"{t} {t} {area} {nb} {nb} {addr} {strip_html(d.get('description'))}")
        try:
            popularity[i] = min(float(d.get("viewCount", 0) or 0) / 20.0, 1.0)
        except (ValueError, TypeError):
            popularity[i] = 0.0

    type_code, type_vocab = factorize(types)
    loc_code, loc_vocab = factorize(locs)

    # --- TF-IDF -> SVD: লেখা থেকে লুকানো প্যাটার্ন শেখা ---
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=MAX_TFIDF_FEATURES,
        min_df=2 if n >= 1000 else 1, sublinear_tf=True, dtype=np.float32,
    )
    tfidf = vectorizer.fit_transform(texts)
    if hasattr(vectorizer, "stop_words_"):
        vectorizer.stop_words_ = set()          # মেমরি বাঁচায়, পরে লাগে না

    n_terms = tfidf.shape[1]
    dims = min(SVD_DIMS, n_terms - 1, n - 1)
    svd = None
    if dims >= 2:
        svd = TruncatedSVD(n_components=dims, algorithm="randomized",
                           n_iter=4, random_state=42)
        vecs = normalize(svd.fit_transform(tfidf)).astype(np.float32)
    else:
        vecs = np.zeros((n, 1), dtype=np.float32)

    # --- KMeans: রুমগুলোকে প্যাটার্ন ক্লাস্টারে ভাগ করা ---
    k = min(N_CLUSTERS_MAX, max(1, n // 50))
    if svd is not None and k >= 2:
        km = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=2048, random_state=42)
        cluster = km.fit_predict(vecs).astype(np.int16)
    else:
        k, cluster = 1, np.zeros(n, dtype=np.int16)

    idx = Index(
        n=n, ids=ids, raw_ids=raw_ids, id_to_row=id_to_row,
        type_code=type_code, type_vocab=type_vocab,
        loc_code=loc_code, loc_vocab=loc_vocab,
        cluster=cluster, popularity=popularity, vecs=vecs,
        tfidf_csc=tfidf.tocsc(), vocab=vectorizer.vocabulary_,
        vectorizer=vectorizer, svd=svd, n_clusters=k,
        imps=np.zeros(n, dtype=np.int32), clicks=np.zeros(n, dtype=np.int32),
        built_at=time.time(),
    )
    logger.info(f"ইনডেক্স তৈরি: {n}টি রুম, {n_terms}টি শব্দ, {dims} ডাইমেনশন, "
                f"{k}টি ক্লাস্টার, সময় {time.time() - t0:.1f} সেকেন্ড")
    return idx


STATE = {"index": None}
BUILD_LOCK = threading.Lock()
STOP = threading.Event()


def load_rooms() -> List[dict]:
    """ইনডেক্সের জন্য শুধু প্রয়োজনীয় ফিল্ডগুলো আনে (মোছা রুম বাদ)।"""
    projection = {"types": 1, "area": 1, "neighborhood": 1, "address": 1,
                  "description": 1, "viewCount": 1, "id": 1}
    return list(rooms_collection.find({"isDeleted": {"$ne": True}}, projection).batch_size(5000))


def sync_stats(idx: Index):
    """ডাটাবেস থেকে প্রতি রুমের ইমপ্রেশন ও ক্লিকের সংখ্যা RAM-এ আনে।"""
    imps = np.zeros(idx.n, dtype=np.int32)
    clicks = np.zeros(idx.n, dtype=np.int32)
    for s in stats_col.find({}, {"room_id": 1, "impressions": 1, "clicks": 1}):
        row = idx.id_to_row.get(s.get("room_id"))
        if row is not None:
            imps[row] = s.get("impressions", 0)
            clicks[row] = s.get("clicks", 0)
    idx.imps, idx.clicks = imps, clicks


def rebuild_index() -> bool:
    if not BUILD_LOCK.acquire(blocking=False):
        return False                              # আগের বিল্ড এখনও চলছে
    try:
        idx = build_index(load_rooms())
        if idx is None:
            logger.warning("কোনো রুম পাওয়া যায়নি; ইনডেক্স তৈরি হয়নি।")
            return False
        sync_stats(idx)
        STATE["index"] = idx                      # এক ঝটকায় বদল, রিকুয়েস্ট কখনও আটকায় না
        try:
            joblib.dump(idx, INDEX_CACHE, compress=3)
        except Exception as e:
            logger.warning(f"ইনডেক্স ক্যাশ সেভ করা যায়নি: {e}")
        return True
    except Exception:
        logger.error("ইনডেক্স তৈরিতে ব্যর্থ", exc_info=True)
        return False
    finally:
        BUILD_LOCK.release()


def init_index():
    """দ্রুত রিস্টার্ট: ডিস্ক ক্যাশ নতুন থাকলে সেটাই লোড করে, নইলে নতুন বানায়।"""
    try:
        if os.path.exists(INDEX_CACHE) and time.time() - os.path.getmtime(INDEX_CACHE) < INDEX_CACHE_MAX_AGE:
            idx = joblib.load(INDEX_CACHE)
            sync_stats(idx)
            STATE["index"] = idx
            logger.info(f"ক্যাশ থেকে ইনডেক্স লোড হয়েছে ({idx.n}টি রুম)।")
            return
    except Exception as e:
        logger.warning(f"ইনডেক্স ক্যাশ ব্যবহারযোগ্য নয়, নতুন করে বানানো হচ্ছে: {e}")
    rebuild_index()


def maintenance_loop():
    """ব্যাকগ্রাউন্ডে নিয়মিত স্ট্যাটস সিঙ্ক ও ইনডেক্স রিবিল্ড করে।"""
    last_stats = last_build = time.time()
    while not STOP.wait(30):
        try:
            now = time.time()
            idx = STATE["index"]
            if idx is not None and now - last_stats > STATS_SYNC_SEC:
                sync_stats(idx)
                last_stats = now
            if now - last_build > REINDEX_EVERY_SEC:
                rebuild_index()
                last_build = now
        except Exception:
            logger.error("মেইনটেন্যান্স লুপে ত্রুটি", exc_info=True)


# ===============================================================
# ৬. শেখা ওজন (RAM-এ ক্যাশ করা)
# ===============================================================
_W = {"w": None, "t": 0.0}


def get_weights() -> dict:
    if _W["w"] is None or time.time() - _W["t"] > WEIGHTS_TTL:
        try:
            doc = model_col.find_one({"_id": "global"}) or {}
            w = {**DEFAULT_WEIGHTS, **(doc.get("weights") or {})}
            _W["w"] = {k: max(-W_CLIP, min(W_CLIP, float(v))) for k, v in w.items()}
            _W["t"] = time.time()
        except Exception as e:
            logger.warning(f"ওজন লোড করা যায়নি, ক্যাশ/ডিফল্ট ব্যবহার হচ্ছে: {e}")
            return _W["w"] or dict(DEFAULT_WEIGHTS)
    return _W["w"]


# ===============================================================
# ৭. দ্রুত ভেক্টর স্কোরিং (সব রুমের উপর একসাথে চলে)
# ===============================================================
def vocab_mask(codes: np.ndarray, vocab: List[str], query: Optional[str]) -> np.ndarray:
    """সার্চ টেক্সট যে রুমের লোকেশন/টাইপে আছে, সেগুলোর জন্য 1, বাকিদের 0।"""
    q = (query or "").strip().lower()
    if not q:
        return np.zeros(len(codes), dtype=np.float32)
    hit = [i for i, s in enumerate(vocab) if q in s]
    if not hit:
        return np.zeros(len(codes), dtype=np.float32)
    return np.isin(codes, hit).astype(np.float32)


def keyword_feature(idx: Index, keywords: List[str]) -> np.ndarray:
    """কীওয়ার্ডের শব্দ যে রুমের লেখায় আছে, তার অনুপাত বের করে।"""
    out = np.zeros(idx.n, dtype=np.float32)
    kws = [k.strip().lower() for k in (keywords or []) if k and k.strip()]
    if not kws:
        return out
    csc = idx.tfidf_csc
    for kw in kws:
        present = None
        for tok in WORD_RE.findall(kw):
            j = idx.vocab.get(tok)
            if j is None:
                present = None
                break
            m = np.zeros(idx.n, dtype=bool)
            m[csc.indices[csc.indptr[j]:csc.indptr[j + 1]]] = True
            present = m if present is None else (present & m)
        if present is not None:
            out += present
    return out / len(kws)


def user_vector(idx: Index, req: RecommendationRequest, viewed_rows: List[int]):
    """ব্যবহারকারীর দেখা রুম ও সার্চ থেকে তার পছন্দের একটি ভেক্টর বানায়।"""
    if idx.svd is None:
        return None
    parts = []
    if viewed_rows:
        parts.append(idx.vecs[viewed_rows].mean(axis=0))
    q = f"{req.search_type} {req.search_location} {req.user_detected_city} {' '.join(req.keywords or [])}".strip()
    if q:
        qv = normalize(idx.svd.transform(idx.vectorizer.transform([q])))[0]
        if np.any(qv):
            parts.append(qv.astype(np.float32))
    if not parts:
        return None
    u = np.sum(parts, axis=0)
    norm = np.linalg.norm(u)
    return (u / norm).astype(np.float32) if norm > 0 else None


def score_all(idx: Index, req: RecommendationRequest, weights: dict, viewed_rows: List[int]):
    """সব রুমের জন্য একসাথে স্কোর (z) ও ফিচার হিসাব করে।"""
    zeros = np.zeros(idx.n, dtype=np.float32)

    u = user_vector(idx, req, viewed_rows)
    text = np.clip(idx.vecs @ u, 0.0, 1.0) if u is not None else zeros

    if viewed_rows:
        session_type = np.isin(idx.type_code, np.unique(idx.type_code[viewed_rows])).astype(np.float32)
        session_loc = np.isin(idx.loc_code, np.unique(idx.loc_code[viewed_rows])).astype(np.float32)
        cluster = np.isin(idx.cluster, np.unique(idx.cluster[viewed_rows])).astype(np.float32)
    else:
        session_type = session_loc = cluster = zeros

    ctr = np.minimum(((idx.clicks + 1.0) / (idx.imps + 20.0)) * 5.0, 1.0).astype(np.float32)

    feats = {
        "text": text,
        "location": vocab_mask(idx.loc_code, idx.loc_vocab, req.search_location),
        "city": vocab_mask(idx.loc_code, idx.loc_vocab, req.user_detected_city),
        "type": vocab_mask(idx.type_code, idx.type_vocab, req.search_type),
        "keyword": keyword_feature(idx, req.keywords or []),
        "session_type": session_type,
        "session_loc": session_loc,
        "cluster": cluster,
        "popularity": idx.popularity,
        "ctr": ctr,
    }

    z = np.full(idx.n, weights["bias"], dtype=np.float32)
    for k, arr in feats.items():
        z += np.float32(weights.get(k, 0.0)) * arr
    return feats, z


# ===============================================================
# ৮. মালিকের (ইউজার) তথ্য খোঁজা + পরিষ্কার আউটপুট
# ===============================================================
USER_NAME_KEYS = ["name", "fullName", "full_name", "username", "displayName"]
USER_PHONE_KEYS = ["phone", "phoneNumber", "phone_number", "mobile", "mobileNumber", "contactNumber"]
HIDDEN_FIELDS = {"_id", "id", "userId", "isDeleted", "updatedAt"}   # এগুলো ক্লায়েন্টকে পাঠানো হয় না
USER_CACHE_TTL = 300                                                # সেকেন্ড
_USER_CACHE: dict = {}


def _first(doc: dict, keys: List[str]):
    for k in keys:
        v = doc.get(k)
        if v not in (None, ""):
            return str(v)
    return None


def fetch_users(user_ids) -> dict:
    """userId -> {"name":..., "phone":...}। যাদের ক্যাশে নেই তাদের জন্য একটাই কুয়েরি, ৫ মিনিট ক্যাশ।
    users কালেকশন থেকে শুধু নাম ও ফোন পড়া হয় (আর কিছু প্রকাশ হয় না)।"""
    now = time.time()
    wanted = {str(u) for u in user_ids if u is not None}
    result, missing = {}, []
    for uid in wanted:
        hit = _USER_CACHE.get(uid)
        if hit and now - hit[0] < USER_CACHE_TTL:
            result[uid] = hit[1]
        else:
            missing.append(uid)

    if missing:
        variants: list = []
        for uid in missing:
            variants.append(uid)
            if ObjectId.is_valid(uid):
                variants.append(ObjectId(uid))
        projection = {k: 1 for k in USER_NAME_KEYS + USER_PHONE_KEYS}
        try:
            found = {}
            for u in users_col.find({"_id": {"$in": variants}}, projection):
                found[str(u["_id"])] = {
                    "name": _first(u, USER_NAME_KEYS),
                    "phone": _first(u, USER_PHONE_KEYS),
                }
            if len(_USER_CACHE) > 50000:
                _USER_CACHE.clear()
            for uid in missing:
                info = found.get(uid)             # ইউজার আর না থাকলে None
                _USER_CACHE[uid] = (now, info)
                result[uid] = info
        except Exception:
            logger.error("ইউজার তথ্য খুঁজতে ব্যর্থ", exc_info=True)
            for uid in missing:
                result[uid] = None
    return result


def public_room(d: dict, p: float, users: dict) -> dict:
    """একটি রুমের চূড়ান্ত JSON: পরিষ্কার ফিল্ড + match_score + মালিকের নাম/ফোন।"""
    raw = serialize(d)
    item = {"id": str(d["_id"])}
    for k, v in raw.items():
        if k not in HIDDEN_FIELDS:
            item[k] = v
    item["match_score"] = f"{int(round(p * 100))}% Match"
    item["user"] = users.get(str(d.get("userId")))
    return item


# ===============================================================
# ৯. অ্যাপ
# ===============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_db()
    init_index()
    threading.Thread(target=maintenance_loop, daemon=True).start()
    yield
    STOP.set()


app = FastAPI(title="ফাইন্ড মাই রুম - দ্রুত সেল্ফ-লার্নিং রেকমেন্ডেশন ইঞ্জিন", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def log_impressions(request_id: str, shown: list):
    """রেসপন্স পাঠানোর পরে চলে, তাই ইউজারের কোনো অপেক্ষা বাড়ায় না।"""
    try:
        now = datetime.now(timezone.utc)
        impressions_col.insert_many([
            {"request_id": request_id, "room_id": rid, "features": feats,
             "predicted": p, "created_at": now, "done": []}
            for rid, feats, p in shown
        ])
        stats_col.bulk_write([
            UpdateOne({"room_id": rid}, {"$inc": {"impressions": 1}}, upsert=True)
            for rid, _, _ in shown
        ])
        model_col.update_one({"_id": "global"}, {"$inc": {"impressions": len(shown)}})
        idx = STATE["index"]
        if idx is not None:
            for rid, _, _ in shown:
                row = idx.id_to_row.get(rid)
                if row is not None:
                    idx.imps[row] += 1
    except Exception:
        logger.error("ইমপ্রেশন লগ করতে ব্যর্থ", exc_info=True)


@app.get("/")
def home():
    return {"status": "online", "service": "Find My Room - Fast Self-Learning Engine v5"}


@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ডাটাবেস পাওয়া যাচ্ছে না: {e}")
    idx = STATE["index"]
    return {"status": "ok", "database": "connected", "index_ready": idx is not None,
            "rooms_indexed": idx.n if idx else 0}


@app.post("/api/v1/recommendations")
def get_recommendations(
    req: RecommendationRequest,
    background: BackgroundTasks,
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=50),
):
    t0 = time.perf_counter()
    idx = STATE["index"]
    empty = {"success": True, "request_id": None, "page": page, "limit": limit,
             "total_items": 0, "total_pages": 1, "has_next": False,
             "has_prev": page > 1, "data": []}
    if idx is None:
        return empty

    try:
        viewed_rows = sorted({idx.id_to_row[v] for v in (req.viewed_room_ids or [])
                              if v in idx.id_to_row})
        weights = get_weights()
        feats, z = score_all(idx, req, weights, viewed_rows)

        # এক্সপ্লোরেশন বোনাস: কম ইমপ্রেশনের রুম একটু বাড়তি সুযোগ পায় (নতুন রুমের প্রতি ন্যায্যতা)
        rank_score = z + np.float32(EXPLORE_C) / np.sqrt(idx.imps + 1.0, dtype=np.float32)
        if viewed_rows:
            rank_score[viewed_rows] = -np.inf     # আগে দেখা রুম আর দেখানো হবে না

        available = idx.n - len(viewed_rows)
        m = min(MAX_RESULTS, available)
        if m <= 0:
            return empty

        top = np.argpartition(-rank_score, m - 1)[:m]
        top = top[np.argsort(-rank_score[top])]

        total_items = int(m)
        total_pages = max(1, (total_items + limit - 1) // limit)
        start = (page - 1) * limit
        rows = top[start:start + limit]

        # শুধু যে রুমগুলো দেখানো হবে সেগুলোই MongoDB থেকে আনা হয়
        docs = {}
        if len(rows):
            docs = {str(d["_id"]): d for d in
                    rooms_collection.find({"_id": {"$in": [idx.raw_ids[r] for r in rows]}})}
        users = fetch_users([d.get("userId") for d in docs.values()])

        items, shown = [], []
        for r in rows:
            d = docs.get(idx.ids[r])
            if d is None or d.get("isDeleted") is True:
                continue                          # শেষ ইনডেক্সের পরে মোছা রুম বাদ
            p = float(sigmoid(z[r]))
            items.append(public_room(d, p, users))
            f = {k: float(arr[r]) for k, arr in feats.items()}
            f["bias"] = 1.0
            shown.append((idx.ids[r], f, p))

        request_id = uuid.uuid4().hex
        if shown:
            background.add_task(log_impressions, request_id, shown)

        return {
            "success": True,
            "request_id": request_id,
            "page": page,
            "limit": limit,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "rooms_scanned": idx.n,
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
            "data": items,
        }
    except Exception as e:
        logger.error(f"রেকমেন্ডেশনে ত্রুটি: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/feedback")
def feedback(fb: FeedbackRequest):
    """ভুল অনুমান -> বড় শাস্তি। সঠিক অনুমান -> ছোট পুরস্কার।"""
    event = fb.event.lower()
    if event not in EVENT_TARGET:
        raise HTTPException(status_code=400, detail=f"event অবশ্যই এগুলোর একটি হতে হবে: {list(EVENT_TARGET)}")
    try:
        imp = impressions_col.find_one_and_update(
            {"request_id": fb.request_id, "room_id": fb.room_id, "done": {"$ne": event}},
            {"$addToSet": {"done": event}},
        )
        if not imp:
            return {"success": True, "learned": False,
                    "reason": "ইমপ্রেশন পাওয়া যায়নি অথবা এই ইভেন্ট আগেই গণনা হয়েছে"}

        feats = imp["features"]
        weights = get_weights()
        z = sum(weights.get(k, 0.0) * feats.get(k, 0.0) for k in FEATURES)
        p = float(sigmoid(z))

        target = EVENT_TARGET[event]
        error = target - p
        step = LEARNING_RATE * EVENT_STRENGTH[event] * error

        inc = {f"weights.{k}": step * feats.get(k, 0.0) for k in FEATURES}
        inc["updates"] = 1
        if target == 1.0:
            inc["positives"] = 1
        model_col.update_one({"_id": "global"}, {"$inc": inc})
        _W["t"] = 0.0                              # পরের রিকুয়েস্টে নতুন ওজন লোড হবে

        if target == 1.0:
            stats_col.update_one({"room_id": fb.room_id}, {"$inc": {"clicks": 1}}, upsert=True)
            idx = STATE["index"]
            if idx is not None:
                row = idx.id_to_row.get(fb.room_id)
                if row is not None:
                    idx.clicks[row] += 1

        return {
            "success": True, "learned": True, "event": event,
            "score_before": int(round(p * 100)), "error": round(error, 4),
            "verdict": "অনুমান খুব কম ছিল (পুরস্কার)" if error > 0
                       else "অনুমান খুব বেশি ছিল (শাস্তি)",
        }
    except Exception as e:
        logger.error(f"ফিডব্যাকে ত্রুটি: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model")
def model_status():
    """মডেল এখন কী শিখেছে তার অবস্থা দেখায়।"""
    doc = model_col.find_one({"_id": "global"}) or {}
    imps, pos = doc.get("impressions", 0), doc.get("positives", 0)
    idx = STATE["index"]
    return {
        "weights": {k: round(v, 3) for k, v in get_weights().items()},
        "updates": doc.get("updates", 0),
        "impressions": imps,
        "positive_events": pos,
        "overall_ctr_percent": round(pos / imps * 100, 2) if imps else 0.0,
        "index": None if idx is None else {
            "rooms": idx.n, "pattern_clusters": idx.n_clusters,
            "vector_dims": int(idx.vecs.shape[1]),
            "age_minutes": round((time.time() - idx.built_at) / 60, 1),
        },
    }


@app.post("/api/v1/admin/reindex")
def admin_reindex(x_admin_key: str = Header(default="")):
    """সর্বশেষ রুম থেকে নতুন করে প্যাটার্ন শেখে (প্রতি ৩০ মিনিটে নিজে নিজেও চলে)।"""
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="অ্যাডমিন কী ভুল")
    if BUILD_LOCK.locked():
        return {"started": False, "reason": "রিবিল্ড আগে থেকেই চলছে"}
    threading.Thread(target=rebuild_index, daemon=True).start()
    return {"started": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)