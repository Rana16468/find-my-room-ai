
# Find My Room AI: Recommendation Microservice

An AI-powered, self-learning room recommendation service for the Find My Room house listing platform. Built with **Python**, **FastAPI** and **MongoDB**. The whole project lives in a single file, `main.py`, so running that file starts the entire service.

## Links

- **Live Service:** https://find-my-room-ai.onrender.com
- **GitHub Repository:** https://github.com/Rana16468/find-my-room-ai
- **API Docs (Swagger UI):** https://find-my-room-ai.onrender.com/docs

> The service runs on Render's free tier, so the first request may take a few seconds to wake up.

## Project Structure

```
find-my-room-ai/
├── main.py            # The entire service (run this file)
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (create this yourself, do not commit)
└── .gitignore
```

## Prerequisites

- [Python 3.9 or higher](https://www.python.org/downloads/)
- pip (comes with Python)
- A MongoDB database (local or [MongoDB Atlas](https://www.mongodb.com/atlas))

Check your Python version:

```bash
python --version
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Rana16468/find-my-room-ai
cd find-my-room-ai
```

### 2. Create a virtual environment (recommended)

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you prefer to install the packages manually:

```bash
pip install fastapi uvicorn pymongo numpy scipy scikit-learn joblib python-dotenv
```

### 4. Set up environment variables

Create a `.env` file next to `main.py` and add the following:

```env
MONGO_URI=your_mongodb_connection_string
ADMIN_KEY=your_admin_secret_key
CORS_ORIGINS=*
```

| Variable | Required | Description |
| --- | --- | --- |
| `MONGO_URI` | Yes | MongoDB connection string, for example `mongodb+srv://<user>:<password>@<cluster>/<database>` |
| `ADMIN_KEY` | Yes | Secret key for the admin reindex endpoint (sent in the `X-Admin-Key` header) |
| `CORS_ORIGINS` | No | Comma-separated list of allowed origins. Default is `*` (all). In production, use your frontend URL |

> **Note:** Never commit your real `.env` file. Keep it in `.gitignore`.

### 5. Run the project

```bash
python main.py
```

The server starts at **http://localhost:8000**.

You can also start it directly with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** to test the API in your browser.

## API Endpoints

| Method | Endpoint | Purpose | Auth |
| --- | --- | --- | --- |
| `GET` | `/` | Service status | None |
| `GET` | `/health` | Database ping and index readiness | None |
| `POST` | `/api/v1/recommendations` | Get ranked, personalized room recommendations | None |
| `POST` | `/api/v1/feedback` | Send a user action (`click`, `save`, `contact`, `skip`) so the model can learn | None |
| `GET` | `/api/v1/model` | Inspect learned weights and learning statistics | None |
| `POST` | `/api/v1/admin/reindex` | Force an index rebuild | `X-Admin-Key` header |

### Example: get recommendations

```bash
curl -X POST "http://localhost:8000/api/v1/recommendations?page=1&limit=12" \
  -H "Content-Type: application/json" \
  -d '{
    "search_location": "mirpur",
    "search_type": "family",
    "keywords": ["balcony", "gas"],
    "user_detected_city": "dhaka"
  }'
```

Keep the `request_id` from the response. It is required when sending feedback.

### Example: send feedback

```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "your_request_id",
    "room_id": "your_room_id",
    "event": "click"
  }'
```

### Example: force reindex (admin)

```bash
curl -X POST "http://localhost:8000/api/v1/admin/reindex" \
  -H "X-Admin-Key: your_admin_secret_key"
```

## How It Works (Short Version)

1. **Pattern recognition:** Room descriptions are converted into vectors (TF-IDF, then SVD to 64 dimensions) and grouped into pattern clusters (K-Means).
2. **Scoring:** Every room is scored using 11 features (text similarity, location, type, keywords, popularity, click-through rate, and more) and shown as a match score such as `87% Match`.
3. **Learning:** User actions sent to `/api/v1/feedback` update the model instantly with online logistic regression. No manual retraining is needed.
4. **Maintenance:** The index rebuilds automatically every 30 minutes to include new rooms.

## Tech Stack

- Python 3, FastAPI, Uvicorn
- MongoDB (PyMongo)
- NumPy, SciPy, scikit-learn
- joblib, python-dotenv

## Troubleshooting

| Problem | Solution |
| --- | --- |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` again |
| MongoDB connection error | Check `MONGO_URI`, and make sure your IP is allowed in MongoDB Atlas Network Access |
| `401` on `/admin/reindex` | The `X-Admin-Key` header does not match `ADMIN_KEY` in `.env` |
| CORS error from the frontend | Set `CORS_ORIGINS` to your frontend URL |
| `python` not found | Use `python3` instead |

## Related Projects

| Project | Link |
| --- | --- |
| Frontend (Next.js) | https://github.com/Rana16468/myroom |
| Backend API (Node.js, Express & MongoDB) | https://github.com/Rana16468/house-listing-backend |

# Find My Room: Self-Learning Room Recommendation Engine

**Language / ভাষা:** [English](#english) | [বাংলা](#বাংলা)

---

<a id="english"></a>
# English

## 1. What is this project?

Find My Room is an **AI-powered room recommendation backend** built with **FastAPI** and **MongoDB**. Given a user's search (location, room type, keywords) and the rooms they have already viewed, it ranks every available room and returns the best matches, each with a **match score** such as `87% Match`.

It is **self-learning**: every click, save, contact, or skip from users is sent back to the engine, and the ranking model updates itself in real time. No manual retraining is required, and the more people use it, the better the results get.

### Key highlights

- Understands room descriptions (text) and groups similar rooms into pattern clusters
- Ranks all rooms in a few milliseconds using vectorized NumPy operations
- Learns online from user feedback with a logistic-regression model
- Balances showing proven rooms against exploring new ones (cold-start handling)
- Rebuilds itself automatically as new rooms are added
- Fast restarts through a disk cache
- Paginated results, up to 240 ranked rooms per request (20 pages of 12)

---

## 2. How it works (architecture)

```
                 +------------------------------------------+
                 |                MongoDB                   |
                 | list_rooms | users | rec_model           |
                 | rec_impressions | rec_room_stats         |
                 +--------+-----------------------^---------+
                          |                       |
            load rooms    |                       | log impressions,
            (every 30 min)|                       | save learned weights
                          v                       |
   +----------------------------------+           |
   |   In-memory Index (RAM)          |           |
   |  - TF-IDF text vectors           |           |
   |  - SVD 64-dim embeddings         |           |
   |  - K-Means pattern clusters      |           |
   |  - type / location codes         |           |
   |  - impressions + clicks per room |           |
   +----------------+-----------------+           |
                    |                             |
                    v                             |
   +----------------------------------+           |
   |  Scoring engine (NumPy)          |-----------+
   |  11 features -> weighted sum     |
   |  -> sigmoid -> probability       |
   +----------------+-----------------+
                    |
                    v
        POST /api/v1/recommendations  ---->  ranked rooms + request_id
                                                     |
        POST /api/v1/feedback  <---- click/save/contact/skip from the app
              (updates model weights instantly)
```

### Step 1: Pattern recognition (building the index)

On startup (and again every 30 minutes), the engine loads all non-deleted rooms and learns the patterns in them:

1. **Text understanding (TF-IDF).** For each room, the type, area, neighborhood, address, and description (HTML stripped) are combined into one text. TF-IDF turns it into a weighted word vector (up to 20,000 terms, sublinear term frequency).
2. **Dimensionality reduction (Truncated SVD).** The large sparse vectors are compressed to **64 dense dimensions**. Rooms with similar meaning end up close together even if they use different words (this is also known as Latent Semantic Analysis).
3. **Pattern clusters (MiniBatch K-Means).** Rooms are grouped into up to **40 clusters** (about one cluster per 50 rooms). Each cluster is a "pattern" of similar rooms.
4. **Categorical codes.** Room types and locations are converted to integer codes for very fast comparisons.
5. **Popularity.** View count is normalized to a 0 to 1 score.

All of this lives in RAM and is also saved to a disk cache (`index_cache.joblib`), so a restart within 6 hours loads in seconds instead of rebuilding.

### Step 2: Understanding the user

For each request, the engine builds a **user taste vector** from:

- the average embedding of the rooms the user already viewed (`viewed_room_ids`)
- the embedding of their search query (type, location, detected city, keywords)

### Step 3: Scoring every room

For every room, 11 features are computed at once:

| Feature | Meaning |
|---|---|
| `bias` | Base offset of the model |
| `text` | Cosine similarity between the user vector and the room vector |
| `location` | Room location matches the searched location |
| `city` | Room location matches the user's detected city |
| `type` | Room type matches the searched type |
| `keyword` | Fraction of requested keywords found in the room text |
| `session_type` | Same type as rooms the user viewed |
| `session_loc` | Same location as rooms the user viewed |
| `cluster` | Same pattern cluster as rooms the user viewed |
| `popularity` | Normalized view count |
| `ctr` | Smoothed click-through rate of the room, `(clicks+1)/(impressions+20)` |

The score is a weighted sum of these features passed through a sigmoid:

```
z = bias + sum(weight_i * feature_i)
match probability p = sigmoid(z)        ->  shown as "NN% Match"
```

### Step 4: Ranking with exploration

Rooms are ranked by `z + 0.6 / sqrt(impressions + 1)`. The second term is an **exploration bonus**: rooms that have rarely been shown get a small boost so the system can discover whether they are good. This is the same idea behind UCB-style bandit algorithms and it solves the cold-start problem. Rooms the user already viewed are excluded. The top 240 are paginated.

### Step 5: Online learning (how it improves)

Every recommendation response includes a `request_id`. The impressions (which rooms were shown and their feature values) are saved in the background. When the app reports what the user did, the model learns immediately:

| Event | Target | Strength |
|---|---|---|
| `click` | 1 (positive) | 1.0 |
| `save` | 1 (positive) | 1.5 |
| `contact` | 1 (positive) | 2.0 |
| `skip` | 0 (negative) | 0.5 |

```
error  = target - predicted_probability
step   = learning_rate(0.15) * event_strength * error
weight_i += step * feature_i        (clipped to [-6, 6])
```

- If the model **under-predicted** a room the user liked, the weights of the features that were active increase (reward).
- If the model **over-predicted** a room the user skipped, those weights decrease (penalty).

This is **online stochastic gradient descent on logistic regression**. Each (request, room, event) is counted only once, so duplicate events cannot distort learning. Weights are shared globally, stored in MongoDB, and cached in memory for 5 seconds. Room-level impressions and clicks also feed the `ctr` feature and the exploration bonus.

### Step 6: Background maintenance

| Task | Interval |
|---|---|
| Sync impressions/clicks from DB into RAM | every 5 minutes |
| Rebuild the pattern index with new rooms | every 30 minutes |
| Impression records auto-expire (TTL) | after 7 days |
| Disk cache considered fresh | up to 6 hours |

---

## 3. API reference

Base URL (local): `http://localhost:8000`. Interactive docs (Swagger UI): `/docs`.

| Method | Endpoint | Purpose | Auth |
|---|---|---|---|
| `GET` | `/` | Service status | None |
| `GET` | `/health` | Database ping and index readiness | None |
| `POST` | `/api/v1/recommendations` | Get ranked, personalized room recommendations | None |
| `POST` | `/api/v1/feedback` | Send a user action so the model can learn | None |
| `GET` | `/api/v1/model` | Inspect learned weights and learning statistics | None |
| `POST` | `/api/v1/admin/reindex` | Force an index rebuild | `X-Admin-Key` header |

### `POST /api/v1/recommendations`

Query parameters: `page` (default 1) and `limit` (default 12, max 50).

Request body (all fields optional):

```json
{
  "viewed_room_ids": ["65f1c0...", "65f1c1..."],
  "search_location": "mirpur",
  "search_type": "family",
  "keywords": ["balcony", "gas"],
  "user_detected_city": "dhaka"
}
```

Response (shortened):

```json
{
  "success": true,
  "request_id": "9f2c1a...",
  "page": 1,
  "limit": 12,
  "total_items": 240,
  "total_pages": 20,
  "has_next": true,
  "has_prev": false,
  "rooms_scanned": 100000,
  "took_ms": 38.2,
  "data": [
    {
      "id": "65f1c2...",
      "types": "family",
      "area": "mirpur",
      "match_score": "87% Match",
      "user": { "name": "Owner Name", "phone": "01XXXXXXXXX" }
    }
  ]
}
```

Keep the `request_id`. It is required when sending feedback.

### `POST /api/v1/feedback`

```json
{
  "request_id": "9f2c1a...",
  "room_id": "65f1c2...",
  "event": "click"
}
```

`event` must be one of `click`, `save`, `contact`, `skip`. The response tells whether the model learned, the predicted score before learning, and the error.

### `GET /api/v1/model`

Returns the current weights, number of updates, total impressions, positive events, overall CTR percentage, and index info (rooms, clusters, vector dimensions, age).

### `POST /api/v1/admin/reindex`

Header: `X-Admin-Key: <ADMIN_KEY>`. Starts a background rebuild. Returns `401` for a wrong key.

---

## 4. Tech stack

- **Python 3**, **FastAPI**, **Uvicorn**
- **MongoDB** (PyMongo)
- **NumPy**, **SciPy**, **scikit-learn** (TF-IDF, TruncatedSVD, MiniBatchKMeans)
- **joblib** (index disk cache), **python-dotenv**

## 5. MongoDB collections

| Collection | Purpose |
|---|---|
| `list_rooms` | Room listings (read only) |
| `users` | Owner name and phone (read only, nothing else is exposed) |
| `rec_model` | Global learned weights and counters |
| `rec_impressions` | Shown rooms and features per request (auto-deleted after 7 days) |
| `rec_room_stats` | Per-room impressions and clicks |

## 6. Setup and run

```bash
pip install fastapi uvicorn pymongo numpy scipy scikit-learn joblib python-dotenv
```

Create a `.env` file next to `main.py`:

```env
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>/<database>
ADMIN_KEY=your-secret-admin-key
CORS_ORIGINS=*
```

| Variable | Required | Description |
|---|---|---|
| `MONGO_URI` | Yes | MongoDB connection string |
| `ADMIN_KEY` | Yes | Secret for the reindex endpoint |
| `CORS_ORIGINS` | No | Comma-separated allowed origins (default `*`) |

Run:

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to try the API.

## 7. Design decisions

- **All scoring is vectorized** in NumPy over every room at once, so ranking 100,000 rooms stays fast.
- **Impression logging runs after the response is sent** (background task), so users never wait for it.
- **Explainable model.** Weights are a simple, inspectable logistic regression, so you can see exactly what the system has learned via `/api/v1/model`.
- **Safe by default.** Deleted rooms are excluded, internal fields are hidden from responses, and only the owner's name and phone are read from `users`.
- **Resilient.** If the database is briefly unavailable, cached weights are used; index rebuilds never run twice at once.

---
---

<a id="বাংলা"></a>
# বাংলা

## ১. এই প্রজেক্ট কী?

Find My Room হলো **FastAPI** ও **MongoDB** দিয়ে বানানো একটি **এআই-চালিত রুম রেকমেন্ডেশন ব্যাকএন্ড**। ইউজারের সার্চ (লোকেশন, রুমের ধরন, কীওয়ার্ড) এবং সে আগে যেসব রুম দেখেছে, তার ভিত্তিতে এটি সব রুম র‍্যাংক করে সেরাগুলো ফেরত দেয়, প্রতিটির সাথে একটি **ম্যাচ স্কোর** থাকে, যেমন `87% Match`।

এটি **সেল্ফ-লার্নিং**: ইউজারের প্রতিটি ক্লিক, সেভ, যোগাযোগ বা স্কিপ ইঞ্জিনে ফেরত পাঠানো হয় এবং মডেল সাথে সাথে নিজেকে আপডেট করে। আলাদা করে ট্রেনিং করতে হয় না, যত বেশি মানুষ ব্যবহার করে ফলাফল তত ভালো হয়।

### প্রধান বৈশিষ্ট্য

- রুমের বর্ণনা (টেক্সট) বুঝে এবং একই ধরনের রুমকে প্যাটার্ন ক্লাস্টারে ভাগ করে
- NumPy-র ভেক্টরাইজড হিসাবে কয়েক মিলিসেকেন্ডে সব রুম র‍্যাংক করে
- লজিস্টিক রিগ্রেশনের মাধ্যমে ইউজারের ফিডব্যাক থেকে অনলাইনে শেখে
- প্রমাণিত ভালো রুম দেখানো এবং নতুন রুম যাচাই করার মধ্যে ভারসাম্য রাখে (কোল্ড-স্টার্ট সমস্যার সমাধান)
- নতুন রুম যোগ হলে নিজে নিজেই ইনডেক্স আবার বানায়
- ডিস্ক ক্যাশের কারণে রিস্টার্ট খুব দ্রুত
- পেজিনেশনসহ রেজাল্ট, প্রতি রিকুয়েস্টে সর্বোচ্চ ২৪০টি রুম (১২টি করে ২০ পেজ)

---

## ২. এটি কীভাবে কাজ করে (আর্কিটেকচার)

```
                 +------------------------------------------+
                 |                MongoDB                   |
                 | list_rooms | users | rec_model           |
                 | rec_impressions | rec_room_stats         |
                 +--------+-----------------------^---------+
                          |                       |
            রুম লোড       |                       | ইমপ্রেশন লগ,
            (প্রতি ৩০ মিনিট)|                      | শেখা ওজন সেভ
                          v                       |
   +----------------------------------+           |
   |   ইন-মেমরি ইনডেক্স (RAM)         |           |
   |  - TF-IDF টেক্সট ভেক্টর          |           |
   |  - SVD ৬৪-ডাইমেনশন এমবেডিং       |           |
   |  - K-Means প্যাটার্ন ক্লাস্টার     |           |
   |  - টাইপ / লোকেশন কোড             |           |
   |  - প্রতি রুমের ইমপ্রেশন ও ক্লিক    |           |
   +----------------+-----------------+           |
                    |                             |
                    v                             |
   +----------------------------------+           |
   |  স্কোরিং ইঞ্জিন (NumPy)          |-----------+
   |  ১১টি ফিচার -> ওজনযুক্ত যোগফল     |
   |  -> সিগময়েড -> সম্ভাবনা           |
   +----------------+-----------------+
                    |
                    v
        POST /api/v1/recommendations  ---->  র‍্যাংক করা রুম + request_id
                                                     |
        POST /api/v1/feedback  <---- অ্যাপ থেকে click/save/contact/skip
              (সাথে সাথে মডেলের ওজন আপডেট হয়)
```

### ধাপ ১: প্যাটার্ন রিকগনিশন (ইনডেক্স তৈরি)

সার্ভার চালু হওয়ার সময় (এবং এরপর প্রতি ৩০ মিনিটে) ইঞ্জিন মোছা নয় এমন সব রুম লোড করে এবং তাদের প্যাটার্ন শেখে:

1. **টেক্সট বোঝা (TF-IDF):** প্রতিটি রুমের টাইপ, এরিয়া, নেইবারহুড, ঠিকানা ও বর্ণনা (HTML ট্যাগ মুছে) এক করে একটি টেক্সট বানানো হয়। TF-IDF সেটিকে ওজনযুক্ত শব্দ-ভেক্টরে রূপান্তর করে (সর্বোচ্চ ২০,০০০ শব্দ)।
2. **ডাইমেনশন কমানো (Truncated SVD):** বিশাল ভেক্টরকে **৬৪ ডাইমেনশনে** সংকুচিত করা হয়। ফলে ভিন্ন শব্দ ব্যবহার করলেও একই অর্থের রুমগুলো কাছাকাছি চলে আসে (একে Latent Semantic Analysis-ও বলা হয়)।
3. **প্যাটার্ন ক্লাস্টার (MiniBatch K-Means):** রুমগুলোকে সর্বোচ্চ **৪০টি ক্লাস্টারে** ভাগ করা হয় (প্রায় প্রতি ৫০ রুমে ১টি)। প্রতিটি ক্লাস্টার একই ধরনের রুমের একটি "প্যাটার্ন"।
4. **ক্যাটাগরি কোড:** রুমের টাইপ ও লোকেশনকে সংখ্যায় রূপান্তর করা হয়, যাতে তুলনা অনেক দ্রুত হয়।
5. **জনপ্রিয়তা:** ভিউ কাউন্টকে ০ থেকে ১ স্কোরে নরমালাইজ করা হয়।

এই সবকিছু RAM-এ থাকে এবং ডিস্ক ক্যাশেও (`index_cache.joblib`) সেভ হয়, তাই ৬ ঘণ্টার মধ্যে রিস্টার্ট করলে নতুন করে বানাতে হয় না, কয়েক সেকেন্ডেই লোড হয়ে যায়।

### ধাপ ২: ইউজারকে বোঝা

প্রতিটি রিকুয়েস্টে ইঞ্জিন একটি **ইউজার টেস্ট ভেক্টর** বানায়:

- ইউজার আগে যেসব রুম দেখেছে (`viewed_room_ids`) তাদের এমবেডিংয়ের গড়
- তার সার্চ কোয়েরির (টাইপ, লোকেশন, শনাক্ত করা শহর, কীওয়ার্ড) এমবেডিং

### ধাপ ৩: প্রতিটি রুমের স্কোর বের করা

প্রতিটি রুমের জন্য একসাথে ১১টি ফিচার হিসাব করা হয়:

| ফিচার | অর্থ |
|---|---|
| `bias` | মডেলের মূল ভিত্তি মান |
| `text` | ইউজার ভেক্টর ও রুম ভেক্টরের কোসাইন সাদৃশ্য |
| `location` | রুমের লোকেশন সার্চ করা লোকেশনের সাথে মিলে |
| `city` | রুমের লোকেশন ইউজারের শনাক্ত করা শহরের সাথে মিলে |
| `type` | রুমের টাইপ সার্চ করা টাইপের সাথে মিলে |
| `keyword` | চাওয়া কীওয়ার্ডের কত অংশ রুমের লেখায় আছে |
| `session_type` | ইউজারের দেখা রুমগুলোর মতো একই টাইপ |
| `session_loc` | ইউজারের দেখা রুমগুলোর মতো একই লোকেশন |
| `cluster` | ইউজারের দেখা রুমগুলোর মতো একই প্যাটার্ন ক্লাস্টার |
| `popularity` | নরমালাইজ করা ভিউ কাউন্ট |
| `ctr` | রুমের স্মুথ করা ক্লিক-থ্রু রেট, `(clicks+1)/(impressions+20)` |

স্কোর হলো এই ফিচারগুলোর ওজনযুক্ত যোগফল, যা সিগময়েডের মধ্য দিয়ে যায়:

```
z = bias + sum(weight_i * feature_i)
ম্যাচ সম্ভাবনা p = sigmoid(z)        ->  "NN% Match" হিসেবে দেখানো হয়
```

### ধাপ ৪: এক্সপ্লোরেশনসহ র‍্যাংকিং

রুমগুলো `z + 0.6 / sqrt(impressions + 1)` দিয়ে র‍্যাংক হয়। দ্বিতীয় অংশটি হলো **এক্সপ্লোরেশন বোনাস**: যেসব রুম খুব কম দেখানো হয়েছে তারা সামান্য বাড়তি সুবিধা পায়, যাতে সিস্টেম জানতে পারে সেগুলো ভালো কি না। এটি UCB-ধরনের ব্যান্ডিট অ্যালগরিদমের মতো ধারণা এবং কোল্ড-স্টার্ট সমস্যা সমাধান করে। ইউজার আগে দেখেছে এমন রুম বাদ যায়। সেরা ২৪০টি পেজ করে দেওয়া হয়।

### ধাপ ৫: অনলাইন লার্নিং (কীভাবে শিখে ভালো হয়)

প্রতিটি রেকমেন্ডেশন রেসপন্সে একটি `request_id` থাকে। কোন কোন রুম দেখানো হয়েছে এবং তাদের ফিচার মান ব্যাকগ্রাউন্ডে সেভ হয়। অ্যাপ যখন ইউজারের কাজ জানায়, মডেল সাথে সাথে শেখে:

| ইভেন্ট | টার্গেট | শক্তি |
|---|---|---|
| `click` | ১ (পজিটিভ) | ১.০ |
| `save` | ১ (পজিটিভ) | ১.৫ |
| `contact` | ১ (পজিটিভ) | ২.০ |
| `skip` | ০ (নেগেটিভ) | ০.৫ |

```
error  = target - predicted_probability
step   = learning_rate(0.15) * event_strength * error
weight_i += step * feature_i        ([-6, 6] সীমার মধ্যে ক্লিপ করা)
```

- ইউজার যে রুম পছন্দ করেছে সেটির অনুমান **কম** হলে, সক্রিয় ফিচারগুলোর ওজন বাড়ে (পুরস্কার)।
- ইউজার যে রুম স্কিপ করেছে সেটির অনুমান **বেশি** হলে, ওজন কমে (শাস্তি)।

এটি লজিস্টিক রিগ্রেশনের উপর **অনলাইন স্টোকাস্টিক গ্রেডিয়েন্ট ডিসেন্ট**। প্রতিটি (request, room, event) শুধু একবারই গণনা হয়, তাই ডুপ্লিকেট ইভেন্ট শেখাকে নষ্ট করতে পারে না। ওজনগুলো গ্লোবাল, MongoDB-তে সংরক্ষিত এবং ৫ সেকেন্ড RAM-এ ক্যাশ থাকে। রুমভিত্তিক ইমপ্রেশন ও ক্লিক `ctr` ফিচার এবং এক্সপ্লোরেশন বোনাসেও কাজে লাগে।

### ধাপ ৬: ব্যাকগ্রাউন্ড মেইনটেন্যান্স

| কাজ | সময়ের ব্যবধান |
|---|---|
| ডাটাবেস থেকে ইমপ্রেশন/ক্লিক RAM-এ সিঙ্ক | প্রতি ৫ মিনিট |
| নতুন রুমসহ প্যাটার্ন ইনডেক্স রিবিল্ড | প্রতি ৩০ মিনিট |
| ইমপ্রেশন রেকর্ড নিজে মুছে যায় (TTL) | ৭ দিন পর |
| ডিস্ক ক্যাশ নতুন ধরা হয় | সর্বোচ্চ ৬ ঘণ্টা |

---

## ৩. API রেফারেন্স

বেস URL (লোকাল): `http://localhost:8000`। ইন্টারেক্টিভ ডক (Swagger UI): `/docs`।

| মেথড | এন্ডপয়েন্ট | কাজ | অথ |
|---|---|---|---|
| `GET` | `/` | সার্ভিসের স্ট্যাটাস | নেই |
| `GET` | `/health` | ডাটাবেস পিং ও ইনডেক্স প্রস্তুত কি না | নেই |
| `POST` | `/api/v1/recommendations` | র‍্যাংক করা পার্সোনালাইজড রুম রেকমেন্ডেশন | নেই |
| `POST` | `/api/v1/feedback` | ইউজারের কাজ পাঠানো, যাতে মডেল শিখতে পারে | নেই |
| `GET` | `/api/v1/model` | শেখা ওজন ও লার্নিং পরিসংখ্যান দেখা | নেই |
| `POST` | `/api/v1/admin/reindex` | জোর করে ইনডেক্স রিবিল্ড | `X-Admin-Key` হেডার |

### `POST /api/v1/recommendations`

কোয়েরি প্যারামিটার: `page` (ডিফল্ট ১) ও `limit` (ডিফল্ট ১২, সর্বোচ্চ ৫০)।

রিকুয়েস্ট বডি (সব ফিল্ড ঐচ্ছিক):

```json
{
  "viewed_room_ids": ["65f1c0...", "65f1c1..."],
  "search_location": "mirpur",
  "search_type": "family",
  "keywords": ["balcony", "gas"],
  "user_detected_city": "dhaka"
}
```

রেসপন্স (সংক্ষিপ্ত):

```json
{
  "success": true,
  "request_id": "9f2c1a...",
  "page": 1,
  "limit": 12,
  "total_items": 240,
  "total_pages": 20,
  "has_next": true,
  "has_prev": false,
  "rooms_scanned": 100000,
  "took_ms": 38.2,
  "data": [
    {
      "id": "65f1c2...",
      "types": "family",
      "area": "mirpur",
      "match_score": "87% Match",
      "user": { "name": "Owner Name", "phone": "01XXXXXXXXX" }
    }
  ]
}
```

`request_id` সংরক্ষণ করুন, ফিডব্যাক পাঠানোর সময় এটি লাগে।

### `POST /api/v1/feedback`

```json
{
  "request_id": "9f2c1a...",
  "room_id": "65f1c2...",
  "event": "click"
}
```

`event` অবশ্যই `click`, `save`, `contact`, `skip`-এর একটি হতে হবে। রেসপন্সে জানানো হয় মডেল শিখেছে কি না, শেখার আগের স্কোর কত ছিল এবং এররের মান কত।

### `GET /api/v1/model`

বর্তমান ওজন, আপডেটের সংখ্যা, মোট ইমপ্রেশন, পজিটিভ ইভেন্ট, সার্বিক CTR শতাংশ এবং ইনডেক্সের তথ্য (রুম, ক্লাস্টার, ভেক্টর ডাইমেনশন, বয়স) ফেরত দেয়।

### `POST /api/v1/admin/reindex`

হেডার: `X-Admin-Key: <ADMIN_KEY>`। ব্যাকগ্রাউন্ডে রিবিল্ড শুরু করে। ভুল কী দিলে `401` ফেরত আসে।

---

## ৪. টেক স্ট্যাক

- **Python 3**, **FastAPI**, **Uvicorn**
- **MongoDB** (PyMongo)
- **NumPy**, **SciPy**, **scikit-learn** (TF-IDF, TruncatedSVD, MiniBatchKMeans)
- **joblib** (ইনডেক্স ডিস্ক ক্যাশ), **python-dotenv**

## ৫. MongoDB কালেকশন

| কালেকশন | কাজ |
|---|---|
| `list_rooms` | রুমের লিস্টিং (শুধু পড়া হয়) |
| `users` | মালিকের নাম ও ফোন (শুধু পড়া হয়, আর কিছু প্রকাশ হয় না) |
| `rec_model` | গ্লোবাল শেখা ওজন ও কাউন্টার |
| `rec_impressions` | প্রতি রিকুয়েস্টে দেখানো রুম ও ফিচার (৭ দিন পর নিজে মুছে যায়) |
| `rec_room_stats` | প্রতি রুমের ইমপ্রেশন ও ক্লিক |

## ৬. সেটআপ ও চালানো

```bash
pip install fastapi uvicorn pymongo numpy scipy scikit-learn joblib python-dotenv
```

`main.py`-এর পাশে একটি `.env` ফাইল বানান:

```env
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>/<database>
ADMIN_KEY=your-secret-admin-key
CORS_ORIGINS=*
```

| ভ্যারিয়েবল | আবশ্যক | বিবরণ |
|---|---|---|
| `MONGO_URI` | হ্যাঁ | MongoDB কানেকশন লিংক |
| `ADMIN_KEY` | হ্যাঁ | রিইনডেক্স এন্ডপয়েন্টের গোপন কী |
| `CORS_ORIGINS` | না | কমা দিয়ে আলাদা করা অনুমোদিত ওয়েবসাইট (ডিফল্ট `*`) |

চালান:

```bash
python main.py
# অথবা
uvicorn main:app --host 0.0.0.0 --port 8000
```

API পরীক্ষা করতে `http://localhost:8000/docs` খুলুন।

## ৭. ডিজাইনের সিদ্ধান্তসমূহ

- **সব স্কোরিং ভেক্টরাইজড:** NumPy দিয়ে সব রুমের হিসাব একসাথে হয়, তাই ১ লাখ রুম র‍্যাংক করতেও দ্রুত থাকে।
- **রেসপন্স পাঠানোর পরে ইমপ্রেশন লগ:** ব্যাকগ্রাউন্ড টাস্কে হয়, তাই ইউজারকে অপেক্ষা করতে হয় না।
- **ব্যাখ্যাযোগ্য মডেল:** ওজনগুলো সহজ লজিস্টিক রিগ্রেশনের, তাই `/api/v1/model` দিয়ে সিস্টেম ঠিক কী শিখেছে তা দেখা যায়।
- **ডিফল্টভাবে নিরাপদ:** মোছা রুম বাদ যায়, ভেতরের ফিল্ড রেসপন্সে লুকানো থাকে, এবং `users` থেকে শুধু মালিকের নাম ও ফোন পড়া হয়।
- **স্থিতিশীল:** ডাটাবেস কিছুক্ষণ না পাওয়া গেলে ক্যাশ করা ওজন ব্যবহার হয়, আর ইনডেক্স রিবিল্ড একসাথে দুইবার চলে না।