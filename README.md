# Find My Room - Recommendation Engine

This is a fast, self-learning recommendation engine built with FastAPI, scikit-learn, and MongoDB.

## Features
- Real-time personalization using TF-IDF and SVD for text analysis.
- Continuous online learning based on user feedback (clicks, saves, etc.).
- Efficient vectorised scoring over thousands of items.
- Caches results to disk for fast startup.

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd recomandationAi
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server:**
   ```bash
   python main.py
   ```
   Or using Uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Deploying on Render

1. Create a new **Web Service** on Render.
2. Connect your GitHub/GitLab repository.
3. Use the following configuration:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy the service.

*Note: Ensure your MongoDB instance allows connections from Render's IP addresses (or allow access from anywhere `0.0.0.0/0` if applicable).*

## API Endpoints

- `GET /` - Health and status
- `POST /api/v1/recommendations` - Get room recommendations
- `POST /api/v1/feedback` - Send user feedback to improve the model
- `GET /api/v1/model` - View current model state and weights
- `POST /api/v1/admin/reindex` - Trigger manual index rebuild (Requires `x_admin_key` header)
