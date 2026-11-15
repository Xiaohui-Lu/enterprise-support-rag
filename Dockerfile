# The support assistant's image (D-052): the chat UI built with Node, then the service on
# Python 3.12 with CPU-only PyTorch, with the articles and the embedding model baked in at
# their pinned revisions. Secrets never enter it: they arrive at run time (compose.yaml).
#
#   docker compose --profile app up --build

# --- The chat UI (D-051) ------------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY web/ ./
RUN npm run build

# --- The service (D-050) ------------------------------------------------------------------
FROM python:3.12-slim AS app
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

ENV SRAG_ROOT=/app \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/hf \
    PATH="/opt/venv/bin:$PATH"
RUN useradd --system --create-home --uid 10001 srag \
    && mkdir -p /app/cache /app/data/raw /opt/hf \
    && chown srag:srag /app/cache /app/data/raw /opt/hf
WORKDIR /app

# Dependencies first, exactly as uv.lock pins them, so a code change doesn't reinstall them.
# On Linux, pyproject.toml takes PyTorch from its CPU-only index. uv's download cache lives in
# a build cache mount, not in a layer: left in the layer, it added 1.3 GB to the image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# The articles and the model, at their pinned revisions, so every build holds the same bytes
# and the container never needs Hugging Face (D-052). Loading the model once, offline and as
# the user the service runs as, checks both, and caches the model's own code where it can.
COPY data/splits/ data/splits/
USER srag
RUN srag download
RUN python -c "from huggingface_hub import snapshot_download; \
from support_rag.embeddings import MODELS; m = MODELS['voyage-4-nano']; \
snapshot_download(m.repo_id, revision=m.revision)"
ENV HF_HUB_OFFLINE=1
RUN python -c "from support_rag.embeddings import MODELS, SentenceTransformerEncoder; \
SentenceTransformerEncoder(MODELS['voyage-4-nano']).encode_queries(['a test question'])"

COPY --from=web /web/dist web/dist
EXPOSE 8000
CMD ["srag", "serve", "--host", "0.0.0.0", "--port", "8000"]
