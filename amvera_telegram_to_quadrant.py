
import os
import asyncio
import math
import time
import logging
import uuid
import json
import re
from typing import Optional, Tuple

from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.sessions import StringSession
import httpx

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

# OpenAI client is optional (used for embeddings in original code)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Load environment
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ========== ENV / config ==========
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH")

SESSION_STRING = os.getenv("TELETHON_SESSION_STRING")
BOT_TOKEN = os.getenv("BOT_TOKEN")
SESSION_FILE = os.getenv("SESSION_FILE", "/data/telethon_session.session")
SESSION_NAME = os.getenv("SESSION_NAME", "amvera_session")

SOURCE_CHANNEL = os.getenv("SOURCE_CHANNEL")
DEST_CHANNEL = os.getenv("DEST_CHANNEL")

# OpenAI embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "telegram_news")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))

# LLM classifier (Kong / proxy)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://kong-proxy.yc.amvera.ru/api/v1/models/llama")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama8b")

DEFAULT_VECTOR_SIZES = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
VECTOR_SIZE = DEFAULT_VECTOR_SIZES.get(EMBEDDING_MODEL, 1536)

# Keyword hints (русские + англ варианты) — ранняя фильтрация
KEYWORD_HINTS = [
    "акци", "бирж", "курс", "доллар", "евро", "рубл", "цен", "инфляц",
    "процент", "ставк", "рынок", "фонд", "индекс", "nasdaq", "sp", "s&p",
    "дивиден", "финанс", "эконом", "рецесс", "ввп", "облигац", "доходн",
    "отчет", "отчетность", "баланс", "выручк", "прибыл", "речь", "ставк"
]

# Global clients
openai_client: Optional[object] = None
qdrant_client: Optional[QdrantClient] = None
tg: Optional[TelegramClient] = None


# ========== Utilities ==========
def make_client():
    if SESSION_STRING:
        logger.info("Using TELETHON_SESSION_STRING (StringSession) for Telethon authentication.")
        return TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH)
    if BOT_TOKEN:
        logger.info("Using BOT_TOKEN (bot account).")
        return TelegramClient(SESSION_NAME, API_ID, API_HASH)
    logger.info(f"Using session file: {SESSION_FILE}")
    return TelegramClient(SESSION_FILE, API_ID, API_HASH)


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def normalize_text_for_uuid(text: str) -> str:
    return (text or "").strip().lower()


def has_keyword_hint(text: str) -> bool:
    """Быстрая проверка по ключевым словам (case-insensitive)."""
    if not text:
        return False
    low = text.lower()
    for kw in KEYWORD_HINTS:
        if kw in low:
            return True
    return False


# ========== Init clients ==========
async def init_clients():
    global openai_client, qdrant_client
    loop = asyncio.get_event_loop()

    # OpenAI embeddings client
    if OPENAI_API_KEY and OpenAI is not None:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.exception(f"Failed to init OpenAI client: {e}")
            openai_client = None
    else:
        if OPENAI_API_KEY and OpenAI is None:
            logger.warning("OpenAI SDK not available - install openai package or adapt code.")
        else:
            logger.warning("OPENAI_API_KEY not set - embeddings disabled.")
        openai_client = None

    # Qdrant client
    if QDRANT_URL:
        try:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            logger.info("Qdrant client initialized")
        except Exception as e:
            logger.exception(f"Failed to initialize Qdrant client: {e}")
            qdrant_client = None
    else:
        logger.warning("QDRANT_URL missing - Qdrant disabled")
        qdrant_client = None


async def preflight_checks():
    loop = asyncio.get_event_loop()
    if openai_client:
        try:
            await loop.run_in_executor(None, lambda: openai_client.embeddings.create(model=EMBEDDING_MODEL, input=["health check"]))
            logger.info("OpenAI embedding check OK")
        except Exception as e:
            logger.exception(f"OpenAI health check failed: {e}")
    if qdrant_client:
        try:
            await loop.run_in_executor(None, lambda: qdrant_client.get_collections())
            logger.info("Qdrant health check OK")
        except Exception as e:
            logger.exception(f"Qdrant health check failed: {e}")


async def ensure_collection():
    if qdrant_client is None:
        logger.warning("Qdrant client not configured — skipping collection creation.")
        return
    loop = asyncio.get_event_loop()
    try:
        cols = await loop.run_in_executor(None, lambda: qdrant_client.get_collections())
        names = [c.name for c in cols.collections]
    except Exception as e:
        logger.warning(f"Failed to get collections: {e}")
        names = []

    if COLLECTION_NAME in names:
        logger.info(f"Collection '{COLLECTION_NAME}' exists")
        return

    logger.info(f"Creating collection '{COLLECTION_NAME}' size={VECTOR_SIZE}")
    def _create():
        try:
            return qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=rest_models.VectorParams(size=VECTOR_SIZE, distance=rest_models.Distance.COSINE),
            )
        except TypeError:
            return qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors=rest_models.VectorParams(size=VECTOR_SIZE, distance=rest_models.Distance.COSINE),
            )
    try:
        await loop.run_in_executor(None, _create)
        logger.info("Collection created")
    except Exception as e:
        logger.exception(f"Failed to create collection: {e}")


# ========== Embeddings / Qdrant ops ==========
async def embed_text(text: str):
    if not openai_client:
        raise RuntimeError("OpenAI client not configured")
    if not text:
        return []
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(None, lambda: openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text))
        emb = resp.data[0].embedding
        return emb
    except Exception as e:
        logger.exception(f"Embedding failed: {e}")
        raise


async def is_similar_to_existing(vec):
    if qdrant_client is None:
        return False, 0.0, None
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=5, with_payload=True, with_vectors=True)
        )
    except TypeError:
        try:
            results = await loop.run_in_executor(
                None,
                lambda: qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=5, with_payload=True)
            )
        except Exception as e:
            logger.warning(f"Qdrant search failed (fallback): {e}")
            return False, 0.0, None
    except Exception as e:
        logger.warning(f"Qdrant search failed: {e}")
        return False, 0.0, None

    best_sim = 0.0
    best_hit = None
    for r in results:
        candidate_vector = getattr(r, "vector", None)
        if candidate_vector is None and getattr(r, "payload", None):
            candidate_vector = r.payload.get("_vector") or r.payload.get("vector")
        if candidate_vector:
            try:
                sim = cosine_sim(vec, candidate_vector)
                if sim > best_sim:
                    best_sim = sim
                    best_hit = r
            except Exception:
                logger.debug("Skipping candidate vector due to mismatch")
    return (best_sim >= SIMILARITY_THRESHOLD), best_sim, best_hit


async def upsert_point(point_id: str, vector, payload: dict):
    if qdrant_client is None:
        logger.warning("Qdrant not configured — skipping upsert")
        return
    try:
        uuid.UUID(point_id)
        final_id = point_id
    except Exception:
        final_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(point_id)))
    pt = rest_models.PointStruct(id=final_id, vector=vector, payload=payload)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[pt]))
    except Exception as e:
        logger.exception(f"Upsert failed: {e}")
        raise


# ========== LLM-based topic classifier (русский prompt) ==========
async def classify_with_llm(text: str) -> Tuple[bool, str]:
    """
    Возвращает (is_relevant, reason).
    Использует LLM_BASE_URL + LLM_API_KEY.
    Если LLM недоступен или ответ не распарсить — делаем keyword fallback.
    """
    text = text.strip()
    if not text:
        return False, "empty_text"

    if not LLM_BASE_URL or not LLM_API_KEY:
        return keyword_fallback(text)

    prompt = (
        "Вы — точный классификатор. Определите, относится ли следующее сообщение к финансово-экономической "
        "тематике: акции (фондовый рынок), движение индексов, курсы валют, процентные ставки, "
        "макроэкономические показатели (ВВП, инфляция и т.п.), финансовая отчётность компаний, облигации, "
        "доходности, банковские/финансовые новости и другие экономические события.\n\n"
        "ОТВЕЧАЙТЕ ТОЛЬКО ОДНИМ КОРРЕКТНЫМ JSON-ОБЪЕКТОМ И НИЧЕМ БОЛЕЕ, В ТАКОМ ВИДЕ:\n"
        '{"relevant": true|false, "reason": "короткое объяснение на русском", "labels": ["акции","курс_валют","инфляция"]}\n\n'
        "Поле 'relevant' — true если сообщение релевантно, false — если нет. "
        "В 'labels' перечислите короткие метки (например: \"акции\", \"курс_валют\", \"инфляция\", \"фин_отчётность\", \"облигации\"). "
        "В 'reason' дайте краткое пояснение (1–2 коротких фразы).\n\n"
        f"Сообщение для анализа:\n\n{text}\n\n"
        "Возвращайте ровно один JSON-объект и ничего больше."
    )

    payload = {"model": LLM_MODEL, "messages": [{"role": "user", "text": prompt}]}
    headers = {"X-Auth-Token": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(LLM_BASE_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidate_text = None
        if isinstance(data, dict):
            candidate_text = (
                data.get("result", {})
                .get("alternatives", [{}])[0]
                .get("message", {})
                .get("text")
            )
            if not candidate_text:
                candidate_text = data.get("text") or data.get("result") or None
                if isinstance(candidate_text, dict):
                    candidate_text = json.dumps(candidate_text)
        if candidate_text is None:
            candidate_text = json.dumps(data)

        # Попытка извлечь JSON из ответа
        json_obj = None
        try:
            json_obj = json.loads(candidate_text)
        except Exception:
            m = re.search(r'\{.*\}', candidate_text, flags=re.S)
            if m:
                try:
                    json_obj = json.loads(m.group(0))
                except Exception:
                    json_obj = None

        if isinstance(json_obj, dict):
            relevant = bool(json_obj.get("relevant") is True or json_obj.get("relevant") == "true")
            reason = str(json_obj.get("reason") or json_obj.get("explanation") or "")
            labels = json_obj.get("labels", [])
            reason_summary = reason or (", ".join(labels) if labels else "")
            return relevant, reason_summary or "classified_by_llm"
        else:
            logger.warning("LLM response parsing failed, using keyword fallback")
            return keyword_fallback(text)

    except Exception as e:
        logger.exception(f"LLM classify request failed: {e}")
        return keyword_fallback(text)


def keyword_fallback(text: str) -> Tuple[bool, str]:
    low = text.lower()
    found = [kw for kw in KEYWORD_HINTS if kw.lower() in low]
    if found:
        return True, f"keyword_fallback: {', '.join(found[:5])}"
    return False, "keyword_fallback: no_keywords"


# ========== Event handler (core logic) ==========
@events.register(events.NewMessage)
async def global_handler(event):
    try:
        if SOURCE_CHANNEL and str(event.chat_id) not in (str(SOURCE_CHANNEL), SOURCE_CHANNEL):
            return

        msg = event.message
        text = (msg.message or "").strip()
        if not text:
            logger.info("Empty message — skip")
            return

        # Ранняя фильтрация по ключевым словам: если нет ни одного хинта — skip
        if not has_keyword_hint(text):
            logger.info("Early keyword filter: no keyword hints found — skip")
            return

        logger.info(f"New message (id={msg.id}) — passed keyword filter, embedding...")
        try:
            emb = await embed_text(text)
        except Exception:
            logger.exception("Embedding error — skipping message")
            return

        if not emb:
            logger.warning("Empty embedding — skip")
            return

        # check similarity
        similar, sim_score, hit = await is_similar_to_existing(emb)
        logger.info(f"Similarity: {sim_score:.4f} (threshold={SIMILARITY_THRESHOLD})")
        if similar:
            logger.info("Message is similar to existing — skip sending")
            return

        # If unique, check topic relevance via LLM
        logger.info("Classifying message topic with LLM...")
        is_relevant, reason = await classify_with_llm(text)
        logger.info(f"LLM relevance: {is_relevant}, reason: {reason}")

        if not is_relevant:
            logger.info("Message not relevant to finance/economics — skip and DO NOT upsert")
            return

        # Relevant => upsert then send
        normalized = normalize_text_for_uuid(text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, normalized))
        payload = {
            "text": text,
            "chat_id": str(event.chat_id),
            "message_id": msg.id,
            "date": str(msg.date.isoformat() if getattr(msg, "date", None) else time.time()),
            "llm_relevance_reason": reason,
        }

        try:
            await upsert_point(point_id, emb, payload)
            logger.info("Upserted point into Qdrant")
        except Exception:
            logger.exception("Upsert failed — not sending to avoid duplicates")
            return

        # Send to destination channel
        if DEST_CHANNEL:
            try:
                await tg.send_message(DEST_CHANNEL, text)
                logger.info(f"Sent to destination {DEST_CHANNEL}")
            except ValueError as ve:
                logger.error("Cannot get entity for DEST_CHANNEL — ensure bot/user is member and has rights")
                logger.exception(ve)
            except Exception as e:
                logger.exception(f"Failed to send message to DEST_CHANNEL: {e}")
        else:
            logger.warning("DEST_CHANNEL not set — message not sent")

    except Exception as e:
        logger.exception(f"Unhandled error in handler: {e}")


# ========== Main ==========
async def main():
    global tg
    if API_ID == 0 or not API_HASH:
        logger.error("API_ID or API_HASH missing")
        raise SystemExit(1)
    if not (SESSION_STRING or BOT_TOKEN or os.path.exists(SESSION_FILE)):
        logger.error("No non-interactive auth set: set TELETHON_SESSION_STRING or BOT_TOKEN or mount SESSION_FILE")
        raise SystemExit(1)

    tg = make_client()
    await init_clients()

    try:
        if BOT_TOKEN:
            await tg.start(bot_token=BOT_TOKEN)
        else:
            await tg.start()
    except EOFError:
        logger.exception("Interactive login attempted — provide TELETHON_SESSION_STRING or BOT_TOKEN")
        raise
    except Exception as e:
        logger.exception(f"Telegram client start failed: {e}")
        raise

    # Register handler (with chat filter if SOURCE_CHANNEL specified)
    if SOURCE_CHANNEL:
        tg.add_event_handler(global_handler, events.NewMessage(chats=SOURCE_CHANNEL))
    else:
        tg.add_event_handler(global_handler, events.NewMessage())

    logger.info("Telegram client started")
    await preflight_checks()
    await ensure_collection()
    logger.info("Listening for new messages...")
    await tg.run_until_disconnected()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception:
        logger.exception("Fatal error in main")
