import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5437/rag")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def get_company_profile() -> str:
    """Текст профиля компании из COMPANY_PROFILE_FILE или COMPANY_PROFILE."""
    path = os.environ.get("COMPANY_PROFILE_FILE", "").strip()
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                t = f.read().strip()
                if t:
                    return t
        except OSError:
            pass
    return os.environ.get("COMPANY_PROFILE", "").strip()
