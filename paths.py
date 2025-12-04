import os

# base directory inside your project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_ROOT, "data")

# ensure data folder exists
os.makedirs(BASE_DIR, exist_ok=True)

# file paths
KEYWORDS_FILE = os.path.join(BASE_DIR, "keywords.json")
KEYWORD_EMBEDDINGS_FILE = os.path.join(BASE_DIR, "keyword_embeddings.json")
SAVE_PATH = os.path.join(BASE_DIR, "semantic_search_results.json")
SAVE_PATH_SENTENCES = os.path.join(BASE_DIR, "sentence_embeddings.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "keyword_embeddings.json")

# Tesseract path
# On Render (Linux) tesseract will be installed at /usr/bin/tesseract
# On Windows, the env variable "TESSERACT_CMD" overrides it
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"   # Windows default
)
