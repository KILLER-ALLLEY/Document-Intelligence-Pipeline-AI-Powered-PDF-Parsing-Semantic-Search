# ESG Sentence Analyzer

A Flask + PDF.js app that extracts and highlights ESG-related sentences from PDFs (normal & scanned) and allows semantic search.

## Features
- Extracts sentences from normal PDFs and scanned PDFs using OCR fallback.
- Highlights ESG-related sentences in the PDF viewer.
- Keyword-based semantic search using sentence embeddings.

## Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd esg-sentence-analyzer
````

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
# Activate the environment:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download NLTK data (for sentence tokenization)**

```python
import nltk
nltk.download('punkt')
```

5. **Run the app**

```bash
python app.py
```

6. **Open in browser**

```
http://localhost:8080
```

---

## Usage

* Upload a PDF file (normal or scanned).
* ESG-related sentences will be highlighted in the viewer.
* Use the search box to find keywords and view relevant sentences.

---

## Notes

* Keep your API keys in the `api_key/` folder (ignored by Git).
* Ensure `pdf.js` static files (`index.html`, `viewer.html`) are present for proper PDF viewing.
* Small sample PDFs can be used for demo; avoid committing large files.

---

## Demo Video

[Watch here](https://www.loom.com/share/314735c7414d4129b421ac5b5a53c3e0?sid=fb5db2d4-044b-4db0-ac8a-7d97fdd9c5d9) 

---

## Next Steps / Future Improvements

* Deploy online for easy access.
* Optimize OCR speed for scanned PDFs.
* Improve UI/UX for better usability.
* Enhance semantic search with more advanced embeddings.
---
