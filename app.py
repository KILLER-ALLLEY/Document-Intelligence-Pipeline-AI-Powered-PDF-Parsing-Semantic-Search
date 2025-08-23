from flask import Flask, render_template, request, url_for, send_file
from io import BytesIO
from services.pdf_service import extract_pdf_sentences_with_ocr_fallback
from create_embeddings.create_embeddings_sentences import create_embeddings
from semantic_search.semantic_search import run_semantic_search

app = Flask(__name__)

pdf_storage = {}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", rows=[])

@app.route("/upload", methods=["POST"])
def upload_pdf():
    pdf_file = request.files.get("pdf_file")
    if not pdf_file:
        return "No file uploaded", 400

    pdf_bytes = pdf_file.read()
    pdf_storage["file"] = pdf_bytes
    pdf_storage["filename"] = pdf_file.filename

    try:
        print(f"📄 Processing PDF: {pdf_file.filename}")
        extracted_sentences = extract_pdf_sentences_with_ocr_fallback(pdf_bytes)
        
        if not extracted_sentences:
            print("⚠️ No sentences extracted from PDF")
            return render_template("index.html", rows=[], error="No text could be extracted from the PDF")

        print(f"✅ Extracted {len(extracted_sentences)} pages")
        total_sentences = sum(len(page.get('sentences', [])) for page in extracted_sentences)
        print(f"✅ Total sentences: {total_sentences}")

        print("🔄 Creating embeddings...")
        embeddings_result = create_embeddings(extracted_sentences)
        
        if not embeddings_result:
            print("⚠️ No embeddings created")
            return render_template("index.html", rows=[], error="Failed to create embeddings")

        print(f"✅ Created {len(embeddings_result)} embeddings")

        print("🔍 Running semantic search...")
        search_results = run_semantic_search(embeddings_result)

        # Handle structured errors from semantic search
        if isinstance(search_results, dict) and search_results.get("status") == "error":
            return render_template(
                "index.html",
                rows=[],
                error=search_results.get("message", "Unknown error during semantic search")
            )

        # Filter only results with keyword matches
        results = [r for r in search_results if r.get("keywords")]
        print(f"🎯 Found {len(results)} results with keyword matches")

        # Handle case where no ESG matches were found
        if not results:
            return render_template(
                "index.html",
                rows=[],
                error="No ESG related content found in this document. "
                      "Next steps: verify file and manually review."
            )

        viewer_url = url_for("pdf_viewer")
        return render_template("index.html", rows=results, auto_open_pdf=viewer_url)

    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return render_template("index.html", rows=[], error=f"Error processing PDF: {str(e)}")

@app.route("/pdf_viewer")
def pdf_viewer():
    if "file" not in pdf_storage:
        return "No PDF uploaded", 404

    return render_template("viewer.html", pdf_url=url_for("serve_pdf"))

@app.route("/serve_pdf")
def serve_pdf():
    if "file" not in pdf_storage:
        return "No PDF uploaded", 404

    return send_file(
        BytesIO(pdf_storage["file"]),
        download_name=pdf_storage["filename"],
        mimetype="application/pdf"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
