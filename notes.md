# ESG Sentence Analyzer – Product Decisions and Tradeoffs

## Key Product Decisions
- **Framework**: Chose Flask for a lightweight web app with easy integration of HTML/JS frontend.
- **PDF rendering**: Used PDF.js to display PDFs accurately in the browser.
- **Semantic search**: Implemented using sentence-transformers to provide meaningful keyword-based search.
- **OCR fallback**: Added pytesseract + PIL to handle scanned PDFs.

## Tradeoffs Made
- **Performance**: Sentence-transformers provides accurate results but is relatively slow. Due to limited time and budget, I did not optimize for speed. With more resources, improvements could include smaller embedding models, approximate nearest neighbors, or caching precomputed embeddings.
- **UI/UX**: Advanced UI/UX enhancements were skipped to focus on core functionality.
- **Feature scope**: Only basic keyword highlighting was implemented; clustering or ranking of search results was not included.

## Journey
Due to limited time and other commitments, I worked on this project mostly during nighttime hours (10 PM to 5 AM). While I couldn’t implement every feature or fully polish the UI, I focused on building a sophisticated core system. You can explore the code to see the main functionality and design choices.

## Next Steps / Improvements
- Optimize OCR and semantic search performance for large PDFs.
- Deploy the app online for easy access and demonstration.
- Enhance UI/UX for better interactivity and responsiveness.
- Expand semantic search capabilities with larger or custom embeddings.

