# ESG Sentence Analyzer – Product Decisions and Tradeoffs

## Key Product Decisions
- **Framework**: Chose Flask for lightweight web app and easy integration with HTML/JS frontend.
- **PDF rendering**: Used PDF.js for accurate PDF viewing in browser.
- **Semantic search**: Used sentence-transformers for meaningful keyword search.
- **OCR fallback**: Added pytesseract + PIL for scanned PDFs.

## Tradeoffs Made
- OCR for scanned PDFs can be slow; prioritized functionality over speed.
- Skipped advanced UI/UX improvements due to time constraints.
- Only basic keyword highlighting implemented; did not implement clustering or ranking of results.

## Next Steps / Improvements
- Optimize OCR performance for large PDFs.
- Deploy the app online for easy access.
- Enhance semantic search with larger/custom embeddings.
- Improve UI/UX: better PDF viewer, responsive design, interactive highlights.
- I already had embeddings easily could have made the chat with the pdf

  ## Journey
  I really didnt have much time and was not doing this full time cause had some work so i had to work with this at night...so my main working hours was 10pm to 5am so i couldnt add all the features and make it really polished but i made it sophsticated...you can go tthorugh the code.
