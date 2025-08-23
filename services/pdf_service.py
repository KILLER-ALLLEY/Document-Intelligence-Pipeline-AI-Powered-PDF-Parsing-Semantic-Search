import fitz
import nltk
import logging
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from nltk.tokenize import sent_tokenize
from collections import Counter
from paths import TESSERACT_CMD

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def extract_pdf_sentences_with_ocr_fallback(
        pdf_path: str,
        max_vspace: float = 10.0,
        max_hspace: float = 20.0,
        dpi: int = 300) -> List[Dict[str, Any]]:
    """
    Extract text from PDF into sentences with bounding boxes.
    Uses OCR fallback when no text is found.
    Removes repeated headers/footers and structural headers.
    """

    # ---------------- NLTK DEPENDENCIES ----------------
    def _ensure_nltk_dependencies() -> None:
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)

    # ---------------- HELPER FUNCS ----------------
    def _merge_bboxes(spans: List[Dict[str, Any]]) -> List[float]:
        if not spans:
            return [0.0, 0.0, 0.0, 0.0]
        bboxes = [span['bbox'] for span in spans]
        return [
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes)
        ]

    def _should_merge_spans(current_font: Optional[str],
                            current_size: Optional[float], new_font: str,
                            new_size: float, last_bbox: List[float],
                            new_bbox: List[float]) -> bool:
        if current_font is None:
            return True
        if new_font != current_font or new_size != current_size:
            return False
        vspace = new_bbox[1] - last_bbox[3]
        hspace = new_bbox[0] - last_bbox[2]
        return not (vspace > max_vspace and hspace > max_hspace)

    def _extract_blocks_from_page(page: fitz.Page) -> List[Dict[str, Any]]:
        page_dict = page.get_text("dict")
        blocks = []
        for block_data in page_dict["blocks"]:
            if block_data["type"] != 0:
                continue
            current_text = ""
            current_font = None
            current_size = None
            current_spans = []
            last_bbox = [0.0, 0.0, 0.0, 0.0]
            for line in block_data["lines"]:
                for span_data in line["spans"]:
                    text = span_data["text"]
                    if not text or not text.strip():
                        continue
                    font = span_data["font"]
                    size = span_data["size"]
                    bbox = [float(c) for c in span_data["bbox"]]
                    if _should_merge_spans(current_font, current_size, font,
                                           size, last_bbox, bbox):
                        if current_text and not current_text.endswith(
                                ' ') and not text.startswith(' '):
                            current_text += " "
                        current_text += text
                        current_font = font
                        current_size = size
                        current_spans.append({"text": text, "bbox": bbox})
                        last_bbox = bbox
                    else:
                        if current_text.strip():
                            blocks.append({
                                "text": current_text.strip(),
                                "spans": current_spans
                            })
                        current_text = text
                        current_font = font
                        current_size = size
                        current_spans = [{"text": text, "bbox": bbox}]
                        last_bbox = bbox
            if current_text.strip():
                blocks.append({
                    "text": current_text.strip(),
                    "spans": current_spans
                })
        return blocks

    def _map_spans_to_sentence(block_text: str, sentence_text: str,
                               sentence_start: int,
                               spans: List[Dict]) -> List[Dict]:
        sentence_end = sentence_start + len(sentence_text)
        overlapping_spans = []
        char_pos = 0
        span_positions = []
        for span in spans:
            span_text = span["text"]
            span_start_pos = block_text.find(span_text, char_pos)
            if span_start_pos != -1:
                span_end_pos = span_start_pos + len(span_text)
                span_positions.append({
                    "span": span,
                    "start": span_start_pos,
                    "end": span_end_pos
                })
                char_pos = span_end_pos
        for span_info in span_positions:
            if (span_info["end"] > sentence_start
                    and span_info["start"] < sentence_end):
                overlapping_spans.append(span_info["span"])
        return overlapping_spans

    def _split_block_into_sentences(block: Dict[str, Any]) -> List[Dict[str, Any]]:
        block_text = block["text"]
        sentences = sent_tokenize(block_text)
        result_sentences = []
        char_offset = 0
        for sentence in sentences:
            sentence_text = sentence.strip()
            if not sentence_text:
                continue
            sentence_start = block_text.find(sentence_text, char_offset)
            if sentence_start == -1:
                sentence_start = char_offset
            spans_in_sentence = _map_spans_to_sentence(block_text,
                                                       sentence_text,
                                                       sentence_start,
                                                       block["spans"])
            if spans_in_sentence:
                result_sentences.append({
                    "text": sentence_text,
                    "bbox": _merge_bboxes(spans_in_sentence)
                })
            char_offset = sentence_start + len(sentence_text)
        return result_sentences

    # ---------------- OCR HELPERS ----------------
    def parse_bbox_number(s):
        return int(''.join(filter(str.isdigit, s)))

    def ocr_bbox_pixels_to_pdf_points(x0, y0, x1, y1, dpi, page, pix):
        """
        Convert OCR bbox (pixels, top-left origin) -> PDF points (bottom-left origin).
        FIXED: Y-axis flip correction.
        """
      
        scale = 72.0 / dpi
        px0, py0 = x0 * scale, y0 * scale
        px1, py1 = x1 * scale, y1 * scale

       
        page_width, page_height = page.rect.width, page.rect.height
        img_width_pts, img_height_pts = pix.width * scale, pix.height * scale

        px0 = px0 * (page_width / img_width_pts)
        px1 = px1 * (page_width / img_width_pts)
        py0 = py0 * (page_height / img_height_pts)
        py1 = py1 * (page_height / img_height_pts)

        # Flip Y-axis to match PDF coords
        # py0, py1 = page_height - py1, page_height - py0

        return [px0, py0, px1, py1]

    def words_to_sentences_inside_block(block_words):
        if not block_words:
            return []
        text = " ".join(word["text"] for word in block_words)
        min_x0 = min(word["bbox"][0] for word in block_words)
        min_y0 = min(word["bbox"][1] for word in block_words)
        max_x1 = max(word["bbox"][2] for word in block_words)
        max_y1 = max(word["bbox"][3] for word in block_words)
        return [{"text": text, "bbox": [min_x0, min_y0, max_x1, max_y1]}]

    def _extract_with_ocr(page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        page_sentences = []
        print(f"⚠️ Using OCR (word-level, block-scoped) for page {page_num}")
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0).prerotate(page.rotation)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        hocr_bytes = pytesseract.image_to_pdf_or_hocr(img,
                                                      extension="hocr",
                                                      lang="eng")
        soup = BeautifulSoup(hocr_bytes, "html.parser")

        ocr_blocks = soup.find_all("div", class_="ocr_carea")
        if not ocr_blocks:
            ocr_blocks = soup.find_all("p", class_="ocr_par")

        for blk in ocr_blocks:
            title = blk.get("title", "")
            if "bbox" not in title:
                continue
            bbox_part = title.split(";")[0]
            _, x0s, y0s, x1s, y1s = bbox_part.split()
            bx0, by0, bx1, by1 = map(parse_bbox_number,
                                     (x0s, y0s, x1s, y1s))

            block_words = []
            for wspan in blk.find_all("span", class_="ocrx_word"):
                wtxt = wspan.get_text(strip=True)
                if not wtxt:
                    continue
                wtitle = wspan.get("title", "")
                if "bbox" not in wtitle:
                    continue
                _, wx0s, wy0s, wx1s, wy1s, *_ = wtitle.split()
                wx0, wy0, wx1, wy1 = map(parse_bbox_number,
                                         (wx0s, wy0s, wx1s, wy1s))
                fx0, fy0, fx1, fy1 = ocr_bbox_pixels_to_pdf_points(
                    wx0, wy0, wx1, wy1, dpi, page, pix)
                block_words.append({
                    "text": wtxt,
                    "bbox": [fx0, fy0, fx1, fy1]
                })

            if block_words:
                page_sentences.extend(
                    words_to_sentences_inside_block(block_words))
        return page_sentences

    # ---------------- FILTER HEADERS/FOOTERS ----------------
    def _filter_headers_and_footers(sentences, page_height, freq_map,
                                    top_thresh=0.12, bottom_thresh=0.08,
                                    min_repeats=3):
        filtered = []
        for sent in sentences:
            y0 = sent["bbox"][1]
            y1 = sent["bbox"][3]
            norm_y0 = y0 / page_height
            norm_y1 = y1 / page_height
            text = sent["text"].strip()
            repeats = freq_map.get(text, 0)

            if repeats >= min_repeats and (norm_y1 < top_thresh or norm_y0 > (1 - bottom_thresh)):
                continue

            if (norm_y1 < top_thresh or norm_y0 > (1 - bottom_thresh)):
                if (text.isupper() and ("\t" in text or "  " in text or len(text.split()) > 6)):
                    continue

            filtered.append(sent)
        return filtered

    # ---------------- MAIN ----------------
    try:
        _ensure_nltk_dependencies()
        doc = fitz.open(stream=pdf_path, filetype="pdf") if isinstance(pdf_path, (bytes, bytearray)) else fitz.open(pdf_path)
        raw_output = []
        for page_number in range(len(doc)):
            try:
                page = doc.load_page(page_number)
                blocks = _extract_blocks_from_page(page)
                page_sentences = []
                for block in blocks:
                    page_sentences.extend(_split_block_into_sentences(block))
                if not page_sentences:
                    page_sentences = _extract_with_ocr(page, page_number + 1)
                raw_output.append({
                    "page_num": page_number + 1,
                    "sentences": page_sentences
                })
            except Exception as e:
                logger.warning(f"Error processing page {page_number + 1}: {e}")
                continue

        all_texts = [s["text"] for page in raw_output for s in page["sentences"]]
        freq_map = Counter(all_texts)


        output = []
        for page in raw_output:
            page_height = doc[page["page_num"] - 1].rect.height
            cleaned = _filter_headers_and_footers(page["sentences"], page_height, freq_map)
            if cleaned:
                output.append({"page_num": page["page_num"], "sentences": cleaned})

        doc.close()
        return output
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        return []
