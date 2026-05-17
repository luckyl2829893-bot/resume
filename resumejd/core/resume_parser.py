import io
import fitz  # PyMuPDF
from docx import Document

def parse_resume(file) -> str:
    """
    Parses a PDF or DOCX file (either standard file path or Streamlit UploadedFile buffer)
    and returns clean concatenated text.
    """
    # 1. Check if the file is a Streamlit UploadedFile or file-like buffer
    if hasattr(file, "read"):
        file_bytes = file.read()
        # Reset file pointer for potential future reads
        if hasattr(file, "seek"):
            file.seek(0)
            
        file_name = getattr(file, "name", "").lower()
        
        if file_name.endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        elif file_name.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            text_blocks = []
            for paragraph in doc.paragraphs:
                text_blocks.append(paragraph.text)
            return "\n".join(text_blocks).strip()
        else:
            # Fallback based on signature/bytes if filename is missing
            try:
                if file_bytes.startswith(b"%PDF"):
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                    return text.strip()
            except Exception:
                pass
            raise ValueError("Unsupported file format. Please upload a standard PDF or DOCX.")
            
    # 2. Treat as a standard path string or Path object
    else:
        file_path = str(file).lower()
        if file_path.endswith(".pdf"):
            doc = fitz.open(file)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        elif file_path.endswith(".docx"):
            doc = Document(file)
            text_blocks = []
            for paragraph in doc.paragraphs:
                text_blocks.append(paragraph.text)
            return "\n".join(text_blocks).strip()
        else:
            raise ValueError("Unsupported file format. Please provide a standard PDF or DOCX file path.")

def extract_layout_profile(file_bytes: bytes) -> dict:
    """
    Performs visual fingerprinting of the uploaded PDF resume.
    Extracts:
    - Dominant accent colors (Teal, orange, red, etc.)
    - Candidate name font size and alignment
    - Exact original section ordering to mirror sequence
    - Bullet point styles
    - Core font size configurations
    """
    profile = {
        "accent_color": (216, 90, 48),  # Default Coral Accent (#D85A30)
        "accent_color_hex": "D85A30",
        "name_font_size": 22,
        "name_align": "C",
        "section_order": ["SUMMARY", "EXPERIENCE", "PROJECTS", "EDUCATION", "SKILLS", "INTERESTS"],
        "bullet_char": "•",
        "heading_font_size": 11,
        "body_font_size": 10
    }
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if len(doc) == 0:
            return profile
            
        page = doc[0]
        blocks = page.get_text("dict").get("blocks", [])
        
        color_counts = {}
        max_name_size = 0
        name_span = None
        
        # Heading and body font size accumulators
        heading_sizes = []
        body_sizes = []
        
        # Section ordering tracking
        found_sections = []
        KNOWN_SECTIONS = [
            "summary", "profile", "objective", "experience", "work",
            "projects", "education", "skills", "technical", "certifications",
            "achievements", "interests", "hobbies", "links"
        ]
        
        # Bullet char counts
        bullet_counts = {"•": 0, "-": 0, "*": 0, "▪": 0, "▸": 0}
        
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    color = span.get("color", 0)
                    font_flags = span.get("flags", 0)
                    is_bold = bool(font_flags & 2) or "bold" in span.get("font", "").lower()
                    
                    if not text:
                        continue
                        
                    # Color Extraction (sRGB conversion)
                    r = (color >> 16) & 0xFF
                    g = (color >> 8) & 0xFF
                    b = color & 0xFF
                    
                    # Ignore black/near-black and white/near-white colors
                    if not (r < 30 and g < 30 and b < 30) and not (r > 220 and g > 220 and b > 220):
                        rgb = (r, g, b)
                        color_counts[rgb] = color_counts.get(rgb, 0) + len(text)
                        
                    # 1. Candidate Name detection (font size > 18)
                    if size > 18:
                        if size > max_name_size:
                            max_name_size = size
                            name_span = span
                            
                    # 2. Section Headings (bold, size between 10-14)
                    elif is_bold and (10 <= size <= 14):
                        heading_sizes.append(size)
                        
                        # Section ordering check
                        text_lower = text.lower()
                        for kw in KNOWN_SECTIONS:
                            if kw in text_lower:
                                # Standardize section names
                                canonical = text.upper().replace("**", "").replace("#", "").strip()
                                if canonical not in found_sections and len(canonical) < 40:
                                    found_sections.append(canonical)
                                    
                    # 3. Body text (size < 11)
                    elif size < 11:
                        body_sizes.append(size)
                        
                    # 4. Bullet character detection
                    for char in bullet_counts.keys():
                        if text.startswith(char):
                            bullet_counts[char] += 1
                            
        # Finalize Dominant accent color
        if color_counts:
            dominant = max(color_counts, key=color_counts.get)
            profile["accent_color"] = dominant
            profile["accent_color_hex"] = f"{dominant[0]:02x}{dominant[1]:02x}{dominant[2]:02x}"
            
        # Finalize Name Alignment & Size
        if name_span:
            profile["name_font_size"] = min(24, max(16, int(name_span["size"])))
            bbox = name_span["bbox"]
            mid_x = (bbox[0] + bbox[2]) / 2
            page_width = page.rect.width
            # If name resides within center region, mark as center-aligned
            if abs(mid_x - (page_width / 2)) < 55:
                profile["name_align"] = "C"
            else:
                profile["name_align"] = "L"
                
        # Finalize Font Size tiers
        if heading_sizes:
            profile["heading_font_size"] = int(sum(heading_sizes) / len(heading_sizes))
        if body_sizes:
            profile["body_font_size"] = round(sum(body_sizes) / len(body_sizes), 1)
            
        # Finalize Section Order
        if found_sections:
            profile["section_order"] = found_sections
            
        # Finalize Bullet Style
        max_bullet = max(bullet_counts, key=bullet_counts.get)
        if bullet_counts[max_bullet] > 0:
            profile["bullet_char"] = max_bullet
            
        doc.close()
    except Exception as e:
        print(f"[Parser] Layout profiling failed (falling back to elegant defaults): {e}")
        
    return profile

