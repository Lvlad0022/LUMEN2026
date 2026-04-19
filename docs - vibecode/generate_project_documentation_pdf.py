from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "project_documentation_hr.md"
OUTPUT = ROOT / "docs" / "projektna_dokumentacija_lumen2026.pdf"
FONT_PATH = Path("/Library/Fonts/Arial Unicode.ttf")
FONT_BOLD_PATH = Path("/Library/Fonts/Arial Unicode.ttf")


def load_markdown_sections(path: Path) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            if current_title or current_lines:
                sections.append((current_title, current_lines))
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title or current_lines:
        sections.append((current_title, current_lines))
    return sections


def build_paragraph_style(
    styles,
    name: str,
    font_name: str,
    font_size: int,
    leading: int,
    *,
    alignment: int = 0,
    space_after: int = 0,
    space_before: int = 0,
    left_indent: int = 0,
    bullet_indent: int = 0,
):
    return ParagraphStyle(
        name=name,
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=font_size,
        leading=leading,
        alignment=alignment,
        spaceAfter=space_after,
        spaceBefore=space_before,
        leftIndent=left_indent,
        bulletIndent=bullet_indent,
    )


def convert_markdown_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("- "):
        return "&bull; " + stripped[2:]
    if stripped.startswith("1. "):
        return stripped
    return stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source markdown: {SOURCE}")

    pdfmetrics.registerFont(TTFont("NotoSans", str(FONT_PATH)))
    pdfmetrics.registerFont(TTFont("NotoSans-Bold", str(FONT_BOLD_PATH)))

    styles = getSampleStyleSheet()
    title_style = build_paragraph_style(styles, "TitleHR", "NotoSans-Bold", 20, 24, alignment=TA_CENTER, space_after=10)
    h1_style = build_paragraph_style(styles, "Heading1HR", "NotoSans-Bold", 15, 18, space_before=10, space_after=6)
    h2_style = build_paragraph_style(styles, "Heading2HR", "NotoSans-Bold", 12, 15, space_before=8, space_after=4)
    body_style = build_paragraph_style(styles, "BodyHR", "NotoSans", 10.5, 14, space_after=3)
    bullet_style = build_paragraph_style(styles, "BulletHR", "NotoSans", 10.5, 14, left_indent=10, bullet_indent=0, space_after=2)

    story = []
    story.append(Paragraph("LUMEN2026 - Projektna dokumentacija", title_style))
    story.append(Paragraph("Sažetak rada, podataka, VAE embeddinga, klasteriranja i usporedbe segmenata", body_style))
    story.append(Spacer(1, 6 * mm))

    sections = load_markdown_sections(SOURCE)
    for title, lines in sections:
        if not title:
            continue
        heading_level = 1 if title and not title.startswith("9.") and not title.startswith("8.") and not title.startswith("7.") and not title.startswith("6.") and not title.startswith("5.") and not title.startswith("4.") and not title.startswith("3.") and not title.startswith("2.") else 2
        story.append(Paragraph(title, h1_style if heading_level == 1 else h2_style))
        for line in lines:
            rendered = convert_markdown_line(line)
            if not rendered:
                story.append(Spacer(1, 2 * mm))
                continue
            if rendered.startswith("&bull; "):
                story.append(Paragraph(rendered, bullet_style))
            elif rendered.startswith("```"):
                continue
            else:
                story.append(Paragraph(rendered, body_style))
        story.append(Spacer(1, 2 * mm))

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
