from pathlib import Path
import ast
import html
import json
import math
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, KeepTogether, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'pdf' / 'python_crash_course_chapters_2_and_3.pdf'
W, H = A4
MARGIN = 49
WIDTH = W - 2 * MARGIN
NAVY = colors.HexColor('#173247')
INK = colors.HexColor('#243B4B')
TEAL = colors.HexColor('#007F86')
MUTED = colors.HexColor('#536A79')
PALE = colors.HexColor('#F0F5F7')
LINE = colors.HexColor('#D9E4E9')
AMBER = colors.HexColor('#E4AB50')

FONTDIR = Path('C:/Windows/Fonts')
for name, filename in [
    ('Body', 'calibri.ttf'), ('BodyB', 'calibrib.ttf'),
    ('BodyI', 'calibrii.ttf'), ('BodyBI', 'calibriz.ttf'),
    ('Code', 'consola.ttf'), ('CodeB', 'consolab.ttf'),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONTDIR / filename)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyB',
                              italic='BodyI', boldItalic='BodyBI')
pdfmetrics.registerFontFamily('Code', normal='Code', bold='CodeB')

STYLES = {
    'body': ParagraphStyle('body', fontName='Body', fontSize=10.5,
                           leading=14.5, textColor=INK, spaceAfter=7.0,
                           allowWidows=0, allowOrphans=0),
    'small': ParagraphStyle('small', fontName='Body', fontSize=9.1,
                            leading=12.4, textColor=MUTED, spaceAfter=6.5,
                            keepWithNext=True),
    'h1': ParagraphStyle('h1', fontName='BodyB', fontSize=27,
                         leading=31, textColor=NAVY, spaceBefore=12,
                         spaceAfter=17, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='BodyB', fontSize=17.2,
                         leading=21.0, textColor=NAVY, spaceBefore=15,
                         spaceAfter=8.5, keepWithNext=True),
    'h3': ParagraphStyle('h3', fontName='BodyB', fontSize=12.4,
                         leading=16, textColor=TEAL, spaceBefore=10,
                         spaceAfter=6, keepWithNext=True),
    'bullet': ParagraphStyle('bullet', fontName='Body', fontSize=10.5,
                             leading=14.5, textColor=INK, leftIndent=12,
                             firstLineIndent=0, bulletIndent=0,
                             spaceAfter=5.8, allowWidows=0, allowOrphans=0),
    'cell': ParagraphStyle('cell', fontName='Body', fontSize=9.1,
                           leading=12.3, textColor=INK, spaceAfter=0),
    'cell_head': ParagraphStyle('cell_head', fontName='BodyB', fontSize=9.2,
                                leading=12.3, textColor=colors.white),
}
STYLES['lead'] = ParagraphStyle('lead', parent=STYLES['body'], keepWithNext=True)


def normalize(text):
    return text.translate(str.maketrans({
        '\u2013': '-', '\u2014': ' - ', '\u2011': '-',
        '\u2010': '-', '\u2212': '-', '\u00a0': ' ',
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
    }))


def inline(text, code_size=9.1):
    stash = []

    def save(markup):
        key = f'ZZTOKEN{len(stash)}ZZ'
        stash.append(markup)
        return key

    text = re.sub(r'`([^`]+)`', lambda m: save(
        f'<font name="Code" size="{code_size}" color="#1E6372">'
        f'{html.escape(m.group(1))}</font>'), text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', lambda m: save(
        f'<a href="{html.escape(m.group(2), quote=True)}" color="#007F86">'
        f'<u>{html.escape(m.group(1))}</u></a>'), text)
    text = html.escape(text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    for i, value in enumerate(stash):
        text = text.replace(f'ZZTOKEN{i}ZZ', value)
    return text


class CodeBlock(Flowable):
    def __init__(self, lines, language='python', continued=False):
        super().__init__()
        self.lines = lines
        self.language = language
        self.continued = continued
        self.font_size = 8.75
        self.leading = 11.55
        self.spaceBefore = 4
        self.spaceAfter = 10

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        self.height = 27 + self.leading * len(self.lines)
        for line in self.lines:
            measured = pdfmetrics.stringWidth(line, 'Code', self.font_size)
            if measured > availWidth - 24:
                raise ValueError(f'Code line exceeds frame: {line}')
        return self.width, self.height

    def split(self, availWidth, availHeight):
        fitting = math.floor((availHeight - 27) / self.leading)
        if fitting >= len(self.lines):
            return [self]
        # Prefer whole short blocks, and never leave a tiny code fragment.
        if len(self.lines) <= 26 or fitting < 5:
            return []
        fitting = min(fitting, len(self.lines) - 3)
        return [CodeBlock(self.lines[:fitting], self.language, self.continued),
                CodeBlock(self.lines[fitting:], self.language, True)]

    def draw(self):
        c = self.canv
        c.setFillColor(PALE)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, 0, 2.3, self.height, fill=1, stroke=0)
        c.setFont('BodyB', 7.2)
        label = 'PYTHON' if self.language == 'python' else 'OUTPUT / NOTEBOOK'
        if self.continued:
            label += ' - CONTINUED'
        c.drawString(12, self.height - 12, label)
        y = self.height - 26
        c.setFont('Code', self.font_size)
        for line in self.lines:
            c.setFillColor(MUTED if line.lstrip().startswith('#') else INK)
            c.drawString(12, y, line)
            y -= self.leading


class Pipeline(Flowable):
    def wrap(self, aw, ah):
        self.width, self.height = aw, 95
        return self.width, self.height

    def draw(self):
        c = self.canv
        gap = 20
        box = (self.width - 2 * gap) / 3
        entries = [
            ('RAW IMAGES', '(N, 28, 28)', 'uint8 pixels'),
            ('MODEL INPUT', '(N, 784)', 'float32 / 255'),
            ('PREDICTIONS', '(N, 10)', 'one score per class'),
        ]
        for i, (label, shape, note) in enumerate(entries):
            x = i * (box + gap)
            c.setFillColor(PALE)
            c.roundRect(x, 17, box, 66, 5, fill=1, stroke=0)
            c.setFillColor(TEAL)
            c.setFont('BodyB', 8)
            c.drawCentredString(x + box / 2, 67, label)
            c.setFont('CodeB', 12)
            c.setFillColor(NAVY)
            c.drawCentredString(x + box / 2, 46, shape)
            c.setFont('Body', 9)
            c.setFillColor(MUTED)
            c.drawCentredString(x + box / 2, 30, note)
            if i < 2:
                c.setStrokeColor(TEAL)
                c.line(x + box + 4, 50, x + box + gap - 4, 50)
                c.line(x + box + gap - 8, 53, x + box + gap - 4, 50)
                c.line(x + box + gap - 8, 47, x + box + gap - 4, 50)
        c.setFont('BodyI', 8.6)
        c.drawString(0, 3, 'N is the number of examples. Track this axis through every step.')


class CourseDoc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, rightMargin=MARGIN,
                         leftMargin=MARGIN, topMargin=51, bottomMargin=48,
                         title='Python Crash Course - Chapters 2 and 3',
                         author='Prepared with Codex',
                         subject='Python companion to Deep Learning with Python, third edition',
                         pageCompression=1, allowSplitting=1)
        frame = Frame(MARGIN, 48, WIDTH, H - 99,
                      leftPadding=0, rightPadding=0, bottomPadding=0,
                      topPadding=0, id='normal')
        self.addPageTemplates(PageTemplate(id='course', frames=frame,
                                           onPage=self.draw_page,
                                           onPageEnd=self.draw_footer))
        self.current_part = ''
        self.heading_counter = 0
        self.map = []

    def beforeDocument(self):
        self.current_part = ''
        self.map = []

    def draw_page(self, canv, doc):
        if doc.page == 1:
            self.draw_cover(canv)
            return
        canv.saveState()
        canv.setStrokeColor(LINE)
        canv.line(MARGIN, H - 33, W - MARGIN, H - 33)
        canv.setFillColor(MUTED)
        canv.setFont('BodyB', 8.0)
        canv.drawString(MARGIN, H - 25, 'PYTHON CRASH COURSE')
        canv.setFont('Body', 8.0)
        canv.drawRightString(W - MARGIN, H - 25, 'CHAPTERS 2 + 3 / THIRD EDITION')
        canv.restoreState()

    def draw_footer(self, canv, doc):
        if doc.page == 1:
            return
        canv.saveState()
        canv.setStrokeColor(LINE)
        canv.line(MARGIN, 34, W - MARGIN, 34)
        canv.setFont('Body', 8.0)
        canv.setFillColor(MUTED)
        label = self.current_part or 'Study guide and contents'
        canv.drawString(MARGIN, 21, label[:85])
        canv.setFont('BodyB', 8.5)
        canv.drawRightString(W - MARGIN, 21, str(doc.page))
        canv.restoreState()

    def draw_cover(self, c):
        c.saveState()
        c.setFillColor(NAVY)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, H - 22, W, 22, fill=1, stroke=0)
        c.setFillColor(AMBER)
        c.rect(MARGIN, H - 112, 43, 5, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#A9DCDD'))
        c.setFont('BodyB', 11)
        c.drawString(MARGIN, H - 88, 'A PRACTICAL PYTHON COMPANION')
        c.setFillColor(colors.white)
        c.setFont('BodyB', 46)
        c.drawString(MARGIN, H - 182, 'Python')
        c.drawString(MARGIN, H - 233, 'crash course')
        c.setFont('Body', 19)
        c.drawString(MARGIN, H - 282, 'Learn through the code you already know.')
        c.setFillColor(colors.HexColor('#C1D1DC'))
        c.setFont('Body', 14)
        c.drawString(MARGIN, H - 330, 'Deep Learning with Python, third edition')
        c.drawString(MARGIN, H - 351, 'Chapters 2 and 3')
        c.setStrokeColor(colors.HexColor('#3F5E71'))
        c.line(MARGIN, 395, W - MARGIN, 395)
        cards = [
            ('01', 'Read the syntax', 'Names, calls, containers, and execution'),
            ('02', 'Trace the data', 'Shapes, slices, loops, and numerical operations'),
            ('03', 'Understand the objects', 'Classes, state, gradients, and framework APIs'),
            ('04', 'Write it yourself', 'Exercises, worked answers, and a complete lab'),
        ]
        for i, (n, title, desc) in enumerate(cards):
            y = 356 - 58 * i
            c.setFont('CodeB', 13)
            c.setFillColor(AMBER)
            c.drawString(MARGIN, y, n)
            c.setFillColor(colors.white)
            c.setFont('BodyB', 13)
            c.drawString(MARGIN + 39, y, title)
            c.setFont('Body', 10.5)
            c.setFillColor(colors.HexColor('#C1D1DC'))
            c.drawString(MARGIN + 39, y - 17, desc)
        c.setFont('Body', 9)
        c.drawString(MARGIN, 58, 'Personal study edition | 26 October 2026')
        c.drawString(MARGIN, 43, 'Based on the third-edition companion notebooks in your workspace.')
        c.restoreState()

    def afterFlowable(self, flowable):
        if not hasattr(flowable, '_heading'):
            return
        level, text, key = flowable._heading
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=(level == 0))
        self.notify('TOCEntry', (level, text, self.page, key))
        self.map.append({'title': text, 'page': self.page, 'level': level})
        if level == 0:
            self.current_part = text.replace(' | ', ' / ')


def table_from_lines(lines):
    rows = []
    for idx, line in enumerate(lines):
        cells = [s.strip() for s in line.strip().strip('|').split('|')]
        if all(re.match(r'^:?-+:?$', x) for x in cells):
            continue
        head = len(rows) == 0
        style = STYLES['cell_head'] if head else STYLES['cell']
        rows.append([Paragraph(inline(x, 8.2), style) for x in cells])
    n = len(rows[0])
    if n == 2:
        widths = [WIDTH * .38, WIDTH * .62]
    elif n == 3:
        widths = [WIDTH * .23, WIDTH * .36, WIDTH * .41]
    else:
        widths = [WIDTH * .17] + [WIDTH * .83 / (n - 1)] * (n - 1)
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT',
                  spaceBefore=4, spaceAfter=10)
    commands = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, 0), .5, NAVY),
        ('LINEBELOW', (0, 1), (-1, -1), .35, LINE),
    ]
    for i in range(1, len(rows)):
        if i % 2:
            commands.append(('BACKGROUND', (0, i), (-1, i), PALE))
    table.setStyle(TableStyle(commands))
    return table


def prepare_sources():
    texts = []
    for path in sorted(SRC.glob('0[1-5]_*.md')):
        text = normalize(path.read_text(encoding='utf-8'))
        if path.name.startswith('03_'):
            text = re.sub(r'\bsection (\d+)\b',
                          lambda m: 'section ' + str(int(m[1]) - 3), text)
        texts.append((path.name, text))
    return texts


def make_story(texts):
    story = [Spacer(1, 1), PageBreak()]
    story.append(Paragraph('Your course map', STYLES['h1']))
    story.append(Paragraph(
        'Work from the foundations to the final lab, or use this clickable map '
        'to revisit a particular Python pattern. Exercise solutions follow '
        'each chapter-focused part.', STYLES['body']))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle('toc0', fontName='BodyB', fontSize=11.3, leading=15,
                       leftIndent=0, firstLineIndent=0, spaceBefore=10,
                       textColor=NAVY),
        ParagraphStyle('toc1', fontName='Body', fontSize=9.9, leading=13.2,
                       leftIndent=12, firstLineIndent=0, spaceBefore=2.7,
                       textColor=INK),
    ]
    toc.dotsMinLevel = 1
    story.append(toc)
    section_num = 0
    bookmark = 0
    code_blocks = []
    for filename, text in texts:
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith('```'):
                language = line[3:].strip()
                block = []
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    block.append(lines[i])
                    i += 1
                if language == 'python':
                    ast.parse('\n'.join(block), filename=f'{filename}#{len(code_blocks)+1}')
                    code_blocks.append({'source': filename, 'code': '\n'.join(block)})
                story.append(CodeBlock(block, language))
                i += 1
                continue
            if line == '[[PIPELINE]]':
                story.append(Pipeline())
                i += 1
                continue
            if line.startswith('|'):
                block = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    block.append(lines[i])
                    i += 1
                story.append(table_from_lines(block))
                continue
            if line.startswith('# '):
                story.append(PageBreak())
                title = line[2:]
                p = Paragraph(inline(title), STYLES['h1'])
                p._heading = (0, title, f'heading{bookmark}')
                bookmark += 1
                story.append(p)
                i += 1
                continue
            if line.startswith('## '):
                section_num += 1
                title = re.sub(r'^\d+\.\s*', '', line[3:])
                title = f'{section_num}. {title}'
                p = Paragraph(inline(title, 14.6), STYLES['h2'])
                p._heading = (1, title.replace('`', ''), f'heading{bookmark}')
                bookmark += 1
                story.append(p)
                i += 1
                continue
            if line.startswith('### '):
                story.append(Paragraph(inline(line[4:]), STYLES['h3']))
                i += 1
                continue
            match = re.match(r'^(\d+\.|-)\s+(.*)$', line)
            if match:
                bullet = '\u2022' if match[1] == '-' else match[1]
                story.append(Paragraph(inline(match[2]), STYLES['bullet'],
                                       bulletText=bullet))
                i += 1
                continue
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                nxt = lines[i].strip()
                if nxt.startswith(('#', '```', '|', '[[', '- ')) or re.match(r'^\d+\. ', nxt):
                    break
                para.append(nxt)
                i += 1
            paragraph = ' '.join(para)
            is_source = paragraph.startswith(('Source:', 'Notebook locator:', 'Exact excerpt'))
            style = STYLES['small'] if is_source else (
                STYLES['lead'] if paragraph.endswith(':') else STYLES['body'])
            story.append(Paragraph(inline(paragraph), style))
    (SRC / 'all_code_blocks.json').write_text(json.dumps(code_blocks, indent=2), encoding='utf-8')
    print(f'Syntax checked {len(code_blocks)} Python blocks across {section_num} sections')
    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    texts = prepare_sources()
    doc = CourseDoc(OUT)
    doc.multiBuild(make_story(texts), maxPasses=4)
    (SRC / 'page_map.json').write_text(json.dumps(doc.map, indent=2), encoding='utf-8')
    pdf = PdfReader(OUT)
    extracted = '\n'.join(p.extract_text() for p in pdf.pages)
    (SRC / 'extracted_text.txt').write_text(extracted, encoding='utf-8')
    summary = {
        'file': str(OUT), 'pages': len(pdf.pages), 'bytes': OUT.stat().st_size,
        'source_words': sum(len(t.split()) for _, t in texts),
        'python_blocks': len(json.loads((SRC / 'all_code_blocks.json').read_text())),
        'bookmarks': len(doc.map),
    }
    (SRC / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
