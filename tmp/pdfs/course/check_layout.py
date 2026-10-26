from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
import pdfplumber
import json

root = Path(__file__).resolve().parent
pdf_path = root.parents[2] / 'output/pdf/python_crash_course_chapters_2_and_3.pdf'
issues = []
page_stats = []
with pdfplumber.open(pdf_path) as pdf:
    for num, page in enumerate(pdf.pages, 1):
        chars = page.chars
        content = [c for c in chars if 43 < c['top'] < page.height - 40]
        outside = [c for c in chars if c['x0'] < -1 or c['x1'] > page.width + 1
                   or c['top'] < -1 or c['bottom'] > page.height + 1]
        if outside:
            issues.append({'page': num, 'outside': ''.join(c['text'] for c in outside)})
        if num > 1:
            body_outside = [c for c in content if c['x0'] < 47 or c['x1'] > page.width - 47]
            if body_outside:
                issues.append({'page': num, 'margin': ''.join(c['text'] for c in body_outside)})
        page_stats.append({'page': num, 'chars': len(content),
                           'body_bottom': max((c['bottom'] for c in content), default=0)})

images = sorted(root.glob('page-*.png'))
font = ImageFont.truetype('C:/Windows/Fonts/calibri.ttf', 18)
for start in range(0, len(images), 12):
    batch = images[start:start + 12]
    sheet = Image.new('RGB', (1200, 1730), '#DDE5EA')
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(batch):
        im = Image.open(path).convert('RGB')
        im.thumbnail((286, 402))
        x = 10 + (i % 4) * 300
        y = 10 + (i // 4) * 568
        sheet.paste(im, (x, y))
        draw.text((x + 5, y + 410), f'Page {start + i + 1}', fill='#173247', font=font)
    sheet.save(root / f'contact-{start // 12 + 1}.png')

result = {'issues': issues, 'page_stats': page_stats, 'rendered_pages': len(images)}
(root / 'layout_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'issues': issues, 'rendered_pages': len(images)}, indent=2))
