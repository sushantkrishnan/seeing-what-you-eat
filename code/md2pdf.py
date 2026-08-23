#!/usr/bin/env python3
"""Render a talk script markdown file to a podium-readable PDF."""
import re, sys, html
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether, HRFlowable, PageBreak)

SRC, OUT, TITLE, SUBTITLE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

INK   = colors.HexColor('#1a1a1a')
GREY  = colors.HexColor('#5f5f5f')
RULE  = colors.HexColor('#d5d5d5')
SPEAK = colors.HexColor('#111111')
BAR   = colors.HexColor('#2f6f4f')
KEYC  = colors.HexColor('#a8330f')
BG    = colors.HexColor('#f4f2ed')

def esc(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'`([^`]+)`', r'<font face="Courier" size="8.5">\1</font>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', t)
    for a, b in (('\u26a0\ufe0f', '<b>!</b> '), ('\u26a0', '<b>!</b> '),
                 ('\u2705', '<b>[done]</b> '), ('\U0001f51c', '<b>[to do]</b> '),
                 ('\u2192', '&#8594;'), ('\u00d7', '&#215;')):
        t = t.replace(a, b)
    return t

S = {
 'h1':   ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=19, leading=23, textColor=INK, spaceAfter=2),
 'sub':  ParagraphStyle('sub', fontName='Helvetica', fontSize=10, leading=14, textColor=GREY, spaceAfter=10),
 'sec':  ParagraphStyle('sec', fontName='Helvetica-Bold', fontSize=15.5, leading=19,
                        textColor=INK, spaceBefore=2, spaceAfter=8),
 'h2':   ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12.5, leading=15, textColor=INK,
                        spaceBefore=12, spaceAfter=5),
 'must': ParagraphStyle('must', fontName='Helvetica-Oblique', fontSize=9, leading=12.5,
                        textColor=BAR, spaceAfter=6),
 'speak':ParagraphStyle('speak', fontName='Times-Roman', fontSize=12, leading=16.5, textColor=SPEAK,
                        leftIndent=9, spaceAfter=7),
 'dir':  ParagraphStyle('dir', fontName='Helvetica', fontSize=8.6, leading=12, textColor=GREY,
                        spaceAfter=6),
 'beat': ParagraphStyle('beat', fontName='Helvetica-Bold', fontSize=8, leading=11, textColor=BAR,
                        spaceBefore=1, spaceAfter=7, leftIndent=9),
 'q':    ParagraphStyle('q', fontName='Helvetica-Bold', fontSize=9.2, leading=12.5,
                        textColor=INK, spaceBefore=6, spaceAfter=3),
 'li':   ParagraphStyle('li', fontName='Helvetica', fontSize=8.8, leading=12.2, textColor=INK,
                        leftIndent=11, bulletIndent=2, spaceAfter=3.5),
 'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8, leading=10.4, textColor=INK),
 'cellh':ParagraphStyle('cellh', fontName='Helvetica-Bold', fontSize=8, leading=10.4, textColor=INK),
}

W = A4[0] - 34*mm

def table(rows):
    head, body = rows[0], rows[1:]
    ncol = len(head)
    data = [[Paragraph(esc(c), S['cellh']) for c in head]] + \
           [[Paragraph(esc(c), S['cell']) for c in r] for r in body]
    # proportional to content length, but never narrower than the longest
    # unbreakable token in the column (filenames in code font were wrapping mid-word)
    lens = [max(len(r[i]) for r in rows) for i in range(ncol)]
    def longest_token(i):
        best = 0
        for r in rows:
            for tok in re.split(r'[ \u2014]', r[i].replace('`','')):
                best = max(best, len(tok))
        return best
    floors = [longest_token(i)*5.35 + 14 for i in range(ncol)]
    tot = sum(lens) or 1
    widths = [max(floors[i], W*lens[i]/tot) for i in range(ncol)]
    if sum(widths) > W:                       # shrink the roomiest columns first
        excess = sum(widths) - W
        slack = [max(0.0, widths[i]-floors[i]) for i in range(ncol)]
        if sum(slack) > 0:
            widths = [widths[i] - excess*slack[i]/sum(slack) for i in range(ncol)]
    else:
        scale = W/sum(widths); widths = [w*scale for w in widths]
    t = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),3.5), ('BOTTOMPADDING',(0,0),(-1,-1),3.5),
        ('LEFTPADDING',(0,0),(-1,-1),5), ('RIGHTPADDING',(0,0),(-1,-1),5),
        ('LINEBELOW',(0,0),(-1,0),0.7,INK),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, BG]),
        ('LINEBELOW',(0,1),(-1,-2),0.25,RULE),
    ]))
    return t

def flush_para(buf, kind, story):
    if not buf: return
    text = ' '.join(buf).strip()
    if not text: return
    if kind == 'speak':
        story.append(Paragraph(esc(text), S['speak']))
    else:
        story.append(Paragraph(esc(text), S['dir']))

src = open(SRC).read().split('\n')
story, buf, kind, tbl = [], [], 'dir', []
seen_title = False
just_sectioned = False
story.append(Paragraph(esc(TITLE), S['h1']))
story.append(Paragraph(esc(SUBTITLE), S['sub']))

i = 0
while i < len(src):
    ln = src[i].rstrip()
    # table block
    if ln.startswith('|'):
        rows = []
        while i < len(src) and src[i].startswith('|'):
            cells = [c.strip() for c in src[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
                rows.append(cells)
            i += 1
        flush_para(buf, kind, story); buf = []
        story.append(Spacer(1, 3)); story.append(table(rows)); story.append(Spacer(1, 8))
        continue
    if ln.startswith('# '):
        # the first one is the document title, which the caller supplies; later ones are
        # real section headings and must not be swallowed
        flush_para(buf, kind, story); buf = []
        if not seen_title:
            seen_title = True
        else:
            story.append(PageBreak())
            story.append(Paragraph(esc(ln[2:]), S['sec']))
            story.append(HRFlowable(width='100%', thickness=1.1, color=INK, spaceAfter=8))
            just_sectioned = True
        i += 1; continue
    if ln.startswith('## '):
        flush_para(buf, kind, story); buf = []
        head = ln[3:]
        col = KEYC if 'KEY' in head else INK
        block = []
        if not just_sectioned:                 # avoid a double rule under a section heading
            block.append(HRFlowable(width='100%', thickness=0.6, color=RULE,
                                    spaceBefore=10, spaceAfter=4))
        just_sectioned = False
        block.append(Paragraph(esc(head), ParagraphStyle('x', parent=S['h2'], textColor=col)))
        # keep the heading with what follows it, so no section header is stranded
        j, taken = i+1, 0
        while j < len(src) and taken < 2:
            nxt = src[j].rstrip()
            if nxt.startswith('**Must land:'):
                block.append(Paragraph(esc(nxt), S['must'])); taken += 1; i = j
            elif nxt.startswith('>'):
                para = []
                while j < len(src) and src[j].startswith('>') and src[j][1:].strip():
                    para.append(src[j][1:].strip()); j += 1
                block.append(Paragraph(esc(' '.join(para)), S['speak'])); taken = 2; i = j-1
            elif nxt.strip() == '':
                j += 1; continue
            else:
                break
            j += 1
        story.append(KeepTogether(block))
        i += 1; continue
    if ln.startswith('---'):
        i += 1; continue
    if re.match(r'^\s*`\[(beat|stretch)\]', ln):
        flush_para(buf, kind, story); buf = []
        m = re.match(r'^\s*`\[(beat|stretch)\]`\s*(.*)', ln)
        tag, rest = m.group(1).upper(), m.group(2)
        label = f'[ {tag} ]' + (f'  {rest}' if rest else '')
        story.append(Paragraph(esc(label), S['beat']))
        i += 1; continue
    m_li = re.match(r'^(\s*)(?:- |(\d+)\. )(.*)', ln)
    if m_li and (ln.startswith('- ') or ln.startswith('  - ') or ln.startswith('   - ')
                 or re.match(r'^\d+\. ', ln)):
        flush_para(buf, kind, story); buf = []
        indent, num, item = m_li.group(1), m_li.group(2), m_li.group(3)
        nested = len(indent) >= 2
        # wrapped continuation lines are indented further and are NOT list items themselves
        while i+1 < len(src):
            nxt = src[i+1]
            if not nxt.strip() or not nxt.startswith('  '): break
            if re.match(r'^\s*(?:- |\d+\. )', nxt): break
            i += 1; item += ' ' + src[i].strip()
        st = ParagraphStyle('li2', parent=S['li'],
                            leftIndent=S['li'].leftIndent + (14 if nested else 0),
                            bulletIndent=S['li'].bulletIndent + (12 if nested else 0),
                            spaceAfter=3 if nested else S['li'].spaceAfter)
        story.append(Paragraph(esc(item), st, bulletText=(f'{num}.' if num else ('–' if nested else '•'))))
        i += 1; continue
    if ln.startswith('>'):
        body = ln[1:].strip()
        if kind != 'speak':
            flush_para(buf, kind, story); buf = []; kind = 'speak'
        if body == '':
            flush_para(buf, kind, story); buf = []
        else:
            buf.append(body)
        i += 1; continue
    if ln.strip() == '':
        flush_para(buf, kind, story); buf = []; kind = 'dir'
        i += 1; continue
    if kind == 'speak':
        flush_para(buf, kind, story); buf = []; kind = 'dir'
    # "Must land:" lines get their own accent style
    if ln.startswith('**Q:'):
        flush_para(buf, kind, story); buf = []
        story.append(Paragraph(esc(ln), S['q'])); i += 1; continue
    if ln.startswith('**Must land:'):
        flush_para(buf, kind, story); buf = []
        story.append(Paragraph(esc(ln), S['must'])); i += 1; continue
    buf.append(ln.strip()); i += 1
flush_para(buf, kind, story)

def deco(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5); canvas.setFillColor(GREY)
    canvas.drawString(17*mm, 12*mm, SRC.split('/')[-1])
    canvas.drawRightString(A4[0]-17*mm, 12*mm, f'{doc.page}')
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.4)
    canvas.line(17*mm, 15*mm, A4[0]-17*mm, 15*mm)
    canvas.restoreState()

doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=17*mm, rightMargin=17*mm,
                      topMargin=16*mm, bottomMargin=20*mm,
                      title=TITLE, author='COMPSCI 760 Group 1')
doc.addPageTemplates([PageTemplate(id='p',
    frames=[Frame(17*mm, 20*mm, W, A4[1]-36*mm, id='f')], onPage=deco)])
doc.build(story)
print('wrote', OUT)
