"""Build the five-page report from its editable JSON manuscript."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[1];source=json.loads((ROOT/'report/report_source.json').read_text());styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyReport',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=8));styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=8,leading=10));story=[];seen=[]
for pi,page in enumerate(source['pages']):
 if pi:story.append(PageBreak())
 for item in page:
  if item['type']=='paragraph':story.append(Paragraph(escape(item['text']),styles['BodyReport']))
  elif item['type']=='heading':story.extend([Paragraph(escape(item['text']),styles['Heading2']),Spacer(1,5)])
  else:
   rows=[[Paragraph(escape(str(c)),styles['Cell']) for c in row] for row in item['rows']];n=len(rows[0]);widths={2:[190,315],3:[235,135,135],4:[110,225,85,85]}[n]
   t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#72869b')),('LINEBELOW',(0,1),(-1,-1),.25,colors.lightgrey),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,10)])
def footer(c,d):
 seen.append(d.page);c.saveState();c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#52616b'));c.drawString(45,25,'DASE7506 MP2 | 3036837930');c.drawRightString(A4[0]-45,25,str(d.page));c.restoreState()
doc=SimpleDocTemplate(str(ROOT/'report/report.pdf'),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=35,bottomMargin=40,title=source['title'],author='3036837930');doc.build(story,onFirstPage=footer,onLaterPages=footer);assert max(seen)<=10
(ROOT/'results/report_build.json').write_text(json.dumps(dict(pages=max(seen),page_limit=10,renderer='ReportLab 4.4.10'),indent=2)+'\n');print('Report pages:',max(seen))
