import requests
from bs4 import BeautifulSoup
import io, base64
from pypdf import PdfReader
from reportlab.pdfgen import canvas 
from reportlab.pdfbase.ttfonts import TTFont 
from reportlab.pdfbase import pdfmetrics 
from reportlab.lib import colors 
from youtube_transcript_api import YouTubeTranscriptApi
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

def create_pdf(pdf_string):
   pdf_bytes = io.BytesIO()
   doc = SimpleDocTemplate(pdf_bytes, pagesize=letter)
   elements = []
   text = pdf_string
   styles = getSampleStyleSheet()
   style = styles['BodyText']
   paragraph = Paragraph(text, style)
   elements.append(paragraph)
   doc.build(elements)
   pdf_data = pdf_bytes.getvalue()
   return pdf_data

def extract_transcript(video_id):
    video_id = video_id[video_id.find('=')+1:]
    print(video_id)
    text = ""
    val = YouTubeTranscriptApi.get_transcript(video_id)
    for i in val:
        text += i["text"]
    return text

def extract_data_from_pdf(base64_pdf_string):
    pdf_file = base64.b64decode(base64_pdf_string)
    file = io.BytesIO(pdf_file)
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text_temp = page.extract_text()
        text += text_temp
    return text

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    return text

def split_into_chunks(text, chunk_size=3000):
  chunks = []
  start = 0
  end = chunk_size
  while start < len(text):
    chunk = text[start:end]
    chunks.append(chunk)
    start = end
    end += chunk_size

  # Handle the last chunk if it's less than chunk_size
  if end > len(text):
    chunks.append(text[start:])

  return chunks
