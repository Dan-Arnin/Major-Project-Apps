import requests
from bs4 import BeautifulSoup
import arxiv, os

os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY

def replace_to_html(link_string):
  values = ["abs", "pdf"]
  for i in values:
    if i in link_string:
      new_link = link_string.replace(i, "html", 1)
  return new_link
  
def extract_text(url):
  response = requests.get(url)

  if response.status_code == 200:
    html_content = response.text
  else:
    print(url)
    print(f"Error: Failed to retrieve website content. Status code: {response.status_code}")
    return ""

  soup = BeautifulSoup(html_content, "lxml")

  text = soup.get_text()
  return text


def arxiv_papers(query_):
  client = arxiv.Client()
  search = arxiv.Search(
  query = query_,
  max_results = 20,
  sort_by = arxiv.SortCriterion.SubmittedDate
  )

  results = client.results(search)
  print()
  all_results = list(results)
  paper_data = []
  for index, paper in enumerate(all_results):
    temp_dict = {}
    temp_dict["link"] = str(paper.entry_id)
    temp_dict["authors"] = ", ".join(list(map(str, list(author for author in paper.authors))))
    temp_dict["title"] = paper.title
    temp_dict["summary"] = paper.summary
    contents = extract_text(replace_to_html(temp_dict["link"]))
    if contents:
        temp_dict["content"] = contents
        paper_data.append(temp_dict)
  if len(paper_data) >= 10:
    return paper_data[:10]
  return paper_data