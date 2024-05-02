# import certifi
from googlesearch import search
from goose3 import Goose


def extract_text(article_url):
    g = Goose()
    try:
        article = g.extract(article_url)
        if len(article.cleaned_text) > 2000 and len(article.cleaned_text) < 15000:
            return {
                "content": article.cleaned_text,
                "title": article.title,
                "url": article_url,
            }
    except Exception as e:
        pass
        # print(f"Error extracting text from article: {e}")
    # return {"content": "", "title": "","url":""}


def fetch_articles(user_query, start=1, stop=2):
    articles = []
    try:
        for url in search(user_query, start=start, stop=stop):
            articles.append(url)
    except Exception as e:
        pass
        # print(f"Error fetching articles: {e}")

    return articles
