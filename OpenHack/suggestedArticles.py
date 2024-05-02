from fetchExtractArticles import fetch_articles
from keywordExtraction import get_key_words


def get_suggestions(user_query, summary):
    keywords = get_key_words(summary.replace("\n", ". "))

    urls = {}
    for key, score in keywords:
        urls[user_query + " " + key] = fetch_articles(user_query + " " + key, 0, 1)

    # for key in urls:
    # print(key, urls[key])
    return urls
