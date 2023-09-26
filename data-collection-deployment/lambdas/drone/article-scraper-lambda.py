import json
from newspaper import Article


def lambda_handler(event, context):
    url = event['url']
    print(url)

    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        print(text)
    except:
        text = ''

    return {
        'statusCode': 200,
        'text': text
    }
