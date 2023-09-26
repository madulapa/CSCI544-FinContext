from io import StringIO
import json
import boto3
import pandas as pd
import swifter
from openbb_terminal.sdk import openbb
import os

drone_func_name = os.environ['DRON_FUNC_NAME']
bucket = os.environ['FINCONTEXT_BUCKET']
s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')


def lambda_handler(event, context):
    ticker = event['ticker']
    news = openbb.stocks.news(ticker)
    news_text = pd.DataFrame(news[['articleHeadline', 'articleURL']])

    def invoke_retrieve_article_text(url):
        payload = json.dumps({"url": url})
        response = lambda_client.invoke(
            FunctionName=drone_func_name,
            InvocationType='RequestResponse',
            Payload=payload
        )
        response_payload = json.loads(
            response['Payload'].read().decode("utf-8"))
        return response_payload['text']

    news_text['articleText'] = news_text['articleURL'].swifter.apply(
        invoke_retrieve_article_text)

    # Save DataFrame to CSV and upload to S3
    csv_buffer = StringIO()
    news_text.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key='stock_news.csv',
                  Body=csv_buffer.getvalue())

    return {
        'statusCode': 200,
        'body': json.dumps('Stock news saved to CSV in S3!')
    }
