# Use the Amazon Linux 2 as the base image
FROM public.ecr.aws/lambda/python:3.10

# Set the working directory
WORKDIR /var/task

RUN python -V

RUN pip install newspaper3k

# Copy your Lambda function code
COPY article-scraper-lambda.py .

# Set the CMD to your handler (assuming the function name is 'lambda_handler')
CMD ["article-scraper-lambda.lambda_handler"]
