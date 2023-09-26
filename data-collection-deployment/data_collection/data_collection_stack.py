from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_ecr_assets as ecr_assets
)
from constructs import Construct


class DataCollectionStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create an S3 bucket
        bucket = s3.Bucket(self, "OpenBBBucket",
                           versioned=True, bucket_name='fincontext-data',
                           removal_policy=RemovalPolicy.DESTROY)

        # Create a Docker image asset from the Dockerfile
        obb_docker_image = ecr_assets.DockerImageAsset(
            self,
            "OpenBBLambdaImage",
            directory=".\lambdas\commander"
        )

        article_text_docker_image = ecr_assets.DockerImageAsset(
            self,
            "ArticleTextLambdaImage",
            directory=".\lambdas\drone"
        )

        article_text_lambda_function = _lambda.DockerImageFunction(
            self,
            "ArticleTextLambdaFunction",
            code=_lambda.DockerImageCode.from_ecr(
                article_text_docker_image.repository,
                tag_or_digest=article_text_docker_image.image_tag
            ))

        # Create a Lambda function with the Docker image
        obb_lambda_function = _lambda.DockerImageFunction(
            self,
            "OpenBBLambdaFunction",
            code=_lambda.DockerImageCode.from_ecr(
                obb_docker_image.repository,
                tag_or_digest=obb_docker_image.image_tag
            ),
            environment={
                'DRONE_FUNC_NAME': article_text_lambda_function.function_name,
                'FINCONTEXT_BUCKET': bucket.bucket_name,
            }
        )
