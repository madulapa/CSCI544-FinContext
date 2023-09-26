#!/bin/sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
  exec /usr/bin/aws-lambda-rie python -m awslambdaric $@
else
  # exec /root/miniconda3/bin/activate base && python -m awslambdaric $@

  exec sudo /root/miniconda3/bin/python -m awslambdaric $@
fi     