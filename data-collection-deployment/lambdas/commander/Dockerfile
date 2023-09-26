# Use Amazon Linux 2023 as the base image
FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Install basic utilities and AWS Lambda Runtime Interface Client (RIC)
RUN dnf -y install \
    findutils \
    && dnf clean all \
    && curl -sSL https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie -o /usr/bin/aws-lambda-rie \
    && chmod +x /usr/bin/aws-lambda-rie

# Update system and install required packages
RUN dnf update -y && \
    dnf upgrade -y && \
    dnf install git which gcc gcc-c++ openssl-devel -y

# Install Rust toolchain
RUN curl https://sh.rustup.rs -sSf | \
    sh -s -- --default-toolchain stable -y

ENV PATH=/root/.cargo/bin:$PATH

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/root/miniconda3/bin:$PATH

RUN conda init bash && \
    conda --version && \
    pip install boto3 awslambdaric

# Set the working directory
WORKDIR /var/task

# Clone OpenBBTerminal repo and set up the Conda environment
# RUN git clone https://github.com/OpenBB-finance/OpenBBTerminal.git ~/OpenBBTerminal && \
#     conda env update -n base --file ~/OpenBBTerminal/build/conda/conda-3-10-env.yaml && \
#     ~/OpenBBTerminal/build/conda/cleanup_artifacts.sh && \
#     conda install swifter openbb -c conda-forge

RUN conda env update -n base --file https://raw.githubusercontent.com/OpenBB-finance/OpenBBTerminal/develop/build/conda/conda-3-10-env.yaml && \
    pip install openbb && \
    conda install swifter -c conda-forge

# Copy your Lambda function code and entrypoint script
COPY openbb-lambda.py .
COPY entrypoint.sh .

# Set the entrypoint
# ENTRYPOINT [ "./entrypoint.sh" ]
ENTRYPOINT [ "python -m awslambdaric" ]

# Set the CMD to your handler (assuming the function name is 'lambda_handler')
CMD ["openbb-lambda.lambda_handler"]
