FROM ubuntu:18.04
ARG USER_ID
ARG GROUP_ID

ARG git_owner="singnet"
ARG git_repo="phi"
ARG git_branch="master"
ARG snetd_version

ENV SERVICE_NAME=tononi-phi
ENV SINGNET_DIR=/opt/${git_owner}
ENV SINGNET_REPOS=/opt/singnet
ENV REPO_NAME=${git_repo}
ENV PROJECT_DIR=/opt/${git_owner}/${git_repo}
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH "${PYTONPATH}:${PROJECT_DIR}/src"
ENV LD_LIBRARY_PATH /usr/local/lib

RUN mkdir -p ${PROJECT_DIR}

RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    vim \
    wget \
    nano \
    curl

RUN cd /tmp && \
    apt update && \
    apt-get install build-essential

RUN python3 -m pip install -U pip

RUN SNETD_GIT_VERSION=`curl -s https://api.github.com/repos/singnet/snet-daemon/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")' || echo "v5.0.1"` && \
    SNETD_VERSION=${snetd_version:-${SNETD_GIT_VERSION}} && \
    cd /tmp && \
    wget https://github.com/singnet/snet-daemon/releases/download/${SNETD_VERSION}/snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    tar -xvf snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    mv snet-daemon-${SNETD_VERSION}-linux-amd64/snetd /usr/bin/snetd && \
    rm -rf snet-daemon-*

RUN cd ${SINGNET_REPOS} && \
    git clone -b ${git_branch} https://github.com/${git_owner}/${REPO_NAME}.git

RUN cd ${PROJECT_DIR}/snet-service/ && \
    python3 -m pip install -U pip==20.2.4 && \
    python3 -m pip install -r requirements.txt && \
    sh buildproto.sh

RUN cd ${PROJECT_DIR} && \
    python3 -m pip install -r requirements.txt && \

WORKDIR ${PROJECT_DIR}/snet-service/
