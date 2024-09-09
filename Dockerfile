FROM continuumio/anaconda3
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
WORKDIR /dfaulo
COPY . /dfaulo/

RUN /bin/bash -c "conda create -n dfaulo_env python=3.10.14 -y"
# 激活环境并安装PyTorch和其他依赖
RUN /bin/bash -c "source activate dfaulo_env \
    && conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y \
    && pip install -r docker_requirements.txt"

# 设置环境变量以确保conda环境被激活
ENV PATH /opt/conda/envs/dfaulo_env/bin:$PATH

