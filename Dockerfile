FROM pnnlmiscscripts/anaconda
WORKDIR /dfaulo
COPY . /dfaulo/
RUN conda create -n dfaulo_env python=3.10.14 -y
# 激活环境并安装PyTorch和其他依赖
RUN /bin/bash -c "source activate dfaulo_env \
    && conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y \
    && pip install -r docker_requirements.txt"

# 设置环境变量以确保conda环境被激活
ENV PATH /opt/conda/envs/dfaulo_env/bin:$PATH

