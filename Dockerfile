ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


COPY requirements.txt requirements.txt
COPY sources.list /etc/apt/sources.list
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN apt-get update -y
RUN apt install curl -y
RUN ln -sf /bin/bash /bin/sh
## 把当前文件夹里的文件构建到镜像的//workspace目录下,并设置为默认工作目录
ADD ./code ./code
ADD ./user_data ./user_data
ADD code/run.sh run.sh
WORKDIR ./code


CMD ["/bin/bash", "run.sh"]
