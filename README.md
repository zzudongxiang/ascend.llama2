# Llama2

## 1. 创建docker

### 1.1 常见的docker命令

1. 显示全部的镜像：`docker images`
2. 创建一个容器：`docker run ...`<查看1.2节内容>
3. 显示全部的容器：`docker ps -a`
4. 启动一个容器：`docker start <id>`
5. 停止一个容器：`docker stop <id>`
6. 进入一个正在运行的程序：`docker exec -it <id> /bin/bash`
7. 删除一个容器：`docker rm <id>`

### 1.2 创建一个docker容器

创建一个docker容器的时候需要同时挂载本地工作目录和相关的程序驱动，因此创建docker容器的命令参数较多，以下为一个创建docker容器命令的基本格式：

```bash
docker run -it --ipc=host --name <name> -v <local_path>:<docker_path>  \
    --workdir=<docker_path> \
    --pids-limit 409600 \
    --privileged --network=host \
    --shm-size=128G \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/bashrc:/etc/bashrc \
    -p 223:223 \
    -u root \
    <image_id> /bin/bash
```

- `-it`：已交互方式运行容器，并分配一个终端
- `-ipc=host`：用于docker内部进程与宿主机进程的通信
- `--name` <name>：指定要创建的容器的名字
- `-v <local_path>:<docker_path>`：挂载本地的`<local_path>`路径到容器中的`<docker_path>`路径
- `--workdir=<docker_path>`：重定向容器的工作路径到`<docker_path>`路径
- `--pids-limit 409600`：<**可省略**>对容器内可以运行的进程数量限制
- `--privileged`：使容器内的root具有容器外部root的权限
- `--network=host`：~~<删除>指定容器的网络与宿主机相同~~
- `--shm-size=128G`：<**可省略**>修改共享内存的大小
- `--device=<device>`：挂载硬件设备
- `-v /usr/local/Ascend/driver:/usr/local/Ascend/driver`：[**NPU**]挂载NPU驱动
- `-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware`：[**NPU**]挂载Firmware
- `-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/`：[**NPU**]挂载NPU工具包
- `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi`：[**NPU**]挂载npu-smi命令
- `-v /usr/local/dcmi:/usr/local/dcmi`：[**NPU**]挂载DCMI
- `-v /etc/ascend_install.info:/etc/ascend_install.info`：[**NPU**]挂载安装信息
- `-v /etc/bashrc:/etc/bashrc`：添加bash命令行的内容
- `-p 223:223`：挂载端口，用于使用ssh直接登录docker容器内部
- `-u root`：以root身份进入容器
- `<image_id>`：要创建容器的镜像ID，可通过`docker images`查看
- `/bin/bash`：进入容器后执行的命令，默认进入控制台

> [!TIP]
>
> 在Ascend上使用的镜像信息为：`ascendhub.huawei.com/public-ascendhub/all-in-one`

完成容器创建后可以通过`npu-smi info`命令查看NPU状态是否正常，若NPU正常挂载，则会输出NPU的数量、状态等基本信息。

### 1.3 真实的创建命令

```bash
docker run -it --ipc=host --name zhangdx-test -v ~:/root  \
    --workdir=/root \
    --privileged \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/bashrc:/etc/bashrc \
    -u root \
    -p 2234:2234 \
    47f84079ea47 /bin/bash
```

## 2. 配置SSH登录

为了使用外部IDE（VSCode或MindStudio等）连接Docker内的编译环境，需要使用SSH将容器内的操作系统通过SSH映射到外部

### 2.1 安装OpenSSH-Server

需要安装SSH Server，并修改管理员密码，或配置SSH-Key登录的方式登录到Docker内部

```bash
# 更新apt
apt update
# 安装SSH-Server和Nano编辑器
apt install -y nano openssh-server
```

### 2.2 修改SSH的配置

使用如下命令修改Docker内部root用户的密码

```bash
passwd
# 修改成功后会有如下显示
# passwd: password updated successfully
```

使用nano（也可以使用vi）修改ssh的配置信息

```bash
nano /etc/ssh/sshd_config
```

- 新增`Port 223`，这里的端口号应该与创建容器时的一致
- 新增`PermitRootLogin yes`
- 新增`PubkeyAuthentication yes`

然后使用如下命令启动ssh并查看ssh的状态

```bash
# 启动ssh
service ssh start
# 查看ssh的状态
service ssh status
```

### 2.3 添加SSH密钥

为了方便后续使用IDE连接docker，建议配置SSH密钥以实现免密登录，使用`ssh-keygen`命令在本地生成公钥和私钥，并将公钥的内容复制到Docker内

```bash
nano ~/.ssh/authorized_keys
```

### 2.4 添加环境变量

在使用ssh直接连接到docker内时，`npu-smi info`命令可能出现无法使用的情况，可根据错误提示按照如下方法解决该问题

- npu-smi: error while loading shared libraries: libdrvdsmi_host.so: cannot open shared object file: No such file or directory

  如果出现该错误是因为相关的环境变量未成功添加到终端，可按照如下方法添加

  ```bash
  # 使用nano修改~/.bashrc文件
  nano ~/.bashrc
  # 在最后一行添加如下内容
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

- npu-smi: error while loading shared libraries: libdrvdsmi_host.so: cannot open shared object file: No such file or directory

  如果出现该错误是NPU驱动相关的环境变量未添加，可按照如下方法添加

  ```bash
  # 查找文件所在的路径
  find / -name libdrvdsmi_host.so
  # 将文件路径添加到环境变量中
  export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
  ```

为了方便在每一次打开终端时都可以正常运行npu-smi指令，可以将以上内容添加到`~/.bashrc`文件中

```bash
# 编辑~/.bashrc文件
nano ~/.bashrc
```

- 在最后添加`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- 添加`export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:$LD_LIBRARY_PATH`
- 添加`export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH`
- 添加`export LD_LIBRARY_PATH=/usr/local/Ascend/driver/tools/hccn_tool/:$LD_LIBRARY_PATH`
- 添加`export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons/:$LD_LIBRARY_PATH`

完成添加后，使用`source ~/.bashrc`命令重新装载`.bashrc`文件生效

## 3. 安装Anaconda

为了解决多种模型和MindSpore对Python版本的需求不一致的问题，安装安装Anaconda创建虚拟Python环境解决该问题。

### 3.1 下载Anaconda

进入Anaconda的官网获取最新版本的conda安装链接，并如下命令安装Anaconda

> [!NOTE]
>
> 下载Anaconda时需要注意对应的版本号和芯片架构类型，例如在Ascend上就是ARM64架构

```bash
# 从官网获取下载链接：https://www.anaconda.com/download#downloads
wget https://repo.anaconda.com/archive/Anaconda3-xxx.sh
# 安装Anaconda
bash Anaconda3-xxx.sh
# 刷新环境变量，否则无法启用conda命令
source ~/.bashrc
```

> [!TIP]
>
> 在Ascend上下载的版本为：`Anaconda3-2024.02-1-Linux-aarch64.sh`

### 2.2 常见的conda命令

1. 查看全部的环境：`conda env list`
2. 激活指定环境：`conda activate <name>`
3. 创建新的环境：`conda create -n <name> python=<3.7.11>`
4. 删除指定环境：`conda remove -n <name> --all`

### 2.3 创建并进入指定Python环境

使用如下命令创建一个名为`llama2`的Python环境，后续的所有操作都在该环境下进行

```bash
# 创建适用于llama2的虚拟环境（如果已经存在则无需重复创建）
conda create -n llama2 python=3.7.11
# 进入虚拟环境
conda activate llama2
```

## 3. 配置Mindspore

由于Llama2模型原生使用NCCL（NVIDIA Collective Communications Library）技术，这将导致在Ascend平台上无法正常的运行原生模型，因此需要使用[Mindspore Mindformers](https://gitee.com/mindspore/mindformers)框架下修改的Llama2模型。

### 3.1 安装CANN

首先使用如下命令确定NPU已成功挂载到服务器上

```bash
lspci | grep accelerators
```

在[昇腾社区](https://www.mindspore.cn/versions)中搜索对应的CANN版本，并下载，选择版本时请选择开发套件`toolkit`，文件格式为`*.run`，下载完成后使用FTP或SFTP等工具将文件传输至服务器中

在安装CANN之前需要安装一些必要的工具包，安装命令如下所示（以Ubuntu系统为例）

```bash
# 更新apt的源
apt update
# 安装必备组件
apt install -y nano gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3
```

使用`python --version`检查Python的版本，根据官网的要求，CANN要求的Python版本范围是：`python3.7.5~3.7.11`、`python3.8.0~3.8.11`、`python3.9.0~3.9.7`和`python3.10.0~3.10.12`，如果Python版本不匹配，则需要使用Anaconda创建对应版本的虚拟环境。

使用pip安装必要的依赖包：

```bash
pip install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions
```

将下载的`*.run`文件通过FTP或SFTP等方式上传到服务器上，然后按照如下命令安装CANN开发包

```bash
# 修改文件的可执行权限
chmod +x ./Ascend-cann-toolkit_7.0.0_linux-aarch64.run
# 开始安装CANN工具包
./Ascend-cann-toolkit_7.0.0_linux-aarch64.run --install
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

> [!TIP]
>
> 这里使用的CANN版本为7.0.0，在选择版本时需要与MindSpore版本一致，也需要与Mindformers版本一致，否则将导致不可预料的问题。CANN与MindSpore版本匹配查询[网站](https://www.mindspore.cn/versions#ascend配套软件包)

### 3.2 构建Mindspore

Mindspore可以使用pip直接安装，安装过程参考[Mindspore](https://gitee.com/mindspore/mindspore)，pip安装的Mindspore版本参考如该[网站](https://www.mindspore.cn/versions)

```bash
# 从以下网站获取对应的pip包网址
# https://www.mindspore.cn/versions
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.11/MindSpore/unified/aarch64/mindspore-2.2.11-cp37-cp37m-linux_aarch64.whl
```

> [!TIP]
>
> 这里选择最新版Mindspore（2.2.11版本），硬件平台是Ascend910_Linux-aarch64的Python3.7版本的安装链接

### 3.3 构建Mindformers

可以使用git下载[Mindformers的源码](https://gitee.com/mindspore/mindformers)，然后使用自带的构建工具构建Mindformers环境，但是由于版本依赖问题，不是很推荐使用这种方法，可以使用pip安装对应版本的包

```bash
# 下载Mindformers
wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.11/MindFormers/any/mindformers-1.0.0-py3-none-any.whl
# 安装Mindformers
pip install mindformers-1.0.0-py3-none-any.whl
```

### 3.4 环境配置结果验证

- 使用`npu-smi info`查看docker创建时NPU是否正常挂载

  <img src="./images/image-20240407110106052.png" alt="image-20240407110106052" style="zoom:50%;" />

- 使用如下命令验证Mindspore的安装结果

  ```bash
  python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
  ```

  ![image-20240407110139690](./images/image-20240407110139690.png)

## 4. 运行Llama2

Mindformers自带Llama2的模型，可以直接使用API调用Llama2，在缺失权重文件和Token文件时会自动下载对应的缺失文件

### 4.1 单机单卡推理

配置单机多卡或多机多卡需要额外的配置服务器的信息，这里先使用单机单卡推理任务验证环境是否配置成功，在任意文件夹创建如下Python文件，并命名为`llama2.py`

可以根据如下几种API调用方法中的任意一种进行单机单卡推理任务，然后使用`python llama2.py`命令运行

#### A. 基于AutoClass

```python
# llama2.py
import mindspore
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)
tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# model的实例化有以下两种方式，选择其中一种进行实例化即可
# 1. 直接根据默认配置实例化
model = AutoModel.from_pretrained('llama2_7b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('llama2_7b')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
# config.xxx = xxx                      # 根据需求自定义修改其余模型配置
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

inputs = tokenizer("I love Beijing, because")["input_ids"]
# 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
outputs = model.generate(inputs, max_new_tokens=30, do_sample=False)
response = tokenizer.decode(outputs)
print(response)
# ['<s>I love Beijing, because it’s a city that is constantly changing. I have been living here for 10 years and I have seen the city change so much.I']
```

#### B. 基于Pipeline

```python
import mindspore
from mindformers.pipeline import pipeline

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

pipeline_task = pipeline("text_generation", model='llama2_7b', max_length=30)
pipeline_result = pipeline_task("I love Beijing, because", do_sample=False)
print(pipeline_result)
# [{'text_generation_text': ['<s>I love Beijing, because it’s a a city that is constantly changing. I have been living here for 10 years and I have']}]
```

### 4.2 单机多卡xxx

xxx

## 附录

```bash
一、docker启动命令：
isula run -it --ipc=host --name zhangdx \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v /var/log/npu/:/usr/slog \
-v /home/zhangdx/:/root/ \
--net=host \
9a2bf88cad61 \
/bin/bash


二、配置容器内的虚拟器apt源：
wget -O /etc/apt/sources.list https://repo.huaweicloud.com/repository/conf/Ubuntu-Ports-bionic.list
apt-get update
```







