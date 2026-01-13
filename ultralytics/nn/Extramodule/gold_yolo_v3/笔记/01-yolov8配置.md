# 1.创建并配置虚拟环境

**利用conda创建虚拟环境（没有安装anaconda需要先安装anaconda）**

```
conda create -n GOLD-YOLO python=3.9 -y
```

```
# 这里额外提供两个指令
# 查看当前虚拟环境列表
conda env list
# 删除虚拟环境
conda remove -n 虚拟环境名 --all
```

**配置虚拟环境下的pytorch环境**

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).（yolov8的python版本要求和pytorch版本要求）

cuda11.8下载地址：https://download.pytorch.org/whl/cu118

如果要下载其他版本的cuda, 修改网页后面的数字即可，例如我要下载cuda11.7，我要访问的网页就是https://download.pytorch.org/whl/cu117

然后咱目前要下载的就只有两个：torch和torchvision。（点击网页中的torch和torchvision下载即可）

例如：下载torch，点击torch，然后选择对应的torch版本、python版本，我这里要下载的是torch2.0，cuda版本是11.8，python版本3.9，就Ctrl+F搜索torch-2.0.0+cu118-cp39-cp39-win_amd64.whl。

torchvision下载类似，下载的文件是torchvision-0.15.0+cu118-cp39-cp39-win_amd64.whl。

在下载文件目录，输入cmd，进入命令行窗口

![image-20240119135752529](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119135752529.png)

**激活虚拟环境**

```
conda activate GOLD-YOLO
```

通过pip install 安装刚才的文件

```
pip install torch-2.0.0+cu118-cp39-cp39-win_amd64.whl
pip install torchvision-0.15.0+cu118-cp39-cp39-win_amd64.whl
```

**查看pytorch是否配置成功**

直接命令行输入python进入python环境

```python
import torch
print(torch.cuda.is_available()) 
```

结果返回true则表示pytorch环境配置成功。

![image-20240119141321981](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119141321981.png)

# 2. 安装YOLOv8

YOLOv8 github地址：https://github.com/ultralytics/ultralytics

YOLOv8 参考文档地址：https://docs.ultralytics.com/modes/train/#introduction

打开Pycharm，新建一个项目New Project

![image-20240119143149642](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119143149642.png)

选择conda environment

![image-20240119143447941](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119143447941.png)

点击OK===>Create(如果刚才location的文件路径已经存在，则会有个提示，直接创建即可)

然后咱们需要在控制台中安装yolov8

![image-20240119143751561](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119143751561.png)

输入下列指令进行安装。

```
pip install ultralytics==8.0.164
```

![image-20240119144034660](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119144034660.png)

在当前文件夹下面新建train.py

```
from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8n.yaml')  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(data='coco128.yaml', epochs=20, imgsz=640)
```

**如果第一行导入出错，可以重启pycharm**

第一次运行：

Downloading https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt to 'yolov8n.pt'...
100%|██████████| 6.23M/6.23M [00:14<00:00, 457kB/s]（当前路径下不存在yolov8n.pt，会自动下载yolov8n.pt）

**那么咱们的yolov8安装在哪里呢？**

anaconda目录+envs+虚拟环境名+Lib+site-packages+ultralytics

例如我的：D:\anaconda3\envs\GOLD-YOLO\Lib\site-packages\ultralytics

可以按住Ctrl点击项目中YOLO进入该类

![image-20240119150016145](C:\Users\xiyang\AppData\Roaming\Typora\typora-user-images\image-20240119150016145.png)

上面展示就是本地ultralytics的路径。

快速进入的话 可以右键model.py ===》 选择Open in =》 Explorer 可以打开该文件的所在文件夹。



## module 'backend_interagg' has no attribute 'FigureCanvas'

\ultralytics\utils\plotting.py 导库中加入

```
import matplotlib
matplotlib.use('TkAgg')
```

