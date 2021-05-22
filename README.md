# SmartDog

这是一个基于深度学习的单目标跟踪软件，该软件可以跟踪视频目标或者通过摄像头跟踪目标。

## demo

![界面](./source/效果1.png)

## 安装方法

**推荐使用Python的解释器版本:** 3.6、3.7

使用前请安装以下python模块：  
certifi==2020.12.5  
numpy==1.20.3  
opencv-python==4.5.2.52  
Pillow==8.2.0  
PyQt5==5.15.4  
PyQt5-Qt5==5.15.2  
PyQt5-sip==12.9.0  
PyYAML==5.4.1  
torch==1.8.1  
typing-extensions==3.10.0.0  
wincertstore==0.2  
yacs==0.1.8

我们也可以通过文件`requirements.txt`来配置环境，具体语句是：`pip install -r requirements.txt`

另外需要注意的是项目里面的models文件夹里面没有包含软件运行的相关跟踪模型，因为跟踪模型文件太大所以我没有把他放到github上面。  
以下是项目中要用到的models文件，我把他放到了百度云盘，同学们可以自行下载：  
链接:https://pan.baidu.com/s/1foPa6tMTXqJ5pzPB0mavsQ  
提取码：un61 
