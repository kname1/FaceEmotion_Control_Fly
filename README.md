> 这是一个面部表情飞控的项目，现在还没有接入飞行器部分，表情识别的相关逻辑已经完毕

# 本人的硬件软件环境：

开发板：

硬件：      开发板Raspberrypi 4b 2GB  摄像头： Raspberryp Pi Camera Rev 1.3

软件：	Raspberrypi OS 64bit (2024-11-19-raspios-bookworm-arm64.img)  Python3.11(预装)

PC：

硬件：AMD R5 5600H	 NVIDIA 3050 Laptop 16GB 

软件：Windows11 	PyCharm Community Edition 2024.3	Python3.8



> h5 模型是在PC上跑的 ， tflite则是在树莓派上运行

~~~python
#创建，激活虚拟环境
Virtualenv,conda均可
#安装依赖
pip install -r requirements.txt

#PC
python Camera_Classific_H5.py
#Rsp
#由于opencv2在树莓派无法获取在线视频流，所以我们使用picamera替代
sudo apt-get update
sudo apt-get install python-picamera python3-picamera
python Camera_Classific_tflite.py
~~~



