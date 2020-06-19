<p align="center">
  <img src="https://img-blog.csdnimg.cn/2020061911515938.png" width="500px" alt="风格转换">
</p>


# WCT Style Transfer :art:

> 图像风格转换的深度学习算法研究和应用开发

[**[Live Demo: WCT Style Transfer]**](http://wct.shbang.ink) *部署在阿里云ECS，2020.10将会过期*


* [简介](#简介)
* [算法](#算法)
  + [WCT风格转换](#WCT风格转换)
* [本地安装](#本地安装)
  + [预训练模型](#预训练模型)
  + [CUDA](#CUDA)
  + [后端](#后端)
  + [前端](#前端)


## 简介

实现一个高效的图像风格转换算法，通过Web应用让更多人使用图像风格转换技术进行娱乐或艺术创作。

### 截屏

<p align="center">
  <img src="https://img-blog.csdnimg.cn/20200619121620158.png" alt="">
</p>


## 算法

为了实现图像风格转换，参考了以下两篇论文，本应用的图像风格转换模块是基于第二篇论文的PyTorch实现:

- [**Image Style Transfer Using Convolutional Neural Networks**](http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- [**Universal Style Transfer via Feature Transforms**](https://arxiv.org/pdf/1705.08086v2.pdf)

### WCT 风格转换

图像风格转换从 [Prisma](https://prisma-ai.com/) app 开始风行全世界。不过它的风格转换方式是通过为每一个风格训练一个模型，所以我们只能用他们提供的风格图片来做风格转换这种好玩的事情。但是，参考《[**Universal Style Transfer via Feature Transforms**](https://arxiv.org/pdf/1705.08086v2.pdf)》实现的风格转换可以实现任意的风格转换，并且具有很高的效率。

<p align="center">
  <img src="https://user-images.githubusercontent.com/5097752/41201821-f40a5cb6-6cf1-11e8-917f-779f4055ffc5.jpg" width="400px" alt="">
</p>

## 本地安装
### 预训练模型
将这些训练好的[模型](https://github.com/S-HuaBomb/pytorch-in-flask/releases/tag/v1.0)放入项目中的 `server/models` 文件夹下。

### CUDA
本人的笔记本开发环境如下：
项目 | 内容
|:---:|:---:|
CPU | AMD 3550H, 2.10GHz
RAM | 16GB
GPU | AMD...
OS | Windows10 (64bit)

所以，在本机没有 CUDA 的硬伤下，本项目跑完一次风格转换的用时是60秒左右。 live demo 中的服务器配置页比较低，是最便宜的没有 GPU 的阿里云服务器。我去掉一层风格转换网络，并且将输入图片裁剪到 256×256 后才能达到 15 秒左右的速度。

CUDA能安装尽量安装。

### 后端

后端使用Python和Flask。只需写一个路由处理前后端通信即可。
进入到 `server` 目录，服务端代码和PyTorch风格转换代码都在里面，启动服务端的脚本文件是:
- `app_stylize.py`

#### 依赖

确保本机安装了Python3.7, 以下这些依赖是必须的: torch, pillow, flask, gevent。
可以直接通过 `requirements.txt` 来安装这些依赖:

```bash
pip install -r requirements.txt
```

#### 运行

```bash
python app_stylize.py
```

运行成功的程序会在 5002 端口监听请求。浏览器进入这个URL：`http://localhost:5002` 就能看到 `/index` 首页路由的简单返回。

### 前端

需要安装好，配置好 `npm` 的系统环境:

- [Node.js](https://nodejs.org)

```
# 克隆本项目到本地
git clone git@github.com:S-HuaBomb/pytorch-in-flask.git
cd pytorch-in-flask

# 安装前端依赖
npm install

# 运行前端
npm run dev
```

在浏览器打开这个URL：`http://localhost:8080`，就能看到前端界面，并且能跟后端完美通信。
