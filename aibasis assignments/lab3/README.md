## 作业4

### 1. 任务介绍
本作业的目的是让同学们熟悉GAN的训练方式.

### 2. 环境安装
建议使用 Anaconda 创建一个新的虚拟环境, 安装所需的依赖包.

```bash
conda create -n aibasis python=3.10 #创建虚拟环境
conda activate aibasis #激活虚拟环境
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118 #安装pytorch
pip install numpy==1.26.3 #安装Numpy
pip install matplotlib #安装matplotlib
```

注: # 后为注释, 只需在终端依次复制“#”之前的命令执行, 在第一次创建后, 每次只需要激活对应虚拟环境即可.

### 3. 任务描述

1. 补充 `dc_gan.py` 中的鉴别器 Discriminator 训练代码（2分）
2. 补充 `dc_gan.py` 中的生成器 Generator 训练代码（2分）
3. 训练 `dc_gan` 10个epoch，保存训练 loss 曲线、模型权重文件、10个epoch的生成效果对比图（代码中已经实现这3个保存功能）（1分）。在实验报告中简单描述一下生成过程中你观察到的现象（1分）
   - 作为参考：使用GPU 2080Ti训练，一个epoch需要约60秒，占1G显存左右；使用CPU E5 2680V4训练，一个epoch需要约200秒。
   - 如果你的设备训练速度太慢，可以只训练5个epoch。
4. 在实验报告中回答以下问题：
    - 生成器和鉴别器的Loss是不是越低越好？为什么（80字左右）（2分）
    - 在本代码中，生成的数字是无法控制的（即采样一个噪声，把噪声输入到生成器中，得到的数字无法预测是什么），简单描述一下（20个字左右）如何才能生成可控的数字（即想生成哪个数字就生成哪个数字）？（2分）

> - **如果你的设备没有GPU，建议使用Google Colab进行训练，Colab提供免费的GPU资源(尽管不太稳定).**
> - **Question: How to use Colab? Ans: Just Google it!**


#### 4. 提交要求

1. 提交 `dc_gan.py` 的代码、loss曲线（命名为 `training_loss.png`）、10张对比图（命名为`compare_epochX_batchend.png`）、D和G的最后一个epoch的权重（命名为`D.pth`和`G.pth`）
2. 提交一份实验报告，需要包含任务3、4中所要求的内容。
3. 在 `pack.py` 中填写组号, 姓名与学号, 在完成所有内容后运行 `python pack.py` 打包文件, 提交得到压缩文件即可. (注, `pack.py` 在压缩时会跳过 `__pycache__` 文件夹, `data/` 文件夹 和 路径里的 `.zip` 文件.)
4. Good luck!