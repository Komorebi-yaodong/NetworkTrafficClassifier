# 展示说明
> 2023年——buaa网络攻防技术大作业“网络流量分类”

> 使用机器学习（随机森林）和深度学习（NeXt网络）的方法对明文流量与密文流量进行“善意流量”与“恶意流量”二分类。

## 环境配置

1. python 3.10+
2. pytorch 1.13.0+
3. sklearn
4. chainlit

## 运行说明

运行系统： `chainlit run demo.py`；

运行位置：`http://localhost:8000`；

由于随机森林模型较大，加载较慢，可能需要几十秒加载时间。

## 模型
模型存储位置：`./model_ckpt/`；

模型文件:`P2P_NeXt_en.ckpt`，`P2P_NeXt_de.ckpt`，`ML_rf_en.joblib`，`ML_rf_de.joblib`；

模型下载地址：`https://pan.baidu.com/s/19kXS_22oTQJQ61N3BLrkhg?pwd=y9ny` 提取码: `y9ny`；

其他：`chainlit.md`中也给出了模型结构，可以自行训练，其中数据处理方式在`pre_rawdata.py`与`pre_data.py`中给出，唯一需要注意的是使用的pcap文件是拆分好的一条双向流，可以在`https://github.com/echowei/DeepTraffic.git`的` 2.encrypted_traffic_classification/2.PreprocessedTools`中找到`ps1`脚本，用来拆分pcap文件为双向流pcap文件。
