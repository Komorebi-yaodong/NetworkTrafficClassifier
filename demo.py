import chainlit as cl
import os
import torch
from joblib import load


from pre_rawdata import pcap2csv,pcap2png
from pre_data import csv2input,png2input
from model.P2P_NeXt_de import Model as classifier_p2p_de
from model.P2P_NeXt_en import Model as classifier_p2p_en


def generate_feature(head,data):
    res = "feature name".ljust(50," ") + "feature".ljust(50," ") + "\n"
    for i in range(len(head)):
        tmp = head[i].ljust(50," ") + str(data[i]).ljust(50," ") + "\n"
        res = res + tmp
    return res

"""
流程：
1. 用户输入文件
2. 调用模型进行分析
3. 返回结果
"""

STATE = "de"
CLASS = ["善意流量","恶意流量"]
MODE = {"de":"Plaintext Mode","en":"Ciphertext Mode"}

# 加载分类器模型
if os.path.exists("./model_ckpt/P2P_NeXt_de.ckpt"):
    cp2p_d = classifier_p2p_de()
    cp2p_d.load_state_dict(torch.load("./model_ckpt/P2P_NeXt_de.ckpt"))
    cp2p_d = cp2p_d.eval()

if os.path.exists("./model_ckpt/P2P_NeXt_en.ckpt"):
    cp2p_e = classifier_p2p_en()
    cp2p_e.load_state_dict(torch.load("./model_ckpt/P2P_NeXt_en.ckpt"))
    cp2p_e = cp2p_e.eval()


head_de = ['Src Port', 'Flow IAT Min', 'Fwd IAT Min', 'Flow Act Min', 'Flow Act Max', 'Flow Act Sum', 'Flow IAT Max', 'Flow Act Mean', 'Bwd IAT Min', 'Fwd IAT Max', 'Flow IAT Std', 'Flow Duration(ms)', 'Flow IAT Mean', 'Fwd Pkts/s', 'Flow Pkts/s', 'Fwd IAT Mean', 'Bwd IAT Max', 'Bwd Pkts/s', 'Fwd IAT Std', 'Bwd IAT Mean', 'Flow Act Std', 'Bwd IAT Std', 'Flow Idle Min', 'Bwd Init Win Bytes', 'Fwd Pld Byte Max', 'Flow Idle Mean', 'Flow Idle Sum', 'Flow Idle Max', 'Bytes Ratio', 'Fwd Pld Bytes/ms', 'Flow Pld Byte Std', 'Fwd Pld Byte Std', 'Flow Pld Byte Mean', 'Flow Pld Bytes/ms', 'Fwd Pld Byte Sum', 'Pkts Ratio', 'Bwd Pld Bytes/ms', 'Fwd Init Win Bytes', 'Sub Flow Fwd Bytes', 'Flow Pld Byte Sum', 'Dst Port', 'Fwd Pld Byte Mean', 'Bwd Pld Byte Std', 'Bwd Pld Byte Mean', 'Sub Flow Bwd Bytes', 'Bwd Pld Byte Sum', 'Fwd Head Byte Mean', 'Flow Pld Byte Max', 'Flow Idle Std', 'Fwd Head Byte Std', 'Bwd Head Byte Mean', 'ACK Count', 'Bwd Head Byte Std', 'Sub Flow Fwd Pkts', 'Bwd Avg Bulk/s', 'Bwd Pld Byte Max', 'Sub Flow Bwd Pkts', 'Flow Pkt Num', 'Fwd Pkt Num', 'Bwd Avg Bytes/Bulk', 'FIN Count', 'Bwd Pkt Num', 'PSH Count', 'Bwd Pkts With Pld', 'Bwd Avg Pkts/Bulk', 'Bwd PSH Count', 'Bwd Head Byte Max', 'Fwd Head Byte Min', 'Fwd PSH Count', 'Fwd Pkts With Pld', 'RST Count', 'SYN Count', 'Fwd Head Byte Max', 'Bwd Head Byte Min']
head_en = ['Src Port', 'Flow IAT Min', 'Fwd IAT Min', 'Flow Act Min', 'Flow IAT Max', 'Flow Act Mean', 'Flow Act Max', 'Flow Act Sum', 'Flow Duration(ms)', 'Fwd Pkts/s', 'Dst Port', 'Flow Pkts/s', 'Flow IAT Mean', 'Bwd Pkts/s', 'Fwd IAT Max', 'Flow IAT Std', 'Fwd Init Win Bytes', 'Bwd Init Win Bytes', 'Bwd IAT Min', 'Fwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Mean', 'Fwd IAT Std', 'Flow Pld Bytes/ms', 'Fwd Pld Bytes/ms', 'Bwd IAT Std', 'Fwd Pld Byte Std', 'Flow Pld Byte Std', 'Fwd Pld Byte Max', 'Bytes Ratio', 'Flow Pld Byte Mean', 'Bwd Pld Bytes/ms', 'Fwd Pld Byte Sum', 'Flow Pld Byte Sum', 'Sub Flow Fwd Bytes', 'Fwd Pld Byte Mean', 'Pkts Ratio', 'Bwd Pld Byte Std', 'Flow Idle Min', 'Sub Flow Bwd Bytes', 'Bwd Pld Byte Mean', 'Flow Idle Sum', 'Bwd Pld Byte Sum', 'Flow Idle Mean', 'Flow Idle Max', 'Flow Act Std', 'Bwd Head Byte Mean', 'Fwd Head Byte Mean', 'Bwd Avg Bulk/s', 'Fwd Head Byte Std', 'ACK Count', 'Sub Flow Fwd Pkts', 'Flow Pkt Num', 'Bwd Avg Bytes/Bulk', 'Sub Flow Bwd Pkts', 'Bwd Head Byte Std', 'Fwd Pkt Num', 'PSH Count', 'Flow Pld Byte Max', 'Bwd Pkt Num', 'Bwd PSH Count', 'Flow Idle Std', 'Bwd Pkts With Pld', 'Bwd Pld Byte Max', 'Fwd Pkts With Pld', 'Fwd PSH Count', 'FIN Count', 'Bwd Avg Pkts/Bulk', 'Fwd Avg Bulk/s', 'Fwd Avg Bytes/Bulk', 'Bwd Head Byte Max', 'RST Count', 'Bwd Head Byte Min', 'Fwd Avg Pkts/Bulk', 'Fwd Head Byte Max', 'Fwd Head Byte Min', 'Fwd Pld Byte Min', 'SYN Count', 'Bwd Pld Byte Min', 'Flow Pld Byte Min']

if os.path.exists("./model_ckpt/ML_rf_de.joblib"):
    rf_de = load("./model_ckpt/ML_rf_de.joblib")

if os.path.exists("./model_ckpt/ML_rf_en.joblib"):
    rf_en = load("./model_ckpt/ML_rf_en.joblib")
## cations

@cl.action_callback("分类模式")
async def on_action(action):
    global STATE,MODE

    if STATE == "de":
        STATE = "en"
    elif STATE == "en":
        STATE = "de"
    else:
        print("error!")
    
    action1 = [
        cl.Action(name="分类模式", value="None", label="切换模式", description="点击切换分类模式")
    ]
    
    await cl.Message(content=f"{MODE[STATE]}", actions=action1).send()
    action2 = [
        cl.Action(name="流量分类", value="None", label="开始分类", description="点击进行流量分类")
    ]
    await cl.Message(content=f"Start with {MODE[STATE]}", actions=action2).send()
    # await action.remove()


@cl.action_callback("流量分类")
async def on_action(action):
    # 询问获取流pcap文件
    file = None
    while file == None:
        file = await cl.AskFileMessage(
            content="请上传会话流文件（.pcap）",accept=["pcap"]
        ).send()
    file_byte = file[0].content
    
    # 原始文件提取信息
    pcap2png(file_byte)
    pcap2csv(file_byte)

    # 预处理文件处理得到训练信息
    data_p2p = png2input()
    data_rf = [csv2input(STATE),]
    # 进行分类
    if STATE == "de":
        # p2p
        with torch.no_grad():
            prediction_p2p = torch.max(cp2p_d(data_p2p).data, 1)[1].cpu().tolist()[0]
        # rf
        head = head_de
        prediction_rf = int(rf_de.predict(data_rf).tolist()[0])

    else:
        # p2p
        with torch.no_grad():
            prediction_p2p = torch.max(cp2p_e(data_p2p).data, 1)[1].cpu().tolist()[0]
        # rf
        head = head_en
        prediction_rf = int(rf_en.predict(data_rf).tolist()[0])

    # 展示结果 P2P
    res_p2p = CLASS[prediction_p2p]
    elements1 = [
        cl.Image(name="卷积网络输入图", display="inline", path="./cache/using.png",size="large")
    ]
    elements2 = [
        cl.Image(name="卷积网络生成图", display="inline", path="./cache/showjpg.jpg",size="large")
    ]

    await cl.Message(content="深度学习RESNEXT模型——分类").send()
    await cl.Message(content="卷积网络输入图像如下（28,28）", elements=elements1,indent=1).send()
    await cl.Message(content="卷积网络视觉分析得到图像如下（32,32）放大为（512,512）", elements=elements2,indent=1).send()
    await cl.Message(content=f"分类结果："+res_p2p,indent=1).send()

    # 展示结果 CSV
    res_rf = CLASS[prediction_rf]
    content = generate_feature(head,data_rf[0])
    await cl.Message(content="机器学习随机森林模型——分类").send()
    await cl.Message(content="流量处理得到的特征向量",indent=1).send()
    await cl.Message(content=content,indent=2).send()
    await cl.Message(content=f"分类结果："+res_rf,indent=1).send()
    # # send back the final answer
    # await cl.Message(content=f"分类结束").send()

    action1 = [
        cl.Action(name="分类模式", value="None", label="切换模式", description="点击切换分类模式")
    ]
    await cl.Message(content=f"{MODE[STATE]}", actions=action1).send()
    action2 = [
        cl.Action(name="流量分类", value="None", label="开始分类", description="点击进行流量分类")
    ]
    await cl.Message(content=f"Start with {MODE[STATE]}", actions=action2).send()



@cl.on_chat_start  # this function will be called every time a user inputs a message in the UI
async def start():
    # while True:

    # 选择 明文分类 还是 密文分类
    action1 = [
        cl.Action(name="分类模式", value="None", label="切换模式", description="点击切换分类模式")
    ]
    await cl.Message(content=f"{MODE[STATE]}", actions=action1).send()

    action2 = [
        cl.Action(name="流量分类", value="None", label="开始分类", description="点击进行流量分类")
    ]
    await cl.Message(content=f"Start with {MODE[STATE]}", actions=action2).send()
        
        

        