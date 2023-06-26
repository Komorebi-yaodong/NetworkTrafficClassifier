import os
from PIL import Image
from array import *
import math
import numpy as np
import torch
from torchvision import transforms
import pandas as pd


save_csv = "./cache/using.csv"
save_png = "./cache/using.png"

def png2input():

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),  # 将数据转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    data_image = array('B')
    Im = Image.open(save_png)
    pixel = Im.load()
    width, height = Im.size
    for x in range(0,width):
        for y in range(0,height):
            data_image.append(pixel[y,x])
    image = np.array(list(data_image),dtype=np.uint8).reshape(28,28,1)
    input = torch.FloatTensor(transform(image)).unsqueeze(0)

    return input


def csv2input(flag="de"):
    head_init = ["Flow ID","Src IP","Dst IP","Src Port","Dst Port","Protocol","Flow Pkt Num","Flow Pld Byte Sum","Flow Pld Byte Max","Flow Pld Byte Min","Flow Pld Byte Mean","Flow Pld Byte Std","Fwd Pkt Num","Fwd Pld Byte Sum","Fwd Pld Byte Max","Fwd Pld Byte Min","Fwd Pld Byte Mean","Fwd Pld Byte Std","Bwd Pkt Num","Bwd Pld Byte Sum","Bwd Pld Byte Max","Bwd Pld Byte Min","Bwd Pld Byte Mean","Bwd Pld Byte Std","Fwd Head Byte Max","Fwd Head Byte Min","Fwd Head Byte Mean","Fwd Head Byte Std","Bwd Head Byte Max","Bwd Head Byte Min","Bwd Head Byte Mean","Bwd Head Byte Std","Flow Duration(ms)","Flow Pkts/s","Flow Pld Bytes/ms","Fwd Pkts/s","Fwd Pld Bytes/ms","Bwd Pkts/s","Bwd Pld Bytes/ms","Pkts Ratio","Bytes Ratio","Flow IAT Max","Flow IAT Min","Flow IAT Mean","Flow IAT Std","Fwd IAT Max","Fwd IAT Min","Fwd IAT Mean","Fwd IAT Std","Bwd IAT Max","Bwd IAT Min","Bwd IAT Mean","Bwd IAT Std","FIN Count","SYN Count","RST Count","PSH Count","ACK Count","URG Count","ECE Count","CWR Count","Fwd PSH Count","Bwd PSH Count","Fwd URG Count","Bwd URG Count","Fwd Init Win Bytes","Bwd Init Win Bytes","Fwd Pkts With Pld","Bwd Pkts With Pld","Sub Flow Fwd Pkts","Sub Flow Fwd Bytes","Sub Flow Bwd Pkts","Sub Flow Bwd Bytes","Flow Act Sum","Flow Act Max","Flow Act Min","Flow Act Mean","Flow Act Std","Flow Idle Sum","Flow Idle Max","Flow Idle Min","Flow Idle Mean","Flow Idle Std","Fwd Avg Pkts/Bulk","Fwd Avg Bytes/Bulk","Fwd Avg Bulk/s","Bwd Avg Pkts/Bulk","Bwd Avg Bytes/Bulk","Bwd Avg Bulk/s"]
    head_de = ['Src Port', 'Flow IAT Min', 'Fwd IAT Min', 'Flow Act Min', 'Flow Act Max', 'Flow Act Sum', 'Flow IAT Max', 'Flow Act Mean', 'Bwd IAT Min', 'Fwd IAT Max', 'Flow IAT Std', 'Flow Duration(ms)', 'Flow IAT Mean', 'Fwd Pkts/s', 'Flow Pkts/s', 'Fwd IAT Mean', 'Bwd IAT Max', 'Bwd Pkts/s', 'Fwd IAT Std', 'Bwd IAT Mean', 'Flow Act Std', 'Bwd IAT Std', 'Flow Idle Min', 'Bwd Init Win Bytes', 'Fwd Pld Byte Max', 'Flow Idle Mean', 'Flow Idle Sum', 'Flow Idle Max', 'Bytes Ratio', 'Fwd Pld Bytes/ms', 'Flow Pld Byte Std', 'Fwd Pld Byte Std', 'Flow Pld Byte Mean', 'Flow Pld Bytes/ms', 'Fwd Pld Byte Sum', 'Pkts Ratio', 'Bwd Pld Bytes/ms', 'Fwd Init Win Bytes', 'Sub Flow Fwd Bytes', 'Flow Pld Byte Sum', 'Dst Port', 'Fwd Pld Byte Mean', 'Bwd Pld Byte Std', 'Bwd Pld Byte Mean', 'Sub Flow Bwd Bytes', 'Bwd Pld Byte Sum', 'Fwd Head Byte Mean', 'Flow Pld Byte Max', 'Flow Idle Std', 'Fwd Head Byte Std', 'Bwd Head Byte Mean', 'ACK Count', 'Bwd Head Byte Std', 'Sub Flow Fwd Pkts', 'Bwd Avg Bulk/s', 'Bwd Pld Byte Max', 'Sub Flow Bwd Pkts', 'Flow Pkt Num', 'Fwd Pkt Num', 'Bwd Avg Bytes/Bulk', 'FIN Count', 'Bwd Pkt Num', 'PSH Count', 'Bwd Pkts With Pld', 'Bwd Avg Pkts/Bulk', 'Bwd PSH Count', 'Bwd Head Byte Max', 'Fwd Head Byte Min', 'Fwd PSH Count', 'Fwd Pkts With Pld', 'RST Count', 'SYN Count', 'Fwd Head Byte Max', 'Bwd Head Byte Min']
    head_en = ['Src Port', 'Flow IAT Min', 'Fwd IAT Min', 'Flow Act Min', 'Flow IAT Max', 'Flow Act Mean', 'Flow Act Max', 'Flow Act Sum', 'Flow Duration(ms)', 'Fwd Pkts/s', 'Dst Port', 'Flow Pkts/s', 'Flow IAT Mean', 'Bwd Pkts/s', 'Fwd IAT Max', 'Flow IAT Std', 'Fwd Init Win Bytes', 'Bwd Init Win Bytes', 'Bwd IAT Min', 'Fwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Mean', 'Fwd IAT Std', 'Flow Pld Bytes/ms', 'Fwd Pld Bytes/ms', 'Bwd IAT Std', 'Fwd Pld Byte Std', 'Flow Pld Byte Std', 'Fwd Pld Byte Max', 'Bytes Ratio', 'Flow Pld Byte Mean', 'Bwd Pld Bytes/ms', 'Fwd Pld Byte Sum', 'Flow Pld Byte Sum', 'Sub Flow Fwd Bytes', 'Fwd Pld Byte Mean', 'Pkts Ratio', 'Bwd Pld Byte Std', 'Flow Idle Min', 'Sub Flow Bwd Bytes', 'Bwd Pld Byte Mean', 'Flow Idle Sum', 'Bwd Pld Byte Sum', 'Flow Idle Mean', 'Flow Idle Max', 'Flow Act Std', 'Bwd Head Byte Mean', 'Fwd Head Byte Mean', 'Bwd Avg Bulk/s', 'Fwd Head Byte Std', 'ACK Count', 'Sub Flow Fwd Pkts', 'Flow Pkt Num', 'Bwd Avg Bytes/Bulk', 'Sub Flow Bwd Pkts', 'Bwd Head Byte Std', 'Fwd Pkt Num', 'PSH Count', 'Flow Pld Byte Max', 'Bwd Pkt Num', 'Bwd PSH Count', 'Flow Idle Std', 'Bwd Pkts With Pld', 'Bwd Pld Byte Max', 'Fwd Pkts With Pld', 'Fwd PSH Count', 'FIN Count', 'Bwd Avg Pkts/Bulk', 'Fwd Avg Bulk/s', 'Fwd Avg Bytes/Bulk', 'Bwd Head Byte Max', 'RST Count', 'Bwd Head Byte Min', 'Fwd Avg Pkts/Bulk', 'Fwd Head Byte Max', 'Fwd Head Byte Min', 'Fwd Pld Byte Min', 'SYN Count', 'Bwd Pld Byte Min', 'Flow Pld Byte Min']

    data = pd.read_csv(save_csv,encoding="utf-8")
    data.columns = head_init
    
    if flag == "de":
        data = data.loc[0,head_de]
    else:
        data = data.loc[0,head_en]
    
    input = data.to_list()
    return input