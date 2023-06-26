import numpy
from PIL import Image
import binascii
import os
import multiprocessing
import time
from datetime import timedelta


from cicflowmeter.PacketReader import PacketReader
from cicflowmeter.FlowGenerator import FlowGenerator

save_csv = "./cache/using.csv"
save_png = "./cache/using.png"



def pcap2csv(file:bytes):
    flowTimeout = 120000000
    activityTimeout = 5000000
    subFlowTimeout = 1000000
    bulkTimeout = 1000000

    packetReader = PacketReader(file)
    flowGenerator = FlowGenerator(flowTimeout, activityTimeout, subFlowTimeout, bulkTimeout)
    basicPacket = packetReader.nextPacket()
    while basicPacket != None:
        flowGenerator.addPacket(basicPacket)
        basicPacket = packetReader.nextPacket()
    flowGenerator.clearFlow()
    flowGenerator.dumpFeatureToCSV(save_csv)


def pcap2png(file:bytes):

    def getMatrixfrom_pcap(file:bytes,width):
        content = file[25:784+25]
        length = len(content)
        if length < 784:
            content = content + b'\x00' * (784-length)
        hexst = binascii.hexlify(content)
        fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])  
        rn = len(fh)//width
        fh = numpy.reshape(fh[:rn*width],(-1,width))  
        fh = numpy.uint8(fh)
        return fh

    PNG_SIZE = 28
    im = Image.fromarray(getMatrixfrom_pcap(file,PNG_SIZE))
    im.save(save_png)