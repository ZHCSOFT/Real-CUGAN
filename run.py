import torch
from torch import nn as nn
import os
import cv2
from upcunet_v3 import RealWaifuUpScaler
from time import time as ttime  


ModelPath = './weights_v3' # model dir
PendingPath = './pending/' # input dir
FinishPath = './finish/' # output dir

ModelName = 'up4x-latest-no-denoise.pth' # default model
Tile = 4 #{0,1,2,3,4,auto}; the larger the number, the smaller the memory consumption

if not os.path.exists(FinishPath):
    os.mkdir(FinishPath)

fileNames = os.listdir(PendingPath)
print("Pending images:")
for i in fileNames:
    print("\t"+i)

fileNames = os.listdir(ModelPath)
print("Model files available:")

for idx, i in enumerate(fileNames):
    print(f"{idx+1}. \t {i}")

choice = int(input("Select model (leave blank for default): "))
if choice:
    ModelName = fileNames[choice-1]
    
Amplification = ModelName[2] # amplifying ratio
if (not os.path.isfile(ModelPath + ModelName)):
    print("Warning: selected model file does not exist")

fileNames = os.listdir(PendingPath)
print(f"using model {ModelPath + '/' + ModelName}")
upscaler = RealWaifuUpScaler(4, ModelPath + '/' + ModelName, half=True, device="cuda:0")

t0 = ttime()

for i in fileNames:
    torch.cuda.empty_cache()
    try:
        img = cv2.imread(PendingPath+i)[:, :, [2, 1, 0]]
        result = upscaler(img, tile_mode=5, cache_mode=2, alpha=1)
        cv2.imwrite(FinishPath+i,result[:, :, ::-1])
    except RuntimeError as e:
        print (i + " FAILED")
        print (e)
    else:
        print(i+" DONE")
        os.remove(PendingPath+i)

t1 = ttime()
print("time_spent", t1 - t0)
