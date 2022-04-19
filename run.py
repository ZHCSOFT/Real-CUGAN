import torch
from torch import nn as nn
import os
import cv2
from glob import glob
from upcunet_v3 import RealWaifuUpScaler
from time import time as ttime  


ModelPath = './weights_v3' # model dir
PendingPath = './pending/' # input dir
FinishPath = './finish/' # output dir

ModelName = 'up4x-latest-no-denoise.pth' # default model
Tile = 4 #{0,1,2,3,4,auto}; the larger the number, the smaller the memory consumption

if not os.path.exists(FinishPath):
    os.mkdir(FinishPath)

pic_paths = glob(PendingPath + '.png')
pic_paths.extend(glob(PendingPath + '.jpg'))
pic_paths.extend(glob(PendingPath + '.jpeg'))
pic_paths.extend(glob(PendingPath + '.bmp'))

all_pending_paths = os.listdir(PendingPath)

print('Pending images:')
for pic_path in pic_paths:
    print("\t" + pic_path)

print('Excluded pending images:')
for pending_path in all_pending_paths:
    if not pending_path in pic_paths:
        print("\t" + pending_path)

pic_paths = os.listdir(ModelPath)
print("Model files available:")

for idx, i in enumerate(pic_paths):
    print(f"{idx+1}. \t {i}")

choice = int(input("Select model (leave blank for default): "))
if choice:
    ModelName = pic_paths[choice-1]
    
amplification = ModelName[2] # amplifying ratio
if (not os.path.isfile(ModelPath + ModelName)):
    print("Warning: selected model file does not exist")

pic_paths = os.listdir(PendingPath)
print(f"using model {ModelPath + '/' + ModelName}")
upscaler = RealWaifuUpScaler(amplification, ModelPath + '/' + ModelName, half=True, device="cuda:0")

t0 = ttime()

for i in pic_paths:
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
