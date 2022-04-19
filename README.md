Real Cascade U-Nets for Anime Image Super Resolution
-------------------------------------------
[English](README.md) **|** [Simplified Chinese](README_CHS.md)

Real-CUGAN is an AI super resolution model for anime images, trained in a million scale anime dataset, using the same architecture as Waifu2x-CUNet. It supports **2x, 3x, 4x** super resolving. For different enhancement strength, now 2x Real-CUGAN supports 5 model weights, 3x and 4x Real-CUGAN supports 3 model weights.


## How to run

Download models and extract weights from archive to `weights_v3/` folder

https://github.com/ZHCSOFT/Real-CUGAN/releases/download/Real-CUGAN/weights_v3.zip

Copy image files to `pending/` for further processing, support `.JPEG`, `.JPG`, `.PNG`, `.BMP` file format. Some png format related issue will be fixed in future

Run `python run.py` following hints it prints

Output files will be saved to `finish/` folder

## Environment required
```
torch>=1.0.0
numpy
opencv-python
```
## Sample (L: origin, R: up4x-latest-no-denoise)
![raw_up4x](https://user-images.githubusercontent.com/79516102/164012929-9fa3d368-8bb6-4b5b-9a1f-fb3d31d3297b.jpg)
