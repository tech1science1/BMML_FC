# BMML_FC
=======
> the code for a framework BMML_FC
-----------------------------------

### Core environment
- Python (≥ 3.8 recommended)
- PyTorch (version matching your CUDA toolkit)
- Common scientific stack: `numpy`, `scipy`, `scikit-learn`, `opencv-python`, `tqdm`, etc.
- attach please find requirements.txt

### We **strongly recommend** installing SAM from the official repository:

- SAM (2D):  
  <https://github.com/facebookresearch/segment-anything.git>
- SAM 2(2D):
  <https://github.com/facebookresearch/sam2.git>
- SAM 3(2D):
  <https://github.com/facebookresearch/sam3.git>

If your work requires 3D extensions, you may consider more advanced SAM variants, e.g.: 
- SAM2point(3D, based on SAM2):
  <https://github.com/ZiyuGuo99/SAM2Point.git>

--- Use /scripts/a1_SAMandCropping2.py to genrate masks and cropped images

--- /scripts/a1_SAMandCropping2.py calculates the additional features, including the DPF features. Because the papaya coding of the 1st author, it takes time to complete...  


### /scripts/unet.py are used to trianed the DL models. **recommend** to get the code from:

- U-Net
  <https://github.com/milesial/Pytorch-UNet.git>
- Attention U-Net:  
  <https://github.com/sfczekalski/attention_unet.git>
- Swin-UNet:  
  <https://github.com/HuCaoFighting/Swin-Unet.git>


### Jilin-1 satellite imagery

The main experiments are conducted on **Jilin-1** high-resolution imagery over coastal salt marshes.

- Jilin-1 satellite images can be obtained through an **education account** at:  
  <https://www.jl1mall.com/edu/?fromUrl=https://www.jl1mall.com/>

Please follow the license and data-use agreements of the provider.


### Repository data layout

- `asset/`  
  - Contains sample data and **SAM-generated masks** in zero-shot mode.



### Files are being organized; more scripts will be updated soon....



