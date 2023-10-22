# SASIC-opt
Forked from SASIC to test out the performance with TensorRT

# SASIC
Official code of CVPR paper *"SASIC: Stereo Image Compression with Latent Shifts and Stereo Attention"* by Matthias Wödlinger, Jan Kotera, Jan Xu, Robert Sablatnig

## Installation

Install the necessary packages from the `requirements.txt` file with pip:

```pip install -r requirements.txt```

## Data
For sample test, we only use the example image in the assets folder. frankfurt_000000_009291_leftImg8bit.png and frankfurt_000000_009291_rightImg8bit.png

## Encoding/decoding
 Step 1: Generate ONNX file using "all_onnx_wrapper.py", a ONNX file ("model_wrapper.onnx") will be generated 
     
     python3 all_onnx_wrapper.py --gpu --left assets/frankfurt_000000_009291_leftImg8bit.png --right assets/frankfurt_000000_009291_rightImg8bit.png --output_filename "frankfurt_000000_009291.sasic" --model experiments/cityscapes_lambda0.01_500epochs/model.pt

 Step 2: Using TensorRT to obtain the trt model, a trt model with fp32 precision will be genarted
     
     trtexec --onnx=model_wrapper.onnx --saveEngine=wrapper_fp32.trt --noTF32

 Step 3: Using "tensorrt_all.py" to get the output images, two png files will be generated corresponding with left and right eye images

     python3 tensorrt_all.py


## Examples
![image](./assets/cityscapes_example-01.png "Qualitative comparison for a sample from the Cityscapes dataset")
![image](./assets/instereo_example-02.png "Qualitative comparison for a sample from the InStereo2k dataset")

## Citation

If you use this project please consider citing our work

```
@InProceedings{Wodlinger_2022_CVPR,
    author    = {W\"odlinger, Matthias and Kotera, Jan and Xu, Jan and Sablatnig, Robert},
    title     = {SASIC: Stereo Image Compression With Latent Shifts and Stereo Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {661-670}
}
```
