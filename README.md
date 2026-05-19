# IMA206 project : Foundation model features upsampling


You have access to 4 hyperspectral images in the `/dataset` folder (*Samson*, *Jasper*, *Apex* and *Urban*).

The `example.ipynb` notebook provides a code example to load an image, display it, extract features from it and train the unmixing network using the features.

### Feature extraction :
For this project, you will need to clone DOFA [[1]](#1) repository, download its pretrained weights (we'll use DOFA v1 large as it has shown the best results) and place it in the `/checkpoints` folder.

DOFA accepts HSI with any number of spectral bands in input, given the list of the corresponding wavelengths for each band, but it requires the image to be exactly of shape $(B, 224, 224)$. You can have a look at all 4 images provided and see that 3 of them are smaller than $224$, meaning we have to upsample them, and one is larger, so we need to crop it.

Once the image is in the correct format, we can instantiate DOFA and use it to extract features from the HSI.

### Spectral Adapter :
For this project, you will need to clone SpectralEarth [[2]](#2) repository, download its pretrained weights (you can experiment on which model's version to use) and place it in the `/data` folder.

HSI have varying number of spectral bands, which is not a problem for DOFA, but it is for NAF (we need a consistent shape to train it). We then need a way to always have the same shape for input HSI. One way is to use SpectralEarth's spectral adapter module.

**NB** : You will need to find the new list of wavelengths once adapted by SpectralEarth to go in DOFA.

<a id="1">[1]</a> 
Xiong et al. "Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation". In: CVPR 2024
url: [https://arxiv.org/pdf/2403.15356](https://arxiv.org/pdf/2403.15356)

<a id="2">[2]</a>
Braham et al. "SpectralEarth: Training Hyperspectral Foundation Models at Scale". In: CVPR 2024
url: [https://arxiv.org/pdf/2408.08447](https://arxiv.org/pdf/2408.08447)