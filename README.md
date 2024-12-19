
# Unified Image Restoration and Enhancement: Degradation Calibrated Cycle Reconstruction Diffusion Model  
#### This is the official implementation code for [CycleRDM]().


![CycleRDM](figs/fig1.png)

### Updates
[**2024.12.20**] Added [dataset links](https://github.com/hejh8/CycleRDM#dataset-links) for training and testing of various tasks. <br>
[**2024.12.20**] The **pretrained weights**  of the four task models were released separately [link1]().  <br>
[**2024.12.19**] Released test code for four image restoration tasks: image derainint, image denoising, image dehazing, and image raindrop removal. <br>

## How to Run the Code?


### Dependencies

* OS: Ubuntu 22.04
* nvidia:
	- cuda: 12.1
* python 3.9

### Install

 Clone Repo
 ```bash
 git clone https://github.com/He-Jinhong/CFGW.git
 cd CycleRDM  
 ```
Create Conda Environment and Install Dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Preparation

Preparing the train and test datasets following our paper Dataset subsection as:

```bash
#### for training dataset ####
data
|--dehazing
   |--train
      |--LQ/*.png
      |--GT/*.png
      |--train.txt
   |--test
      |--LQ/*.png
      |--GT/*.png
      |--test.txt
|--deblurring
|--denoising
|--deraining
|--raindrop
|--inpainting
|--low-light
|--underwater
|--backlight


Then You need to get into the `CycleRDM/config` directory and modify the `Task_train.yml` and `Task_test.yml` settings therein to suit your needs. 

```
For the training datas, we selected only up to 500 images in each task. You can select more training datas as you wish through the [script](https://github.com/hejh8/CycleRDM/scripts/Random_select_data.py).
```bash

cd CycleRDM/scripts

python3 Random_select_data.py 

```

#### Dataset Links



| Image Restoration Task          |                                   deblurring                                   |                                           dehazing                                           |                                           deraining                                           |            raindrop            |                                     denoising                                     |              inpainting              |
|---------------|:-----------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:-------------------------------:|:---------------------------------------------------------------------------------:|:------------------------------------:|
| Datasets      | [BSD](https://drive.google.com/drive/folders/1LKLCE_RqPF5chqWgmh3pj7cg-t9KM2Hd) | [RESIDE-6k](https://drive.google.com/drive/folders/1XVD0x74vKQ0-cqazACUZnjUOWURXIeqH?usp=drive_link) | Rain100H: [train](http://www.icst.pku.edu.cn/struct/att/RainTrainH.zip), [test](http://www.icst.pku.edu.cn/struct/att/Rain100H.zip) | [RainDrop](https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K) | [CBSD68](https://github.com/clausmichele/CBSD68-dataset?tab=readme-ov-file) | [CelebaHQ-256](https://drive.google.com/file/d/1oYDBcJLT5RDuC4k5C7xOMRkZ9N3kfexu/view?usp=sharing) |


For noisy datasets, you can use this [script]() to generate LQ images. For restoration tasks, you can generate LQ images by adding facial occlusion via the [script]().

| Image Enhancement Task |                                    low-light                                    |                              underwater                             |                                    backlight                                  |                                                                                         
|-------------|:-------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
| Datasets    | [LOLv1](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view),[LOLv2](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view) |[LSUI](https://drive.google.com/file/d/10gD4s12uJxCHcuFdX9Khkv37zzBwNFbL/view) | [Backlit300](https://drive.google.com/drive/folders/1tnZdCxmWeOXMbzXKf-V4HYI4rBRl90Qk) | 


### Training

#### Image Restoration Tasks
We will be releasing the training code shortly.


#### Pretrained Models
You can downlaod our pre-training models from [[Google Drive]](), Pre-trained models for all tasks will be published in the future.

### Evaluation
Before performing the following steps, please download our pretrained model first. To evalute our method on image restoration, please modify the benchmark path and model path. 
You need to modify ```test.py and datasets.py``` according to your environment and then

```bash
cd CycleRDM/image_ir
python ir_test.py 
```



### Results

![CycleRDM](https://github.com/hejh8/CycleRDM/tree/main/figs/compare.png)

<details>
<summary><strong>All Degradation Tasks </strong> (click to expand) </summary>

![CycleRDM](https://github.com/hejh8/CycleRDM/tree/main/figs/fig1.png)

</details>

<details>
<summary><strong>Image Restoration Tasks</strong> (click to expand) </summary>

![CycleRDM](https://github.com/hejh8/CycleRDM/tree/main/figs/ir.png)

</details>

<details>
<summary><strong>Low-light Image Enhancement Tasks</strong> (click to expand) </summary>

![daclip](https://github.com/hejh8/CycleRDM/tree/main/figs/low-light.png)

</details>


#### Notice!!
🙁 In testing we found that the current pretrained model is still difficult to process some real-world images  which might have distribution shifts with our training dataset (captured from different devices or with different resolutions or degradations). We regard it as a future work and will try to make our model more practical! We also encourage users who are interested in our work to train their own models with larger dataset and more degradation types.


---

**Acknowledgment:** Our CycleRDM is based on [CFWD](https://github.com/hejh8/CFWD) and [DiffLL](https://github.com/JianghaiSCU/Diffusion-Low-Light). Thanks for their code!



### Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```



```

---

#### Contact
If you have any question, please contact us.

#### --- Thanks for your interest! --- ####


