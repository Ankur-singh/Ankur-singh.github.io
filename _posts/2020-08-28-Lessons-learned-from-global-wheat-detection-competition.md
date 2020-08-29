---
comments: true
title: Lessons learned from global wheat detection competition
tags: [Object detection, Pytorch, Kaggle, writing better code]
style: border
color: danger
description: Everything that goes into winning a medal (bronze) in Kaggle competition. Data augmentation, reading research papers, writing good and modular code for faster experimentation, and much more.  
---

I spent my last month (July 2020) working on Global Wheat Detection competition hosted on Kaggle. I love computer vision and the dataset was small (~615 MB). 

#### About the competition
In this competition, the task was to detect wheat heads from outdoor images of wheat plants, including wheat datasets from around the globe. However, accurate wheat head detection in outdoor field images can be visually challenging. There is often overlap of dense wheat plants, and the wind can blur the photographs. Both make it difficult to identify single heads. Additionally, appearances vary due to maturity, color, genotype, and head orientation. Finally, because wheat is grown worldwide, different varieties, planting densities, patterns, and field conditions must be considered. Models developed for wheat phenotyping need to generalize between different growing environments.

You can read more [about the competition](https://www.kaggle.com/c/global-wheat-detection/) on Kaggle.

---

The very first thing that I do when participating in a Kaggle competition is to look for a starter notebook. Often the selection is based on the time required to get it up and running locally. So once I find a good notebook, I try to recreate it. Simply download it, install the dependencies, update the paths and then `Run All`.  For this competition, I started with this notebook.

These models can take a couple of hours to train. So, you have a lot of time to learn and explore once you get your starter notebook running locally. 

#### Philosophy time

Generally speaking, in Kaggle competitions, your chances of winning are determined by the number of experiments you conduct. looks pretty simple in theory but is very complex in practice. each iteration has to be quick, you need to write modular code, testing your code, reproducible, etc.  

**Experiments to increase CV and LV scores**

Most of the experiments can be grouped in the following : 

- Preprocess and Feature Engineering
- Data Augmentation and External Datasets
- Model Architecture, Loss, etc
- Training Schedule, Optimizer, etc
- Postprocess

#### Back to the competition

Reading the starter notebook. When I am reading any code for the first time, my goal is not to understand it completely. Instead, I try to identify what each part of the code is doing. For example, which part is responsible for creating the dataset?, where the model is created?, How is loss calculated?, etc. 

## Data augmentation

The very first thing that I did after reading the starter code was to learn data augmentation for object detection. The starter notebook already had an implementation for *flip* transform using albumentations library. So, it was very easy to get started with data augmentation. 

I found the following resources useful.

- [Data Augmentation for bounding boxes](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)
- You can find the complete list of augmentations offered by albumentations library, [here](https://github.com/albumentations-team/albumentations#documentation)

The next step was to find out what augmentation other people were using. So, I referred to a lot of kernels and discussions and conducted some small experiments. Finally, I came up with the following list of transforms 

```python
def get_train_transforms(img_sz):
	return A.Compose(
			[
				A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
		    A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
		             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
		    A.ToGray(p=0.01), 
			  A.HorizontalFlip(p=0.5), 
			  A.VerticalFlip(p=0.5), 
			  A.Resize(height=img_sz, width=img_sz, p=1),
		    ToTensor()
			], 
		  p=1.0, 
		  bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels'])
	)
```

#### Other novel techniques

While reading other kernels for data augmentation, I realized everyone was using *Mosaic*. It is a technique where we randomly select 4 images and stitch them together to make a single image. But as soon as I started digging down the rabbit hole, I found out other novel techniques (like mixup and cutmix) for image augmentation. 

This is how the augmented images looked like.

![https://i.imgur.com/waxJLOR.png](https://i.imgur.com/waxJLOR.png)

**Note:** You can find all the code for implementing these techniques, here [https://www.kaggle.com/ankursingh12/data-augmentation-for-object-detection](https://www.kaggle.com/ankursingh12/data-augmentation-for-object-detection). I have also provided links to some research papers for understanding these techniques better.

Initially, when experimenting, I trained each model for only 10 epochs. I think it is good enough training to decide if something will work or not. Clearly, if new augmentation techniques help your model performs better in the first 10 epochs then it will give you an even better score on further training. So, don't waste a lot of time training models for 40 epochs. 

**Results:**  After applying all these data augmentation techniques, my LB score increased to 0.66 (from 0.62). This confirms that these techniques work and I should use them with every model (unless you have a very strong reason to not use them).

## Model Architecture

My LB score was — and the highest score was 0.72. This is a huge gap and no high performing submission was using the Faster-RCNN model. This means I need to look for other model architecture. I knew there are many other models like YOLO and SDD for object detection but I was really concern about their training time. Then I came across this image in one of the kernels.

![https://i.imgur.com/JcOq8zQ.jpg](https://i.imgur.com/JcOq8zQ.jpg)

As you can see EfficientDet is very small in size and still gives much better scores. This is exactly the kind of model I was looking for because a small model means fast training, allowing me to conduct more experiments. So, I decided to use EfficientDet.  

After doing some research, I realized that EfficientDet is based on EfficientNet. So, I started by reading [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf) paper followed by [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) paper. I didn't understand the papers completely but going through them gave me a brief idea about why they are different and powerful. Lucky for me, there is already a [pytorch implementation of EfficientDet](https://github.com/rwightman/efficientdet-pytorch) available on Github. I read [this kaggle](https://www.kaggle.com/shonenkov/training-efficientdet) notebook to see how to use this library. 

This model was pretty big, with 12 GB of gpu RAM I can only load 1 image at a time. So, I read and implemented **gradient accumulation**. [This particular](https://www.kaggle.com/c/global-wheat-detection/discussion/168199) discussion help me a lot, as it was my first time implementing it. Another thing I tried was **Automatic Mixed Precision (AMP)** to reduce model size.

This is another reason why I love kaggle. Its a great place for practitioner. You can find working code for a lot of obscure libraries, learn the best practices in industry & see what Grand Master's do differently. Whether you win the competition or not, you will end up learning!

**Results:** After replacing the Faster-RCNN model with EfficientDet-B5, my LB score jumped to 0.71 (from 0.66). It was a huge improvement.

## Refactoring

By this time, it was a pretty long colab notebook. Every time I need to run the code, I would have to `Shift+Enter` more than 30+ cells. Most of the time, I was training multiple models in separate notebooks. So, I had to `Shift+Enter` another 30 more times. So, I decided to make **python scripts our of these notebooks**, host them on Github, and clone them in colab. Now instead of running 30 cells, it was 3-4 cells. 

When I converted my notebooks into scripts, I faced another challenge. The script was 350+ lines of code. I used a lot of global variables, very few functions with strong assumptions about the input, etc. Long story short, it was a messy code. It was hard to manage and any attempt to change the code would break it. But again, Kaggle competitions are all about experiments. So, I need to write code that is modular, well organized, and scalable. 

I started by breaking down my code into different python files. Separate file for preparing and loading data, building model, and training model. Then I took away a lot of general code and placed them in `[utils.py](http://utils.py)` and finally wrote some test-cases. It took me a lot of time. Also, the result was many more lines of code that were spread across multiple files. But on the contrary, It was much more manageable. Previously, what would take an hour for me to implement and integrate, has dropped to just 15 mins. These things alone were worth the effort. 

**Note**: Writing test-cases if often seen as a burden. But trust me, it helps a lot in the long run and can help you reduce the development time to half. During the complete process of refactoring, I would immediately know if something broke because of the test-cases. Often these models can take hours to train and realizing that you forgot to call `torch.save()` can waste a lot of precious time and compute. START WRITING TEST-CASES.

During this time, my team-mates were testing *yolov5* architecture. They were using the code-base from [this](https://github.com/ultralytics/yolov5) git repository by ultralytics. Because of the license, *yolov5* was disqualified. But none the less, it was a great learning experience. The code is very well written and useable. I would recommend everyone working in computer vision to read it. The *yolov5's* training file was very configurable. I supported a lot of arguments, with easy to understand names. So, I also added the functionality to pass arguments to my script as well.  

All these changes made the code much more customizable and reusable. 

I trained 5 different EfficientDet models and did 1 round of pseudo labeling on all of them. This gave me 0.738 score on LB, pretty good, right? 

Life was good. But towards the end of the competition, I realized a lot of other people are getting much better scores using the same model. So I started looking for things they are doing differently, by asking questions on discussions, read more kernels, etc. I realized that many of them were using different values of learning rates and different types of schedulers. It was the last week of the competition and I didn't want to leave any stone unturned. So, I started trying different schedulers and learning rates. While doing so, I realized that my training loop has a lot of boilerplate code, like code to save last and best checkpoint, Automatic Mixed Precision (AMP), gradient Accumulation, calculating time per epoch (train & valid step separately), progress bar, warmup, writing logs; you got the idea. 

I already knew about [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/) and I have already used it in small projects. It is actually an amazing library that takes care of all the boilerplate code so that you can focus on actual code that matters. This is exactly what I wanted for now. So, I ported my code to the format that PyTorch-lightning expects. Previous refactoring efforts helped me here. Since the code was already formatted into separate functions, it just took me a couple of hours to port everything to PyTorch-lightning. 

My code was reduced from 400+ lines of code to just 120 lines. Plus, you get all the above-mentioned things for free. Nice looking progress bar, support for gradient accumulation, AMP, etc. can be easily controlled using arguments. Less code means fewer chances of error and more manageable code. And finally, the best feature, `fast_dev_run`. It runs a “unit test” by running 1 training batch and 1 validation batch. The point is to detect any bugs in the training/validation loop without having to wait for a full epoch to crash or find a bug. It is a lifesaver, trust me!

I know its a lot to digest and might look overwhelming to some. Trust me, it is okay! I didn't learn all these things in a day, it was a month-long process. Multiple late nights and early mornings. It was all worth it because I learned the most from this part of the journey.

## Inference

Parallelly, on the inference side, I used the following techniques: 

- **Test Time Augmentation (TTA) :** In TTA, we perform random modifications to our test images. Thus, instead of showing original test images, only once to the trained model, we will show it the augmented images several times. We will then average the predictions of each corresponding image and take that as our final prediction.
- **Weighted Boxes Fusion (WBF) :** Its a way of ensembling bounding boxes from object detection models. I used WBF implementation from [this git repo](https://github.com/ZFTurbo/Weighted-Boxes-Fusion).

You can find the inference kernel here: [https://www.kaggle.com/ankursingh12/efficientdet640-inference](https://www.kaggle.com/ankursingh12/efficientdet640-inference)

Multiple things were going on in parallel (like training model, making predictions, refactoring, etc.) but it is very difficult to write it all down in a linear fashion as a blog. But I really hope that you can understand it.

## Configuration Management

As I already said, toward the end of the competition many people were using different learning rates, optimizers and schedulers. I wanted a way to control optimizer and scheduler from the command line, by passing arguments. My first approach was to use if-else condition or create a dispatch dictionary for optimizers and schedulers. Both of these options were not scalable i.e. for each new optimizer or scheduler I have to either write a new condition or create a new key-value pair. I finally found a very helpful function that would take the path of the optimizer or scheduler in the module and load it. 

This is what the function looks like. 

```python
import importlib

def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)

## Usage
optim = load_obj('torch.optim.Adam')    # this will return 'Adam' optimizer from 'torch.optim'
optim = load_obj('torch.optim.SGD')     # this will return 'SGD' optimizer from 'torch.optim'
optim = load_obj('torch.optim.RMSprop') # this will return 'RMSprop' optimizer from 'torch.optim'
```

Now, I can pass any optimizer or scheduler without writing any extra line of code.

Finally, as I started configuring more and more things, it was getting difficult to handle everything with python's built-in Argparser. So I switched to *OmegaConf.* Its are a great library for managing configuration. You can read more about it [here](https://omegaconf.readthedocs.io/en/latest/). Config files are great, this was the first time I used them. You should google "configuration management in python" and you will find many great libraries like hydra. I would highly recommend everyone to start using config files for your projects, they make it super easy for others to use your code without actually worrying about the internals.  

## Summary

If you look at the first part of the blog where I talked about different types of experiments that you can conduct. Most of the things in this article fits perfectly in it.

**Experiments to increase CV and LV scores**

- Preprocess and Feature Engineering
    - Resizing images
- Data Augmentation and External Datasets
    - Albumentations
    - CutMix, Mosaic, etc
- Model Architecture, Loss, etc
    - Faster-RCNN, YOLO and EfficientDet
- Training Schedule, Optimizer, etc
    - Different Schedulers and optimizers
    - Gradient Accumulation
    - Automatic Mixed Precision (AMP)
- Postprocess
    - TTA
    - WBF ensemble

In each stage, many more possible experiments can be conducted like using external data, other architectures, etc. I think sticking to this list and doing a lot of experiments is one of the many ways of winning a kaggle competition. 

My team completed the competition with 0.746 score on Public LB. Our rank was 600+, but I knew many solutions before us were not complying with the rules of the competition and they all will be eliminated. Plus, many solutions overfit the public LB. So, anything can happen when private LB scores are out. Upon competition completion, our public LB rank was 127 and private LB rank was 193. 

This was my first serious competition on Kaggle and I ended up winning my first bronze medal. I will always remember this journey and also this competition. Thanks for sticking till the end. This was a pretty long article, so I really hope that you learned something from it.