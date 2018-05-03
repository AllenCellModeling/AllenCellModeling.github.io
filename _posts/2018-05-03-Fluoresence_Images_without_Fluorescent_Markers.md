---
title: Fluorescence Images without Fluorescent Markers
summary: Using brightfield microscopy images to generate fluorescence marked images.
---
# Fluorescence images without fluorescent markers

## Background and motivation

Fluorescence microscopy is one of the fundamental imaging tools we can use to see what is happening inside cells. The ability to attach fluorescent labels to specific proteins and structures of interest within cells has been nothing short of revolutionary in the field of cell biology, as with the ability to see what cell components are doing, also allows us to learn, piece by piece, how cells function. 

### The challenges of fluorescent imaging 

However, no technique is without its drawbacks and limitations. 
- Fluorescent markers can change the behaviors of the protein they are attached to, preventing normal interactions and regulatory processes. 
- The intense illumination needed to excite fluorescent markers causes photo-damage to the imaged cells. This damage ranges from potentially unnoticeable to a degree sufficient to cause cell death. This effect is worse when more illumination is used (as when there are relatively low levels of expressed protein) and worse when the fluorescent tag is primarily located on or adjacent to DNA (as with dyes that visualize the nucleus). 
- There are literal bandwidth limitations with fluorescent markers: you only have so many wavelengths you can use excite different tags and only so many wavelengths you can use to look at output. 

We are looking for ways we can side-step these limitations of fluorescent markers.

### Maybe labeled structures can be seen without fluorescent labels?

Fluorescent markers specifically highlight the structures of interest to make them easier to see, but these exist in the absence of markers. Large scale structures, such as the nucleus, are visible in traditional bright-field illumination; one can even manually segment or highlight them, given the time. 

![Bright-field with nuclear dye vs hand labeling](https://i.imgur.com/0IDg0yx.png)

However, it takes way too much time to do this for all the images we want to see, and the hand-labeled image isn't as clear as the dye-labeled one. Can we train a model to do this highlighting step for us, by feeding it the bright-field image as an input and the dye-labeled image as a target? 

A category of convolutional neural-net models, those based on the [U-net architecture](https://arxiv.org/abs/1505.04597), have shown themselves to be capable of biological image segmentation and style transfer. These model architectures reduce an image's dimensions, while detecting more and more features that can be used to classify whether a pixel would be fluorescent if it were labeled with a marker. After this categorization, the image is upsampled to its original resolution, with contextual information passed from the downsampling layers to provide context, increasing the realism of the upsampling (and placing the categorization in its original context).

![Placeholder unet architecture](https://raw.githubusercontent.com/zhixuhao/unet/master/img/u-net-architecture.png)

### Our contributions

We've trained such models on paired sets of our fluorescent images and the corresponding brightfield images. The details of our architecture and approach are described [https://www.biorxiv.org/content/early/2018/03/27/289504](in more detail in this pre-print), but let's take a look at how well structure prediction works, how close we are to being able to acquire fluorescence-like images without fluorescent markers.

## Predicting fluorescence images from bright-field images

### Getting support packages, the model, and input images

This notebook runs well on [Google's Colaboratory Environment](https://drive.google.com/file/d/1aXtGzmqXxKTrraVZu0eg7JBf9Zt4iFFE/view?usp=sharing), which gives you access to nice GPUs from your laptop, but we'll need to install some packages and get local copies of the trained model and some input images to test it on.


```python
## Package installation, restart runtime after packages install

# Image read and write
!pip install -q imageio

# Data and model downloading
!pip install -q quilt --user

# Model architecture and image pre-processing
# !pip install -q git+https://github.com/AllenCellModeling/pytorch_fnet.git
!pip install -q git+https://github.com/JacksonMaxfield/pytorch_fnet.git

# PyTorch to run the model 
!pip install -q http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
```


```python
# First the standard packages
import numpy as np
import json
import matplotlib.pyplot as plt
plt.set_cmap('gray')

# For image data
import scipy.ndimage
import tifffile
```


With our environment set up, we can download sample images of our cell lines.


```python
# Download sample images from each of our cell lines
import quilt
quilt.install('aics/cell_line_samples')

# Load the cell line samples dataset
from quilt.data.aics import cell_line_samples
```


    Downloading 10 fragments (1726844072 bytes before compression)...


    100%|██████████| 1.73G/1.73G [01:21<00:00, 21.1MB/s]


Let's take a look at the image we downloaded. Like all of the images in the `cell_line_samples` dataset, it is three dimensional. These images are captured at a series of heights, optically slicing the cell into a z-stack of images. When we display them, we'll take the maximum along the z-axis, the max project, since 3D rendering is hard. 

It has several channels:


```python
# Make plotting max projections easier
def max_project(ax, img):
  max_img = img.max(0)
  max_scale = np.percentile(max_img, 99)
  ax.imshow(max_img, vmax=max_scale)
  ax.axis('off')

# Load the downloaded image into an array
img_fn = cell_line_samples.AICS_13_1165_ome()
img = tifffile.imread(img_fn)

# Show each of the channels we're interested in
channels = {'cell membrane':0, 'lamin':1, 'dna':2, 'bright-field':3}

fig, axes = plt.subplots(1, 4, figsize=(14, 6))
for channel, ax in zip(channels.items(), axes):
  channel_name, channel_index = channel
  ax.set_title(channel_name)
  max_project(ax, img[:,channel_index])
plt.tight_layout()
```


![png](../assets/nbfiles/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers_9_0.png)


What information does each of these channels contain?

* The cell membrane channel captures information from a fluorescent dye (CellMask) used to image this structure
* The lamin channel includes the information from imaging an endogenous fluorescent tag localized to the nuclear envelope (Lamin B1)
* The DNA channel shows the information from a fluorescent dye (Hoechst) used to image this structure
* Bright-field is a non-fluorescence image of the scene (note the inherent constrast but much lower specificity for the structures represented in the three previous channels)

We've claimed that our model has learned how to predict the fluorescence channels from the bright-field image. Can we show that in action? Let's download and read in our model weights:


```python
import torch
import fnet # our implementation of u-net for fluorescent prediction
import fnet.transforms

# Download trained model weights, find the file that predicts the DNA channel
quilt.install('aics/label_free')
from quilt.data.aics import label_free
dna_model_fn = label_free.dna.model()

# Load the weights into our model
dna_model = fnet.fnet_model.Model()
dna_model.load_state(dna_model_fn, gpu_ids=0)
```


    Downloading 18 fragments (1686753606 bytes before compression)...


    100%|██████████| 1.69G/1.69G [01:21<00:00, 20.6MB/s]


### Predicting fluorescent images from brightfield images

Now that our model is loaded, let's prepare the previously loaded image for use with our model.

In this first example, we are going to predict the DNA fluorescent dye image.


```python
# Prep the image for input into the model
full_brightfield = img[:, channels['bright-field'],:,:]
dna_opts = json.load(open(label_free.dna.train_options()))
small_brightfield = fnet.transforms.prep_ndarray(full_brightfield, dna_opts['transform_signal'])
model_input = fnet.transforms.ndarray_to_tensor(small_brightfield)
```


Great! Our image is prepped; time to predict the DNA fluorescent dye image.


```python
# Predict the dna structure and convert back to ndarray
model_output = dna_model.predict(model_input)
predicted_dna = fnet.transforms.tensor_to_ndarray(model_output)
```

The output image has a lower resolution (makes the model easier to fit into memory).

To compare the observed DNA fluorescent dye image with our predicted DNA fluorescent dye image, we prepare the observed image for output as well.


```python
# Two compare the two images, prep the original
full_dna = img[:, channels['dna'],:,:]
small_dna = fnet.transforms.prep_ndarray(full_dna, dna_opts['transform_signal'])
```


```python
# Display the observed and the predicted
fig, axes = plt.subplots(1, 2, figsize=(14,6))

axes[0].imshow(small_dna.max(0))
axes[0].set_title('observed')
axes[0].axis('off')

axes[1].imshow(predicted_dna.max(0))
axes[1].set_title('predicted')
axes[1].axis('off');
```


![png](../assets/nbfiles/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers_18_0.png)


### Visualizing the difference between the observed and predicted images

We'd like a better sense of how our prediction differs from the observed image than we can get from the above two images at a glance. Let's compare them in a couple of other ways. The simplest comparison is simply to toggle back and forth between the two images. This lets your eye pick out differences that aren't apparent when you have to move from one image to the next. 

This may be helpful in finding areas where our prediction is noticeably off.


```python
# Create a gif of observed to predicted
from IPython.display import Image
import imageio

ordered_imgs = [small_dna.max(0), predicted_dna.max(0)]
imageio.mimsave('compare.gif', ordered_imgs, duration=2)

with open('compare.gif', 'rb') as f:
  display(Image(data=f.read(), format='png'))
```


![png](../assets/nbfiles/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers_20_1.png)


To my eye this shows that our predictions are largely accurate in location, with some underprediction of the extent of the DNA dye uptake by the mitotic cell. Also of interest is that our predicted image is a bit smoother or less grainy than is the observed image. We can put this down to stochastic uptake of dye or dye clumping that the model is unable to predict.

What is the absolute difference between our observed image and our predicted image? Let's take a look: 


```python
# Absolute difference
dna_diff = abs(small_dna - predicted_dna)

# Display the observed DNA, the predicted DNA, and their absolute difference
images = {'observed': small_dna,
          'difference': dna_diff,
          'predicted': predicted_dna}

# Normalize the greyscale intensities by finding the max/min intensity of all
max_gs = max([i.max(0).max(0).max(0) for l, i in images.items()])
min_gs = min([i.min(0).min(0).min(0) for l, i in images.items()])

# Display the images with appropriate normalization
fig, axes = plt.subplots(1, len(images), figsize=(18, 6))
for label, ax in zip(images.keys(), axes):
  ax.imshow(images[label].max(0), vmax=max_gs, vmin=min_gs)
  ax.set_title(label)
  ax.axis('off')
```


![png](../assets/nbfiles/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers_22_0.png)


An ideal prediction would have a difference image that is completely empty (black).

While our difference image indicates that we didn't predict exactly the same image as the observed image, this image also gives us more confidence that we are able to predict location reasonably well at first pass. Notably, visible error looks like it correleates with areas where the observed image has signal. This means we aren't getting the magnitude quite right, but we *are* getting the location of the fluorescence right.



That emphasizes the relative accuracy in location of the fluorescent marker. But what about the intensity error? It is hard to get a sense of magnitude from a colorscale. To build intuition about error magnitude, let's look at the observed image, the difference, and the predicted image side-by-side with pixel intensity mapped to height in a surface plot.


```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Update the max greyscale by first taking the sum across z
max_gs = max([i.sum(0).max(0).max(0) for l, i in images.items()])

# Plot each image by creating a mesh and setting the new z axis limits
fig, axes = plt.subplots(1, len(images), figsize=(18,6), subplot_kw={'projection': '3d'})
for label, ax in zip(images.keys(), axes):
  x = np.arange(images[label].shape[2])
  y = np.arange(images[label].shape[1])
  x, y = np.meshgrid(y, x)

  mag_sum = np.sum(images[label], axis=0)
  mag_sum = np.rollaxis(mag_sum, 1, 0)
  mag_sum = scipy.ndimage.gaussian_filter(mag_sum, 1) #else single pixel error predominates

  ax.plot_surface(x, y, mag_sum, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=1)
  
  ax.set_zlim([0, max_gs])
  ax.set_title('sum greyscale intensity\nacross z-stack of ' + label)
  ax.xaxis.pane.set_edgecolor('black')
  ax.yaxis.pane.set_edgecolor('black')
  ax.set_facecolor((1.0, 1.0, 1.0, 1.0))
```


![png](../assets/nbfiles/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers_25_0.png)


The middle surface plot would be completely flat for an ideal model. Note that the largest difference in intensity is at the mitotic cell, which is also where we have the largest intensity in the observed image. One of the other interesting things that pops out of this is the 'noise' overlaid on the error. The model is unable to predict what is likely detector noise.

### Predicting structures not tagged

These error plots give us some confidence in our ability to predict where DNA fluorescence will appear. We can plot simmilarly estimate error for our other tag models, but this is left as an exercise for the reader. Hard values for how well each of these models work are found in [the accompanying paper](https://www.biorxiv.org/content/early/2018/03/28/289504). 

Instead let's end with an estimate of where the nuclear envelope is for several images.


```python
# Get the model file
lamin_model_fn = label_free.lamin_b1.model()
lamin_opts = json.load(open(label_free.lamin_b1.train_options()))

# Load the weights into our model
lamin_model = fnet.fnet_model.Model()
lamin_model.load_state(lamin_model_fn, gpu_ids=0)

# Function to predict an output image using the model provided
def predict_img(img, model, transforms):
  # Prep the image for input into the model
  full_brightfield = img[:, 3,:,:]
  img_in = fnet.transforms.prep_ndarray(full_brightfield, transforms)
  model_input = fnet.transforms.ndarray_to_tensor(img_in)
  
  # Predict the lamin structure and convert back to ndarray
  model_output = model.predict(model_input)
  predicted_out = fnet.transforms.tensor_to_ndarray(model_output)
  
  return predicted_out
```


```python
# Generate predictions for each input brightfield image
predictions = dict()

for img_name in cell_line_samples._data_keys():
  if 'ome' in img_name:
    img_fn = getattr(cell_line_samples, img_name)()
    img = tifffile.imread(img_fn)
    prediction = predict_img(img, lamin_model, lamin_opts['transform_signal'])
    predictions[img_name] = prediction
```


```python
# Display each nuclear envelope prediction
fig, axes = plt.subplots(2, int(len(predictions) / 2), figsize=(14,6))
for img_name, ax in zip(predictions.keys(), axes.flat):
  ax.imshow(predictions[img_name].max(0))
  ax.set_title(img_name)
  ax.axis('off')
```


![png](../assets/nbfiles/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers/2018-05-03-Fluoresence_Images_without_Fluorescent_Markers_30_0.png)


What a nice way to get information from a brightfield image using patterns of structures we've seen in other images. We can see fluorescent structures in cells where we haven't even labeled them. It is hard to overstate how much this can improve the interpretability of large corpora of brightfield images in cases where we can do a little initial dying or labeling.

