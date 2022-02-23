# Synthetic data project
Synthetic data for traing deep neural networks for object detection on traffic signs.

## Bugs and todos
* `dataset.py` : `self.classes` cannot be of type `None`
* TODO: Eval
* dataloader class
* hydra management
* wandb?
* I should remove the Panorama images
* dtype in dataset
* train, eval functions-file
* train, eval should return something
* add criterion?
* load images as tensor
* fix tqdm
* fix warning:
```
:\Users\saibo\anaconda3\envs\BA\lib\site-packages\torch\functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ..\aten\src\ATen\native\TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
```


## Requirements
Virtual environment and dependencies with `conda`:
```bash
conda create --name [name] python=3.9
```

Install requirements (in virtual environment): 
```bash
pip install -r requirements.txt
```



# Mapillary Traffic Sign Dataset

The Mapillary Traffic Sign Dataset (MTSD) contains bounding-box annotations for traffic sign detection and class labels for traffic sign classification.

Detailed descriptions: https://arxiv.org/abs/1909.04422

### Demo Script

For how to load and show images and annotations:
```bash
python visualize_example.py
```

## Data Format

3 directories:

### Splits

3 text files such that each file defines the split where each line corresponds to an image key, which corresponds to an image and an annotation. Example: `splits/train.txt`

### Images

3 directories for each split containing the images.
Example: `images/train/Bh36Ed4HBJatMpSNnFTgTw.jpg`

### Annotations

Contains annotation files where each filename corresponds to an image key.
Example: `annotations/Bh36Ed4HBJatMpSNnFTgTw.json`

The annotations are stored as JSON with the following keys:

 - `width`: the width of the corresponding image
 - `height`: the height of the corresponding image
 - `ispano`: a boolean indicating if the image is a 360° panorama image
 - `objects`: a list of traffic signs in the image

 Each object is itself a dictionary with the following keys:

  - `bbox`: defining the bounding box of the traffic sign within the image.
  - `key`: a unique identifier of the traffic sign.
  - `label`: the class of the traffic sign
  - `properties`: a dictionary defining special properties of the sign.


### Panorama handling

Panoramas are stored as standard images with equirectangular projection and can be loaded as any
image.

For the special case of traffic signs that extend across the image boundary of a panorama (`xmin > xmax`),
we include a dictionary `cross_boundary` in the `bbox` defnition containing the `left` and the `right` crop
of the bounding box. In order to extract the full image crop of the traffic sign, one can crop `left` and
`right` separately and stitch them together.

Example: `annotations/OENb8BfFyAocFzHHM4Mehg.json`

### Partially annotated set
**Not a part of project**

The partially annotated set additional includes correspondance information for each object in
the `correspondance` dictionary:

  - `image_key`: the key of the image in the fully annotated set containing the corresponding traffic sign.
  - `object_key`: the key of the corresponding traffic sign object in the fully annotated set. 
