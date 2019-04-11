# Hand Mask R-CNN

This is an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of a hand in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone. The project is implemented using resources from [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).

## Getting Started

### Prerequisites

* [Python 3.6](https://www.python.org/downloads/)
* [Docker](https://docs.docker.com/install/) 

### Setup

Install python modules

```
pip install -r requirements.txt
```

## Traning

A pre-trained weights for MS COCO ([from matterport](https://github.com/matterport/Mask_RCNN/releases)) and hand is provided from the [releases page](https://github.com/theerapatkitti/hand_mask_rcnn/releases). Training and evaluation code is in [samples/hand/hand.py](samples/hand/hand.py). The module can be ran directly from command line:

```
# Train a new model starting from pre-trained COCO weights
python3 samples/hand/hand.py train --dataset=/path/to/dataset/ --model=coco

# Train a new model starting from ImageNet weights
python3 samples/hand/hand.py train --dataset=/path/to/dataset/ --model=imagenet

# Continue training a model that you had trained earlier
python3 samples/hand/hand.py train --dataset=/path/to/dataset/ --model=/path/to/weights.h5

# Continue training the last model you trained
python3 samples/hand/hand.py train --dataset=/path/to/dataset/ --model=last
```

The training schedule, learning rate, and other parameters should be set in samples/hand/hand.py.

## Evaluation

Evaluation code can be ran by using:

```
# Run evaluation on the specified model
python3 samples/hand/hand.py evaluate --dataset=/path/to/dataset/ --model=/path/to/weights.h5
```

The visualization of the model output can be seen by using [inspect_hand_model](scripts/inspect_hand_model.ipynb).

## Dataset

[Rendered Handpose Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) is used as a dataset to train the Hand Mask R-CNN. To make the dataset able to work with Python COCO tools, the annotation must be in COCO format. [RHD_to_COCO.ipynb](scripts/RHD_to_COCO.ipynb) converts RHD's annotation into COCO format.

## Creating Graph

To make the model able to be used in any language, the pre-processing and post-processing process is added to the graph by using [create_mask_rcnn_graph.ipynb](scripts/create_mask_rcnn_graph.ipynb).

## API

### Setup

Put SavedModel export from [create_mask_rcnn_graph.ipynb](scripts/create_mask_rcnn_graph.ipynb) into docker folder.

### Hand Mask Detection

The API returns bounding boxes and segmentation masks for each hand in the image. The response will have the form:

```
{
    "predictions": {
      [
        "roi": ...,
        "class_id": ...,
        "score": ...,
        "mask": ...,
      ],
      ...
    }
    "success": ...
}
```
To call API:

```
import requests

files = {"image": ("filename", open("path/to/image", "rb"), "mimetype")}

requests.post("http://<ip>:5000/predict", files=files)
```

## Built With

* [Keras](https://keras.io/)
* [Tensorflow](https://www.tensorflow.org/)
* [Docker](https://www.docker.com/)
* [Flask](http://flask.pocoo.org/)

## Citation

[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

[Rendered Handpose Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)

```
@TechReport{zb2017hand,
  author    = {Christian Zimmermann and Thomas Brox},
  title     = {Learning to Estimate 3D Hand Pose from Single RGB Images},
  institution    = {arXiv:1705.01389},
  year      = {2017},
  note      = "https://arxiv.org/abs/1705.01389",
  url       = "https://lmb.informatik.uni-freiburg.de/projects/hand3d/"
}
```