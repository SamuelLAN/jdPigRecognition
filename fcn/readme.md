# FCN: Fully Convolutional Networks for Pigs Segmentation

##### Responsible for segmenting the pigs out of the pictures, that is, removing the background and leaving only the pigs individually.

> The structure diagram of FCN.
>
> <img src="../tmp/fcn_graph.png" alt="The structure diagram of FCN" height="830" width="430">

> The scalar of Tensorboard of the training process of FCN.
>
> <img src="../tmp/fcn_scalar.png" alt="The scalar of Tensorboard of the training process of FCN" height="210" width="420">

> The image of Tensorboard of the training process of FCN.
>> Among them, the "input_image" is the input image, the "output_image" is the image of pigs after segmentation and the "truth mask" is the image of ground truth.
>
> <img src="../tmp/fcn_img_1.png" alt="The image 1 of Tensorboard of the training process of FCN" height="381" width="431">
>
> <img src="../tmp/fcn_img_2.png" alt="The image 2 of Tensorboard of the training process of FCN" height="381" width="431">
>
> <img src="../tmp/fcn_img_3.png" alt="The image 3 of Tensorboard of the training process of FCN" height="381" width="431">

>#### The screenshot of the CMD of the training process of FCN.
>
> <img src="../tmp/fcn_cmd.png" alt="The screenshot of the CMD of the training process of FCN" height="100" width="210">

>#### File structure.
- [load.py](load.py): The base class for loading data.
- [fcn.py](fcn.py): Fcn model. It inherits from "lib/base". You can train the model by running the function FCN.run.
- [get_image.py](get_image.py): It applies fcn.py to segment the pigs of data/TrainImg individually.
- [get_test_image.py](get_test_image.py): It applies fcn.py to segment the pigs of data/TestB individually.
