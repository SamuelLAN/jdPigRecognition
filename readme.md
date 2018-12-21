# Pig Face Recognition of JDD-2017 Competition

> link: http://jddjr.jd.com/item/4

### This project includes three processes.

- video_process

    Convert the videos into image data.

    Use data augmentation to enrich training set. It includes rotatation, translation, scaling, adjusting brightness and chroma, and etc.

    ( Please see the folder [video_process](video_process) and [readme](video_process/readme.md) for details.)

- fcn

    Process image data and segment the pigs individually.

    Use data augmentation to enrich training set. The pigs segmented would be rotated, translated, adjusted brightness and chroma, and etc.

    ( Please see the folder [fcn](fcn) and [readme](fcn/readme.md) for details.)

- classify

    To classify each pigs out of images.

    Use the pigs segmented by FCN as input to the classification model.

    ( Please see the folder [classify](classify) and [readme](classify/readme.md) for details.)

<br>

>##### Environment
> 1、Python environment: This code is compatible with Python 2.7, 3.5, 3.6.
>
> 2、Some Python libraries: Tensorflow 1.0+、numpy、six

<br>

>##### Caution
> 1、I received a feedback that the running of FCN would result in insufficient memory storage. There is an explanation here. The problem does not occur on my computer since my computer's memory storage is large enough: the memory storage of my computer is 16G. If the memory storage is not large enough, such as 8G, you can modify the file "load.py" inside the folder fcn and change the data loading method to on-demand loading. It is recommended to refer to the data loading method in file "classify/load.py", which applies asynchronous on-demand loading method. This method can solve the memory storage shortage problem and still keep the fast speed.

<br>

>##### Competition ranking
> But the <strong style="color: red; font-weight: normal;">log_loss</strong> of the test set, the result of classify, can reach <strong style="color: red; font-weight: normal;">0.32</strong>, which was in the top 20 and eligible to enter the finals. <br><br>Unfortunately, I made mistakes in saving the model and handed in a wrong model, leading to the result of ranking 101. After I found out this problem, it was too late to submit the correct model.

<br>

>#### Feelings and thoughts
> When I signed up for this competition, half of the time had already passed. Plus, due to the inappropriate allocation of time in the former phase, there was no enough time left for me to train the model and debug problems.
>
> In general, there were several problems.
>
> 1、I do not read enough papers and need to further enrich my minds.
>
> 2、The allocation of time was not appropriate. I have wasted most of my time on the segmentation of pigs, instead of other sections which were more importance, such as the classification section.
>
> At first, I tried hard to avoid manually tagging data and to apply unsupervised method to solve problems. However, it turned out that it was a waste of time and meanless. Although I turned to apply FCN model as soon as I found out this problem, lots of time had already been wasted.
