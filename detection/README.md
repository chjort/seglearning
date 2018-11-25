# Object detection methods

Datasets for object detection consists of images (X) and labels (Y). The labels are bounding boxes + class.

Classification+Detection examples:

* http://host.robots.ox.ac.uk/pascal/VOC/voc2012/examples/index.html
* http://www.image-net.org/download-bboxes

## Single object bounding box via. regression. (Classification with localization)
Dataset consists of images labeled with class and bounding box.

    x_i = img
    y_i = (class), (x0, y0, width, height)

The image is passed into a CNN and features are extracted. Then fully connected layers to predict class (classification), and seperate fully connected layers to predict bounding box parameters (regression).
Loss function for bounding box prediction is given by Intersection-Over-Union (IoU).

    J = Intersection(box_hat, box) / Union(box_hat, box)

This method is only able to detect and single object in an image.

## Multiple object detection with sliding window CNN.
A set of window sizes is chosen (n1xn1, n2xn2, n3xn3). The window sizes are smaller than the actual image. Each of the windows are slided over the image and for each slide a CNN + FC classifies the content of the window and regress the bounding box. This happens for all window sizes. The first windows that classifies as an object are chosen as bounding boxes.

This method is computationally expensive, and bounding boxes are not that accurate.


## More effective multiple object detecion with Selective Search.
Consists of a region proposal algorithm, and a CNN + FC to classify label and regress boxes for each region.

The region proposal algorithm (SS) works by bottom-up hierarchichal clustering of pixels based on color, texture, size, and shape. Regions are merged from bottom up until a specific amount of RoIs are found. Each regions is then classified and bounding box predicted.


## R-CNN **###TODO###**
Consists of two parts, a **region proposal network (RPN)** and a CNN + FC to classify label and regress boxes.

The RPN replaces the region proposla algorithm by instead _learning_ which regions to classify. These regions are called Regions of Interest (RoIs). Each region is passed into the CNN + FC to classify and regress boxes.

Test
