# Object detection methods

## Single object bounding box via. regression. (Classification with localization)
Dataset consists of images labeled with class and bounding box.

    x_i = img
    y_i = (class), (x0, y0, width, height)
    
The image is passed into a CNN and features are extracted. Then fully connected layers to predict class (classification), and seperate fully connected layers to predict bounding box parameters (regression).
Loss function for bounding box prediction is given by Intersection-Over-Union (IoU).

    J = Intersection(box_hat, box) / Union(box_hat, box)

This method is only able to detect and single object in an image.

## Multiple object detection with sliding window CNN.
A set of window sizes is chosen (n1xn1, n2xn2, n3xn3). The window sizes are smaller than the actual image. Each of the windows are slided over the image and for each slide a CNN classifies the content of the window. This happens for all window sizes. The first windows that classifies as an object are chosen as bounding boxes.


## Multiple bounding boxes and classes (RCNN - Region CNN)
Consists of two parts, a **feature extractor** (typically pre-trained model like ResNet, VGG, Inception, etc.) and a **region proposal network**.
