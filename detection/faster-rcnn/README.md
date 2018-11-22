# Object detection methods

## Single object bounding box via. regression.
Dataset consists of images labeled with class and bounding box.

    x_i = img
    y_i = (class), (x0, y0, width, height)
    
The image is passed into a CNN and features are extracted. Then fully connected layers to predict class (classification), and seperate fully connected layers to predict bounding box parameters (regression).
Loss function for bounding box prediction is given by Intersection-Over-Union (IoU).

    J = Intersection(box_hat, box) / Union(box_hat, box)

This method is only able to detect and single object in an image.
