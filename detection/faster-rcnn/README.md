# Object detection methods

## Single object bounding box via. regression.
Dataset consists of images labeled with class and bounding box.

    $x = img$
    $y = (class), (x0, y0, width, height)$
    
The image is passed into a CNN and features are extracted. Then fully connected layers to predict class (classification), and seperate fully connected layers to predict bounding box parameters (regression).
