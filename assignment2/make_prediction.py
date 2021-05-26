from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

model_test = load_model("keras_trial_modelv5_vgg_overfit.h5")

def make_prediction(image_path:str, model):
    """
    image_path: path to image
    model: trained Keras model
    
    example:
    make_prediction('./recipes/CMNtMLPJO7e.png', model_test)
    
    returns: picture together with predicted outcome
    """
    # load image and resize to fixed size
    image = load_img(image_path, target_size=(224,224))
    # convert image to array
    image_input = img_to_array(image)
    # show image
    plt.imshow(image_input/255.)
    plt.show()
    # expand dimension to make prediction
    image_input = np.expand_dims(image_input, axis=0)
    # make prediction
    pred = model.predict(image_input)
    if np.squeeze(pred) > 0.5:
        print("#healthy")
    else:
        print('#healthy does not fit for this pic')