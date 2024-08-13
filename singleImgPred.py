import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image


def cornResNet(img):
    model = load_model(r"C:\Users\dhruv\OneDrive\Desktop\web application\saved_model\corn_resnet.h5")
    img_path = 'C:/Users/dhruv/OneDrive/Desktop/web application/images/'+img
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 

    predictions = model.predict(img_array)

    class_names = ['Blight','Common Rust','Grey Leaf Spot','Healthy']
    predicted_class = np.argmax(predictions)
    predicted_label = class_names[predicted_class]
    return predicted_label