import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


tf.get_logger().setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print('Loading ...')

def load_models():
    model1 = load_model("Weights\stage1\TDCNN_1.h5")
    print('model1 loading complete')
    model2 = load_model("Weights\stage2\TDCNN_2.h5")
    print('model2 loading complete')
    model3 = load_model("Weights\stage3\TDCNN_3.h5") 
    print('model3 loading complete')
    return model1, model2, model3

def labels(n_stage):
    label = [0,0,0]
    label[0] = {
        'Healthy' : 0,
        'Unhealhy' : 1
    }
    label[1] = {
        'Coffee Leaf Rust (CLR)' : 0, 
        'Brown Spot Lesions (BSL)' : 1, 
        'Sooty Molds (SM)': 2
    }
    label[2] = {
        "Cercospora Leaf Spots (CLS)" : 0,
        "Phoma Leaf Spots (PLS)" : 1,
        "Coffee Leaf Miner (CLM)" : 2,
        "Red Spider Mite (RSM)" : 3
    }
    return label[n_stage]

def get_label(index,n_stage):
   for class_string, class_index in labels(n_stage).items():
      if class_index == index:
         return class_string

def show_predict(model, preprocessed_image):
    classes = model.predict(preprocessed_image)
    predicted_index = np.argmax(classes[0])
    confidence = max(100 * classes[0])
    return predicted_index, confidence

def report(result, peluang, count):
    for i in range(count):
        print()
        print(f'Stage-{i+1} result : {get_label(result[i],i)}')
        print(f'Confidence level : {peluang[i]:.2f} %')
    
def main(img_path, model1, model2, model3):
    img=load_img(img_path,target_size=(224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    index, confidence = [0,0,0], [0,0,0]
    index[0], confidence[0] = show_predict(model1,x)
    count = 1
    if index[0] == 1:
        index[1], confidence[1] = show_predict(model2,x)
        count += 1
        if index[1] == 1:
            index[2], confidence[2] = show_predict(model3,x)
            count += 1
    return index, confidence, count

if __name__ == "__main__":
    path = 'Test/'
    model1, model2, model3 = load_models()

    while True:
        while True:
            try:
                img_path = path + input('input image :')
                result, peluang, count = main(img_path, model1, model2, model3)
            except IOError:
                print(
                    "The image was not found or is invalid. Please try again, or press CTRL+C to exit the program..."
                )
            else:
                break
        
        print(img_path)
        report(result, peluang, count)
        print("\nClose the image window to continue. . .")
        img=load_img(img_path,target_size=(224,224))
        plt.imshow(img)
        plt.show()




