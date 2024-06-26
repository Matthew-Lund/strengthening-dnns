#Matthew Lund
#mtlund@wpi.edu
#ECE577 Homework3

#imported libs

#For grabbing ImageNet Images
import os
import random
script_directory = os.path.dirname(os.path.abspath(__file__))
save_directory = os.path.join(script_directory, "Adversarial_Images")

#Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

#Numpy and Plotting
import numpy as np
import matplotlib.pyplot as plt

#Foolbox
import foolbox
from foolbox.attacks import BlendedUniformNoiseAttack, FGSM, ContrastReductionAttack, SinglePixelAttack, SaliencyMapAttack

# Load the pre-trained ResNet50 model
model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)
preprocessing = (np.array([103.939, 116.779, 123.68]), 1)

# Foolbox model
fmodel = foolbox.models.TensorFlowModel.from_keras(model, preprocessing=preprocessing, bounds=(0,255))

# Define the directory containing the images
images_directory = os.path.join(script_directory, "Images")

# Get all JPEG filenames in the directory
all_filenames = [f for f in os.listdir(images_directory) if f.endswith('.JPEG')]

# Randomly pick 10 filenames
image_filenames = random.sample(all_filenames, 100)

# Initialize attacks
attack_criterion = foolbox.criteria.Misclassification()
attack_criterion_single = foolbox.criteria.OriginalClassProbability(0.05)

attacks = {
    'BlendedUniformNoiseAttack': BlendedUniformNoiseAttack(fmodel, criterion=attack_criterion),
    'ContrastReductionAttack': ContrastReductionAttack(fmodel, criterion=attack_criterion),
    'FGSM': FGSM(fmodel, criterion=attack_criterion),
    'SinglePixelAttack': SinglePixelAttack(fmodel, criterion = attack_criterion_single),
    'SaliencyMapAttack': SaliencyMapAttack(fmodel, criterion=attack_criterion)
}

# For Loop for Images
for image_file in image_filenames:
    # Load image
    img_path = os.path.join(images_directory, image_file)
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    true_label = np.argmax(fmodel.predictions(img))
    
    #put image into BGR
    img_bgr = img[..., ::-1]
    # Apply each attack

    for attack_name, attack_instance in attacks.items():
        target_label = (true_label + 1) % 1000
        

        # Call the attack with an Adversarial instance
        adversarial = attack_instance(img_bgr, label=true_label, unpack = False)
        img_adv = adversarial.image[..., ::-1]

        noise = img_adv - img
        noise[noise == 0] = 255 

        # Plot original, adversarial, and noise
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(img / 255.0)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Adversarial ({})'.format(attack_name))
        plt.imshow(img_adv / 255.0)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Noise')
        plt.imshow(noise / np.max(np.abs(noise)) / 2.0 + 0.5)
        plt.axis('off')

        # Save the image
        save_path = os.path.join(save_directory, f"{image_file}_{attack_name}.png")
        plt.savefig(save_path)

        plt.close()  # Close the plot to release memory
