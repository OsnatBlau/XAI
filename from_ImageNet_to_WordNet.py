import nltk
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from nltk.corpus import wordnet as wn

nltk.download('omw-1.4')
nltk.download('wordnet')

# Load pre-trained model (ResNet50 for demonstration)
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)


def get_wordnet_hierarchy(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0)

    # Make predictions using ResNet50
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=10)[0]
    print(decoded_predictions)

    if len(decoded_predictions) > 0:
        imagenet_class = decoded_predictions[0][1]  # Get the predicted ImageNet class
        synsets = wn.synsets(imagenet_class)

        hierarchy = []
        for synset in synsets:
            hierarchy.append([synset, synset.hypernym_paths()])

        return imagenet_class, hierarchy
    else:
        return "No ImageNet class found for the given image.", None

# Replace this line with your actual image path
image_path = input("Enter the path to the image: ")
imagenet_class, hierarchy = get_wordnet_hierarchy(image_path)

if hierarchy is not None:
    print(f"Predicted ImageNet Class: {imagenet_class}")
    print("WordNet Hierarchy:")
    for synset, hypernym_paths in hierarchy:
        print(f"Hierarchy for synset '{synset.name()}':")
        for path in hypernym_paths:
            for hypernym in path:
                print(f"--> {hypernym.name().split('.')[0]} ", end='')
            print("\n")
else:
    print(imagenet_class)
