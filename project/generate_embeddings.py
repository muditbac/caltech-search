import tensorflow as tf
from project.config import *
from pathlib import Path
from dtd import autopatch
from tqdm import tqdm
import numpy as np

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (3,)


def get_model():
    """
    :return: model, whose input should be a 4D tensor in 0-255 range.
    """
    input = tf.keras.Input(INPUT_SHAPE)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(input)

    # Passing input through model
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=INPUT_SHAPE,
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    x = model(x)

    return tf.keras.Model(input, x)


def generate_embeddings(model, image_dir):
    complete_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        labels='inferred',
        shuffle=False,
        image_size=IMAGE_SIZE,
        batch_size=2,
    )
    class_names = np.array(complete_dataset.class_names)
    filepaths = np.array(complete_dataset.file_paths)

    all_embeddings = []
    all_labels = []
    for x, y in tqdm(complete_dataset):
        embeddings = model(x)
        all_embeddings.append(embeddings)
        all_labels.append(y)
    all_embeddings = tf.concat(all_embeddings, axis=0)
    all_labels = tf.concat(all_labels, axis=0)

    all_embeddings = all_embeddings.numpy()
    all_labels = class_names[all_labels.numpy()]

    np.save('data/embeddings.npy', all_embeddings)
    np.save('data/embeddings.labels.npy', all_labels)
    np.save('data/embeddings.filepaths.npy', filepaths)

    print(all_embeddings[0])
    print(all_labels[0])

    return all_embeddings, all_labels


if __name__ == '__main__':
    generate_embeddings(get_model(), DATA_FOLDER)
