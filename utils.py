import pandas as pd
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path
from tqdm import tqdm
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
import keras
from matplotlib import pyplot as plt

# constants
DATA_TYPE = "DataType"
PAGES_TO_REMOVE = "Pages To Remove"
DOCUMENT = "Document"

# max shapes
MAX_WIDTH = 1000
MAX_HEIGHT = 300

# learnig parametrs
BATCH_SIZE = 20
N_EPOCHS = 15
THRESHOLD = 0.9
METRICS = ["loss", "recall", "false_positive"]

# functions to process document and pages
def extract_original_pages(
    path_to_pdf: str = "",
    bytes_of_pdf: bytes = bytes(),
) -> list[Image.Image]:
    """
    Convert pages of a PDF-Document into a picture.

    Parameters
    ----------
    path_to_pdf: str = ''
        Path to a PDF-Document
    bytes_of_pdf: bytes = None
        Bytes of a PDF-Document

    Returns
    -------
    list[Image.Image]
        List of images corresponding to the pages
    """
    # safety check
    assert path_to_pdf or bytes_of_pdf, "There is no document"

    return convert_from_bytes(bytes_of_pdf) if bytes_of_pdf else convert_from_path(path_to_pdf)


def cropp_image(image: Image.Image) -> Image.Image:
    """
    Cropp image so that only 10% of height and 40% of width remain.

    Parameters
    ----------
    image: Image.Image
        Image to be cropped.

    Returns
    -------
    Image.Image
        The resting part of the original image.
    """
    left, top, right, bottom = round(0.60 * image.width), 0, image.width, round(0.10 * image.height)
    cropped = image.copy().crop(box=(left, top, right, bottom))

    return cropped


def get_list_from_string(line: str) -> list[int]:
    """Convert line that looks like [1, 2, 4, 5] into the list of integers."""
    elements = line.strip("[]").split(", ")
    return list(map(int, elements))


def create_dataset(path_to_docs: Path, path_to_annotation: Path, path_to_save: Path):
    # find all the pdf-documents
    docs = path_to_docs.glob("*.pdf")
    test_df = pd.read_excel(path_to_annotation)

    # data for future learning
    images: list[Image.Image] = list()
    marks: list[float] = list()
    names: list[str] = list()

    # process all files
    for file_path in tqdm(iterable=list(docs), desc="Processing documents"):
        # get name and check that this file is in test dataset
        file_name = file_path.parts[-1]
        assert (
            file_name in test_df[DOCUMENT].unique()
        ), f"File {file_name} does not belong to test dataset"

        # get ground True data
        df = test_df.where(test_df[DOCUMENT] == file_name).dropna()
        pages_to_remove = set(df[PAGES_TO_REMOVE].apply(get_list_from_string).sum())

        # extract pages
        original_pages: list[Image.Image] = extract_original_pages(path_to_pdf=file_path)

        # process each page
        for idx, page in enumerate(original_pages):
            # cropp image, assign a name  and create a mark
            names.append(str(len(images)) + ".png")
            images.append(cropp_image(image=page))
            marks.append(float(idx in pages_to_remove))
        
    # save cropped images
    for image, name in zip(images, names, strict=True):
        # save an image and assign name
        image.save(path_to_save.joinpath(name))
    
    # create a DF with marks
    df = pd.DataFrame(data={"image": names, "label": marks})
    df.to_csv(path_to_save.joinpath("labels.csv"))

    # report ending
    print("Total amount of images", len(images))
    print("Images are saved to", path_to_save)


def read_images(path_to_images: Path, path_to_labels: Path) -> tuple[np.ndarray, np.ndarray]:
    # read df with labels to exract them
    df = pd.read_csv(path_to_labels, index_col=0)
    df.index = df["image"]
    df.drop(columns="image", inplace=True)

    # read all images and convert them to the numpy
    images_files = path_to_images.glob("*.png")
    images_arr, labels = list(), list()

    for image_path in images_files:
        # read an image
        image = Image.open(image_path)
        # convert to grayscale
        bw_image = image.convert("L")
        # apply threshold to create binary image
        threshold = 64
        bw_image = bw_image.point(lambda x: 255 if x > threshold else 0, mode='1')

        # save black-white image
        images_arr.append(extend_image(np.array(bw_image)))

        # read a label
        image_name = image_path.parts[-1]
        labels.append(df.loc[image_name, "label"])
    
    return np.array(images_arr), np.array(labels)


def extend_image(image: np.ndarray) -> np.ndarray:
    # identify padding size
    w_padding = (MAX_WIDTH - image.shape[1]) // 2
    h_padding = (MAX_HEIGHT - image.shape[0]) // 2

    # create a new image
    matrix = np.zeros((MAX_HEIGHT, MAX_WIDTH, 1), dtype=float)

    # assign values
    matrix[h_padding: h_padding + image.shape[0], w_padding: w_padding + image.shape[1], 0] = image.copy()

    return matrix


def plot_history(history):
    """
    Plot the training and validation loss curves from the model's training history.

    Parameters:
    -----------
        history: The training history object returned by `model.fit`.

    Returns:
    --------
        None
    """
    _, axs = plt.subplots(nrows=len(METRICS), ncols=1, figsize=(18, 10))

    for metric, ax in zip(METRICS, axs):
        val_key = "val_" + metric

        ax.plot(history.history[metric], label=f"Training {metric}", color="blue")
        ax.plot(history.history[val_key], label=f"Validation {metric}", color="orange")
        ax.set_title(f"Learning Curves for {metric}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()



def stratified_cv(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int = 42):
    # Initialize the Stratified K-Fold cross-validator
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Perform Stratified K-Fold Cross-Validation
    recall_values: list[float] = list()
    false_positive_values: list[float] = list()
    loss_values: list[float] = list()
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), start=1):
        # Log
        print(f"Fold {fold}/{n_splits}")

        # Split the data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Initialize and train the model
        model = create_model(in_shape=(MAX_HEIGHT, MAX_WIDTH, 1), out_shape=1)
        history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, verbose=1, validation_data=(X_val, y_val))
        
        # gather metrics
        recall_values.append(round(history.history["val_recall"][-1], 2))
        false_positive_values.append(round(history.history["val_false_positive"][-1], 2))
        loss_values.append(round(history.history["val_loss"][-1], 4))
    
    # report the results
    print(50 * "==")
    print("Recall:", recall_values)
    print("False Positive:", false_positive_values)
    print("Loss:", loss_values)
    print(50 * "==")
    # mean values
    print("Recall:", round(sum(recall_values) / n_splits, 2))
    print("False Positive:", round(sum(false_positive_values) / n_splits, 2))
    print("Loss:", round(sum(loss_values) / n_splits, 4))
    print(50 * "==")

def create_model(in_shape: tuple[int, int, int], out_shape: int) -> keras.Sequential:
    # Define the model
    model = keras.Sequential([
        # Input layer for images of shape [300, 1000]
        keras.layers.Input(shape=in_shape),  

        # Conv2D layer
        keras.layers.Conv2D(
            filters=64, 
            kernel_size=(10, 20), 
            activation='relu',
            use_bias=True,
            kernel_regularizer=keras.regularizers.L2(l2=1e-4),
            bias_regularizer=keras.regularizers.L2(l2=1e-1),
            strides=(3, 5),
            padding="same"
        ),  

        # MaxPooling layer
        keras.layers.MaxPooling2D(pool_size=(3, 5)),

        # Conv2D layer
        keras.layers.Conv2D(
            filters=128, 
            kernel_size=(3, 4), 
            activation='relu',
            use_bias=True,
            kernel_regularizer=keras.regularizers.L2(l2=1e-4),
            bias_regularizer=keras.regularizers.L2(l2=1e-1),
            strides=(1, 1),
            padding="same"
        ),  

        # MaxPooling layer
        keras.layers.MaxPooling2D(pool_size=(2, 4)),

        # Flatten before Dense layer
        keras.layers.Flatten(),

        # Droupout to avoid overfitting
        keras.layers.Dropout(rate=0.20),

        # Output layer with 1 neuron
        keras.layers.Dense(
            units=out_shape, 
            activation='sigmoid',
            use_bias=False,
            kernel_regularizer=keras.regularizers.L2(l2=1e-2) 
        )
    ])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3,
            
        ), 
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),
        metrics=[
            keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
            keras.metrics.FalsePositives(name="false_positive", thresholds=THRESHOLD)
        ]
    )

    return model