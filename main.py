from utils import (
    read_images,    
    stratified_cv,
    create_model,
    MAX_HEIGHT, MAX_WIDTH,
    BATCH_SIZE, N_EPOCHS,
    plot_history
)
from pathlib import Path
import keras
from sklearn.model_selection import StratifiedShuffleSplit

path_to_docs = Path(__file__).parent.joinpath("data")
path_to_annotation = path_to_docs.joinpath("page_classification_validation.xlsx")
path_to_save = Path(__file__).parent.joinpath("pictures")
path_to_labels = path_to_save.joinpath("labels.csv")

SEED = 42
N_SPLITS = 5

if __name__ == "__main__":
    # create_dataset(path_to_docs=path_to_docs, path_to_annotation=path_to_annotation, path_to_save=path_to_save)
    images, labels = read_images(path_to_images=path_to_save, path_to_labels=path_to_labels)

    # apply cross-validation
    # stratified_cv(X=images, y=labels, n_splits=N_SPLITS, seed=SEED)

    # make a stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    for train_index, test_index in sss.split(images, labels):
        X_train, X_test = images[train_index], images[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Initialize and train the model
        model = create_model(in_shape=(MAX_HEIGHT, MAX_WIDTH, 1), out_shape=1)
        history = model.fit(
            X_train, 
            y_train, 
            batch_size=BATCH_SIZE, 
            epochs=N_EPOCHS, 
            verbose=1, 
            validation_data=(X_test, y_test),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_recall", 
                    patience=15, 
                    restore_best_weights=True
                )
            ]
        )
        
        # plot history
        plot_history(history=history)

    # save model
    model.save("model.keras")