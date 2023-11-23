from ultralytics import YOLO


def train(model_type: str, config_file: str, epochs, patience, batch_size):
    # Load a model
    print('Loading the base model...')
    model = YOLO(model_type)

    # Use the model
    results = model.train(data=config_file, epochs=epochs, patience = patience, batch = batch_size)
    print(f'Completed training {model_type}')
