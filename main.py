import sys

from trainer import train
from dataset import load_dataset
from fc_cnn import initialize_model


def main(train_list, val_list, root_folder):
    # Load dataset
    loader_train, loader_val = load_dataset(train_list, val_list, root_folder)
    print('Dataset loaded.')
    # Initialize model
    model, optimizer, criterion = initialize_model()
    print('Model initialized, starting training...\n')
    # Train model with dataset
    training_losses, testing_losses = train(loader_train, loader_val, model, optimizer, criterion)
    print('Training has ended.')

if __name__ == "__main__":
    if len(sys.argv) == 4:
        train_list, val_list, root_folder = sys.argv[1], sys.argv[2], sys.argv[3]
        main(train_list, val_list, root_folder)
    else:
        print('Command format as follows: python3 main.py train_list val_list dataset_path')
        print('Example: python3 main.py train.txt val.txt data')