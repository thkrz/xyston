from torch.utils.data import DataLoader

from . import dataset
# from .nn import model


def xyston_main():
    training_data = dataset.LASLDataset("../../../laslds/train/map.txt", "../../../laslds/train")
    train_dataloader = DataLoader(training_data, batch_size=30, shuffle=True)
    for train_features, train_labels in train_dataloader:
        print(train_features.shape)
        break
    return 0
