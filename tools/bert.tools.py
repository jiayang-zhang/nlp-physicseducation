from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from tensorflow.python.ops.variables import validate_synchronization_aggregation_trainable


# define a function for train-validation data splitting
def train_val_split(dataset, ratio):
    '''
    # Create a ratio:(1-ratio) train-validation split

    dataset: tensor object
    ratio: float <1 and >0
    '''
    # Calculate the number of samples to include in each set
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    return train_dataset, val_dataset


def loader(train_dataset, val_dataset, batch_size):

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order
    train_dataloader = DataLoader(
                        train_dataset,                  # the trainig samples
                        sampler = RandomSampler(train_dataset), # select batches randomly
                        batch_size = batch_size,        # trains with this batch size
    )

    # For validation the order doesn't matter, so we'll just read them sequentially
    validation_dataloader = DataLoader(
                        val_dataset,                # the validation samples
                        sampler = SequentialSampler(val_dataset), # Pull out batches sequentially
                        batch_size = batch_size     # evaluate with this batch size
    )
    return train_dataloader, validation_dataloader
