from data.human36m import Human36mDataset
from data.deepfashion import DeepFashionDataset
from data.market import MarketDataset
from torchvision import transforms as tt
from PIL import Image

__datasets__ = {
    "Human3.6m": Human36mDataset,
    "DeepFashion": DeepFashionDataset,
    "Market": MarketDataset
}


# returns only the class, not yet an instance
def get_transforms(config):
    return {
        "Human3.6m": tt.Compose(
            [
                tt.ToPILImage(),
                tt.Resize(
                    [config["spatial_size"], config["spatial_size"]], Image.BICUBIC
                ),
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "DeepFashion": tt.Compose(
           [tt.ToPILImage(),
            tt.Resize(
                [config["spatial_size"], config["spatial_size"]], Image.BICUBIC
            ),
            tt.ToTensor(),
            tt.Lambda(lambda x: (x * 2.0) - 1.0), ]
       ),
        "Market":tt.Compose(
           [tt.ToPILImage(),
            tt.Resize(
                [config["spatial_size"], config["spatial_size"]], Image.BICUBIC
            ),
            tt.ToTensor(),
            tt.Lambda(lambda x: (x * 2.0) - 1.0), ]
       ),
    }


def get_dataset(config,custom_transforms=None):
    dataset = __datasets__[config["dataset"]]
    if custom_transforms is not None:
        print("Returning dataset with custom transform")
        transforms = custom_transforms
    else:
        transforms = get_transforms(config)[config["dataset"]]
    return dataset, transforms
