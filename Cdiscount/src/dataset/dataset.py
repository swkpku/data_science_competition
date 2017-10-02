import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io, transform
from torchvision import transforms
from io import BytesIO
import bson

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img = transform.resize(sample, (self.output_size, self.output_size))
        return img
    
class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose((2, 0, 1))
        return torch.from_numpy(sample).float()

class CdiscountDataset(Dataset):
    """Cdiscount dataset."""

    def __init__(self, offsets_csv, images_csv, bson_file_path, with_label, transform=None):
        self.offsets_df = pd.read_csv(offsets_csv, index_col=0)
        self.images_df = pd.read_csv(images_csv, index_col=0)
        self.bson_file_path = bson_file_path
        self.with_label = with_label
        self.transform = transform

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        image_row = self.images_df.iloc[idx]
        product_id = image_row["product_id"]
        offset_row = self.offsets_df.loc[product_id]

        # Read this product's data from the BSON file.
        bson_file = open(self.bson_file_path, "rb")
        bson_file.seek(offset_row["offset"])
        item_data = bson_file.read(offset_row["length"])

        # Grab the image from the product.
        item = bson.BSON.decode(item_data)
        img_idx = image_row["img_idx"]
        bson_img = item["imgs"][img_idx]["picture"]
        img = io.imread(BytesIO(bson_img))
        
        if self.transform:
            img = self.transform(img)
            
        target = -1
        if self.with_label:
            target = image_row["category_idx"].item()

        return img, target, product_id

def get_cdiscount_dataset(offsets_csv, images_csv, bson_file_path, with_label, resize):
    return CdiscountDataset(offsets_csv=offsets_csv,
                            images_csv=images_csv,
                            bson_file_path=bson_file_path,
                            with_label=with_label,
                            transform=transforms.Compose([
                                Rescale(resize),
                                ToTensor()
                            ]))
    