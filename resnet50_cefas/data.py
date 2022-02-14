from torch.utils.data import Dataset
import torchvision


class PlanktonDataset(Dataset):
    def __init__(self, ds, output_path=None):
        self.ds = ds
        self.n_images = self.ds.dims['concat_dim']
        self.img_ixs = self.ds.concat_dim.values
        if output_path is not None:
            self.output_path = output_path

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        im_raw = self.ds.sel(concat_dim=self.img_ixs[idx])
        imw = im_raw.image_width.values
        iml = im_raw.image_length.values

        im_pre = im_raw['raster'][0:iml, 0:imw, :].values

        im_pre = torchvision.transforms.ToTensor()(im_pre)
        im_pre = torchvision.transforms.Resize((256,256))(im_pre)

        X_raw = im_raw['raster'].values #raw image
        X_pre = im_pre #resize image

        return X_raw, X_pre