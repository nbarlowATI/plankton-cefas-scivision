import numpy as np
import torch
import torchvision
from .data import PlanktonDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .utils import PlanktonLabels
import pooch


class resnet50:
    def __init__(self, model_weights: dict = None, label_level=None):

        if label_level is None:
            label_level = 'label2_detritus'

        if model_weights is None:
            model_weights = dict(url="doi:10.5281/zenodo.6143685/cop-non-detritus-20211215.pth",
                             known_hash="md5:46fd1665c8b966e472152eb695d22ae3")

        # ---- DOWNLOAD
        self.model_weights = pooch.retrieve(url=model_weights['url'], known_hash=model_weights['known_hash'])

        # ---- LABEL LEVEL
        self.labels_map = PlanktonLabels(experiment=label_level).labels()
        target_classes = len(self.labels_map.values())

        # ---- LOAD PRETRAINED MODEL
        model = torchvision.models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, target_classes)

        # replace default weights by the fine-tune model
        model.load_state_dict(torch.load(self.model_weights, map_location=torch.device('cpu')))  # path of your weights

        #initialise the model in evaluation mode
        self.pretrained_model = model
        self.pretrained_model.eval()

    def show_output(self, obs, preds, xarray=False):

        plt.figure()
        if xarray:
            ix = 0
            im_target = obs.sel(concat_dim=ix)
            image = im_target['raster'].values
            imw = im_target.image_width.values
            iml = im_target.image_length.values
            plt.imshow(image[0:iml, 0:imw, :])
            preds = preds[ix]
        else:
            plt.imshow(obs)
        plt.title("Prediction: {}".format(self.labels_map[int(preds.detach().numpy())]))
        plt.show()

    def predict_batch(self, image: np.ndarray, batch_size: int):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dataset = PlanktonDataset(image)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_labels = []
        with torch.no_grad():
            for i, (_, inputs) in enumerate(dataloader):
                inputs = inputs.to(device)
                outputs = self.pretrained_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.append(preds)

        all_labels_flatten = torch.stack(all_labels).flatten()

        self.show_output(image,all_labels_flatten,xarray=True)

        return all_labels_flatten

    def predict(self, image: np.ndarray) -> np.ndarray:

        X = torchvision.transforms.ToTensor()(image)
        X = torchvision.transforms.Resize((256, 256))(X)
        X = torch.unsqueeze(X, 0)

        y = self.pretrained_model(X)

        _, preds = torch.max(y, 1)

        self.show_output(image,preds,xarray=False)

        return preds


if __name__ == "__main__":
    pass