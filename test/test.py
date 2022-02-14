import numpy as np
import xarray as xr

from resnet50_cefas import resnet50

# load model
model = resnet50(label_level='label3_detritus')

# test numpy array
## create RGB image
image = np.random.randint(255, size=(900, 800, 3), dtype=np.uint8)
##predict
y = model.predict(image)

# test xarray
## recreate xarray.Dataset with dimensions similar to the plankton dataset
N = 15
image = np.random.randint(255, size=(900, 800, 3), dtype=np.uint8)
image_multiple = np.array([image] * N)
image_xr = xr.DataArray(image_multiple, dims=['concat_dim','y', 'x', 'channel'],
                        coords={'concat_dim':  np.arange(image_multiple.shape[0]),
                                'y': np.arange(image_multiple.shape[1]),
                                'x': np.arange(image_multiple.shape[2]),
                                'channel': np.arange(image_multiple.shape[3])})
image_xr = image_xr.to_dataset(name='raster')
image_xr = image_xr.assign(
    image_width = np.random.randint(500, 600),
    image_length = np.random.randint(500, 600)
)
##predict
y = model.predict_batch(image_xr, batch_size=3)