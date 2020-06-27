import torch
import numpy as np
import cv2


class ObservationPreprocessor:

    def __init__(self, image_shape, meas_scale, device='cpu'):
        self.image_shape = image_shape
        self.meas_scale = meas_scale
        self.device = device

    def __call__(self, inputs):
        """ Prepare observations to be fed to a neural network.

        :param inputs: dict(image=np.array(shape=(H, W, C), dtype=float32),
                            meas=np.array(shape=(D_meas,), dtype=float32),
                            goal=np.array(shape=(D_goal,), dtype=float32)
                            )

        :return: dict(image=torch.tensor(shape=(1, C, H, W), dtype=float32),
                      meas=torch.tensor(shape=(1, D_meas), dtype=float32),
                      goal=torch.tensor(shape=(1, D_goal), dtype=float32)
                      )
        """

        image = _preprocess_image(inputs['image'], self.image_shape)
        meas = inputs['meas'] / self.meas_scale - 0.5
        goal = inputs['goal']

        # move to torch
        d = dict(image=image, meas=meas, goal=goal)
        d = {k: torch.from_numpy(v).to(self.device).unsqueeze(0) for k, v in d.items()}

        return d


class BatchObservationPreprocessor:

    def __init__(self, image_shape, meas_scale, device='cpu'):
        self.image_shape = image_shape
        self.meas_scale = meas_scale
        self.device = device

    def __call__(self, inputs):
        """ Prepare batch of observations to be fed to a neural network. A batch is a dict of batches.

        :param inputs: dict(image=np.array(shape=(B, H, W, C), dtype=uint8),
                            meas=np.array(shape=(B,D_meas), dtype=float32),
                            goal=np.array(shape=(B,D_goal), dtype=float32)
                            )

        :return: dict(image=torch.tensor(shape=(B, C, H, W), dtype=float32),
                      meas=torch.tensor(shape=(B,D_meas), dtype=float32),
                      goal=torch.tensor(shape=(B,D_goal), dtype=float32)
                      )
        """

        images = inputs['image']
        batch_size = images.shape[0]
        images = np.stack([_preprocess_image(images[i], self.image_shape) for i in range(batch_size)])
        meas = inputs['meas'] / self.meas_scale - 0.5
        goal = inputs['goal']

        # move to torch
        d = dict(image=images, meas=meas, goal=goal)
        d = {k: torch.from_numpy(v).to(self.device) for k, v in d.items()}

        return d


class ListObservationPreprocessor(BatchObservationPreprocessor):

    def __call__(self, inputs):
        """ Prepare list of observations to be fed to a neural network.
            Convert list to batch. Then use BatchPreprocessor.
            List of dicts to dict of numpy arrays

        :param inputs:
        :return:
        """

        out = _collate_dicts(inputs)
        d = super().__call__(out)
        return d


def _preprocess_image(image, output_shape):

    H, W = output_shape
    image = cv2.resize(image, (W, H))

    if image.ndim == 2:
        image = np.expand_dims(image, -1)  # [H, W, 1]

    # [H, W, C] -> [C, H, W]
    image = np.moveaxis(image, 2, 0)

    # make floats in [-0.5, 0.5]
    image = np.asarray(image, dtype=np.float32) / 255. - 0.5
    return image


def _collate_dicts(dicts):

    keys = dicts[0].keys()
    dict_out = {}
    for key in keys:
        lst = [d[key] for d in dicts]
        dict_out[key] = np.stack(lst)

    return dict_out
