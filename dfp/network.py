import torch
import torch.nn as nn
from sacred import Ingredient


model = Ingredient('model')


@model.config
def cfg():

    dim_hidden = 128
    dim_joint_hidden = 512
    nonlinearity = 'lrelu'
    alpha = 0.2

    channels = [32, 64, 64]
    kernels = [8, 4, 3]
    strides = [4, 2, 1]
    dim_perception = 512


class MLP(nn.Module):
    def __init__(self, dims, nonlinearity='relu', alpha=0.2, activate_last=False, dropout=0):
        super().__init__()

        assert len(dims) >= 2

        nonlinearity = nonlinearity or 'identity'
        activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(alpha),
            'identity': nn.Identity()
        })

        layers = []
        for d1, d2 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))
            layers.append(activations[nonlinearity])
            if dropout:
                layers.append(nn.Dropout(dropout))

        if not activate_last:
            num_extra = 2 if dropout else 1
            layers = layers[:-num_extra]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CNN(nn.Module):
    def __init__(self, channels, kernels, strides, nonlinearity='relu', alpha=0.2,
                 activate_last=False, flatten=False):
        super().__init__()

        nonlinearity = nonlinearity or 'identity'
        activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(alpha),
            'identity': nn.Identity()
        })

        conv_layers = []
        for c_in, c_out, k, s in zip(channels[:-1], channels[1:], kernels, strides):
            conv_layers.append(nn.Conv2d(c_in, c_out, k, s))
            conv_layers.append(activations[nonlinearity])

        if not activate_last:
            conv_layers = conv_layers[:-1]

        self.conv = nn.Sequential(*conv_layers)
        self.flatten = flatten

    def forward(self, x):
        x = self.conv(x)  # [B, C, H, W]

        if self.flatten:
            x = x.flatten(1)  # [B, D]

        return x


class DFPNetwork(nn.Module):
    def __init__(self, perception_net, meas_net, goal_net, v_stream, adv_stream):
        super().__init__()

        self.perception_net = perception_net
        self.meas_net = meas_net
        self.goal_net = goal_net
        self.v_stream = v_stream
        self.adv_stream = adv_stream

    def forward(self, inputs, action=None):
        """

        :param inputs: dict observation with
                        image 'image'
                        measurement 'meas' and
                        goal 'desired_goal'

        :param action: int, action given during training.
        :return:
        """
        image = inputs['image']
        meas = inputs['meas']
        goal = inputs['goal']

        # sensor, meas, goal
        s = self.perception_net(image)
        m = self.meas_net(meas)
        g = self.goal_net(goal)

        # joint vector
        x = torch.cat((s, m, g), 1)

        # value stream (action agnostic)
        v = self.v_stream(x).unsqueeze(-1)  # [B, D, 1]

        # advantage stream
        adv = self.adv_stream(x)    # [B, D x A]

        dim_goal = goal.size(-1)
        num_actions = adv.size(-1) // dim_goal
        adv = adv.reshape(-1, dim_goal, num_actions)  # [B, D, A]

        # subtract mean
        adv_norm = adv - adv.mean(-1, keepdim=True)

        # prediction = value + advantage
        p = v + adv_norm  # [B, D, A]

        if action is not None:   # training
            B = image.size(0)
            batch_range = torch.arange(B).to(x.device)
            p = p[batch_range, :, action]  # [B, D]

        return p


@model.capture
def make_model(image_shape, dim_meas, dim_goal, num_actions, dim_hidden, dim_joint_hidden,
               dim_perception, nonlinearity, alpha, channels, kernels, strides):

    C, H, W = image_shape
    channels_ = [C] + list(channels)
    cnn = CNN(channels_, kernels, strides, nonlinearity=nonlinearity, alpha=alpha,
              activate_last=True, flatten=True)

    # infer dimension of cnn's output vectors
    sample = torch.rand(1, C, H, W)
    dim_cnn_flat = len(cnn(sample).flatten())
    dims = [dim_cnn_flat, dim_perception]
    fc = MLP(dims=dims, nonlinearity=nonlinearity, alpha=alpha, activate_last=True)

    perception_net = nn.Sequential(cnn, fc)

    meas_dims = [dim_meas, dim_hidden, dim_hidden]
    meas_net = MLP(meas_dims, nonlinearity=nonlinearity, alpha=alpha, activate_last=True)

    goal_dims = [dim_goal, dim_hidden, dim_hidden]
    goal_net = MLP(goal_dims, nonlinearity=nonlinearity, alpha=alpha, activate_last=True)

    dim_joint = dim_perception + dim_hidden + dim_hidden
    val_dims = [dim_joint, dim_joint_hidden, dim_goal]
    v_stream = MLP(val_dims, nonlinearity=nonlinearity, alpha=alpha, activate_last=False)

    adv_dims = [dim_joint, dim_joint_hidden, dim_goal*num_actions]
    adv_stream = MLP(adv_dims, nonlinearity=nonlinearity, alpha=alpha, activate_last=False)

    model = DFPNetwork(perception_net, meas_net, goal_net, v_stream, adv_stream)
    return model
