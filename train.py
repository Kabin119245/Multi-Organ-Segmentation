from monai.networks.layers import Norm
from monai.losses import DiceLoss

import torch
from preporcess import prepare
from utilities import train
import torch
import torch.nn as nn
from monai.networks.layers.factories import  Norm
from monai.networks.nets import SegResNet

data_dir = '/mnt/myhdd/Data_Train_Test/'
model_dir = '/mnt/myhdd/Data_Train_Test/result/' 
data_in = prepare(data_dir, cache=True)

device = torch.device("cpu")
model = SegResNet(
    spatial_dims = 3,
    in_channels= 1,
    out_channels= 5,
    norm = Norm.BATCH,
    act=('RELU', {'inplace': True})
).to(device)



loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 200, model_dir)