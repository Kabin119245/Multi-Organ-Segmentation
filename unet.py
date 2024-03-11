from monai.networks.nets import UNet
from monai.networks.layers import Norm
from enum import Enum
from monai.losses import DiceLoss,GeneralizedDiceLoss

import torch
from utilities import train


data_dir = 'D:\\minor test1\\Data_Train_Test'
model_dir = 'D:\\minor test1\\results1' 


device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model,  loss_function, optimizer, 200, model_dir)