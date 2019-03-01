import attention_model
import torchsummary
import setting

import torch
import torch.nn as nn
from torchsummary import summary


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, y):
        x1 = self.features(x)
        x2 = self.features(y)
        return x1, x2

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.set_default_tensor_type('torch.FloatTensor')

if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SimpleConv().to(device)
    #
    # summary(model, [(1, 16, 16), (1, 28, 28)])
    test_model = attention_model.get_model()
    torchsummary.summary(test_model, input_size=[(setting.n_feature_type, setting.d_model), (setting.n_feature_type, setting.d_model)])