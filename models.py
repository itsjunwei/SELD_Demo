"""Main Python script to hold all the models that I want to test out"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

def count_parameters(model):
    """Returns the total number of parameters of the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=(3, 3), stride=(1, 1), 
                 padding=(1, 1), add_bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=add_bias)
        self.bn = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = F.relu_(self.bn(self.conv(x)))
        return x


class ConvBlockTwo(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1, 
                 bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise: groups=in_channels
        self.depthwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels,
                                   bias=bias)
        
        # Pointwise: 1x1
        self.pointwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   bias=bias)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out




class ResLayer(nn.Module):
    """Initialize a ResNet layer"""
    def __init__(self, in_channels, out_channels, dsc=False):
        super(ResLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        if dsc:
            self.conv1 = DepthwiseSeparableConv(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=(3,3), stride=(1,1), 
                                                padding=(1,1), bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=(3,3), stride=(1,1), 
                                padding=(1,1), bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        if dsc:
            self.conv2 = DepthwiseSeparableConv(in_channels=out_channels,
                                                out_channels=out_channels,
                                                kernel_size=(3,3), stride=(1,1), 
                                                padding=(1,1), bias=False)
        else:
            self.conv2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=(3,3), stride=(1,1),
                                padding=(1,1), bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=(1,1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)   

    def forward(self, x):

        identity = x.clone()
        out = F.relu_(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None: 
            out += self.shortcut(identity)
        else:
            out += identity
        out = F.relu_(out)
        return out



class BtnDSCLayer(nn.Module):
    """Initialize a Bottleneck Depthwise Separable Convolution layer"""
    def __init__(self, in_channels, out_channels, downsample_factor: int = 4):
        super(BtnDSCLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.down_channels = int(self.in_channels / downsample_factor)

        self.btn_in = nn.Conv2d(in_channels=in_channels,
                                out_channels=self.down_channels,
                                kernel_size=(1,1), stride=(1,1), bias=False)
        
        self.dsc = DepthwiseSeparableConv(in_channels=self.down_channels,
                                            out_channels=self.down_channels,
                                            kernel_size=(3,3), stride=(1,1), 
                                            padding=(1,1), bias=False)

        self.btn_out = nn.Conv2d(in_channels=self.down_channels,
                                 out_channels=out_channels,
                                 kernel_size=(1,1), stride=(1,1), bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=self.down_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=(1,1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)   

    def forward(self, x):

        identity = x.clone()

        out = F.relu_(self.bn1(self.btn_in(x)))
        out = self.bn2(self.btn_out(self.dsc(out)))

        if self.shortcut is not None: 
            out += self.shortcut(identity)
        else:
            out += identity
        out = F.relu_(out)
        return out
    
class SELDNet(nn.Module):
    def __init__(self, in_feat_shape, out_feat_shape, 
                 gru_size = 256, verbose=False,
                 n_res_filters = 64, n_attn_layers = 2,
                 n_attn_heads = 8, p_dropout = 0.1):
        super().__init__()

        self.verbose = verbose
        
        self.cnn_filt_1 = ConvBlock(in_channels=in_feat_shape[0],
                                    out_channels=n_res_filters)
        self.max_pool1 = nn.MaxPool2d((2,2))
        self.cnn_dropout_1 = nn.Dropout2d(p=p_dropout)
        
        self.cnn_filt_2 = ConvBlock(in_channels=n_res_filters,
                                    out_channels=n_res_filters)
        self.max_pool2 = nn.MaxPool2d((4,2))
        self.cnn_dropout_2 = nn.Dropout2d(p=p_dropout)
        
        self.cnn_filt_3 = ConvBlock(in_channels=n_res_filters,
                                    out_channels=n_res_filters)
        self.max_pool3 = nn.MaxPool2d((5,2))
        self.cnn_dropout_3 = nn.Dropout2d(p=p_dropout)
        
        self.gru_input_dim = n_res_filters * int(np.floor(in_feat_shape[-1]/8))
        self.gru = nn.GRU(input_size = self.gru_input_dim, hidden_size=gru_size,
                          num_layers = 2, batch_first=True, bidirectional=True, dropout=p_dropout)
        
        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=gru_size, num_heads=n_attn_heads, dropout=p_dropout, batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(gru_size))


        # decoding layers
        fnn_out_size = int(gru_size // 2)
        self.fc1 = nn.Linear(in_features=gru_size,
                             out_features=fnn_out_size, bias=True)
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.leaky = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=fnn_out_size,
                             out_features=out_feat_shape[-1], bias=True)
        self.dropout2 = nn.Dropout(p=p_dropout)


    def forward(self, x):

        x = self.cnn_dropout_1(self.max_pool1(self.cnn_filt_1(x)))
        x = self.cnn_dropout_2(self.max_pool2(self.cnn_filt_2(x)))
        x = self.cnn_dropout_3(self.max_pool3(self.cnn_filt_3(x)))
        if self.verbose:
            print("After CNN : {}".format(x.shape))

        # Preparing for biGRU layers
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        # Fully connected decoding layers
        x = self.leaky(self.fc1(self.dropout1(x)))
        x = torch.tanh(self.fc2(self.dropout2(x)))

        return x



class ResNet(nn.Module):
    def __init__(self, in_feat_shape, out_feat_shape, 
                 gru_size = 256, verbose=False,
                 res_layers = [64, 64, 128, 256, 512],
                 use_dsc = False, btn_dsc=False, lightweight=False,
                 fps=10):
        super().__init__()

        self.res_layers = res_layers
        self.lightweight = lightweight
        if self.lightweight:
            self.res_layers = [64, 64, 128, 256]
        self.verbose = verbose
        self.dsc = use_dsc
        self.btn_dsc = btn_dsc

        # resnet stem
        self.stem = ConvBlockTwo(in_channels= in_feat_shape[0],
                                out_channels=res_layers[0])
        if fps == 10:
            self.stempool = nn.AvgPool2d((2,2))
        else:
            self.stempool = nn.AvgPool2d((5,2))

        # resnet layer 1
        if self.btn_dsc:
            self.ResNet_1 = BtnDSCLayer(in_channels=self.res_layers[0], out_channels=self.res_layers[1])
            self.ResNet_2 = BtnDSCLayer(in_channels=self.res_layers[1], out_channels=self.res_layers[1])
        else:
            self.ResNet_1 = ResLayer(in_channels=self.res_layers[0], out_channels=self.res_layers[1], dsc=self.dsc)
            self.ResNet_2 = ResLayer(in_channels=self.res_layers[1], out_channels=self.res_layers[1], dsc=self.dsc)
        if fps == 10:
            self.pooling1 = nn.AvgPool2d((2,2))
        else:
            self.pooling1 = nn.AvgPool2d((2,2))

        # resnet layer 2
        if self.btn_dsc:
            self.ResNet_3 = BtnDSCLayer(in_channels=self.res_layers[1], out_channels=self.res_layers[2])
            self.ResNet_4 = BtnDSCLayer(in_channels=self.res_layers[2], out_channels=self.res_layers[2])
        else:
            self.ResNet_3 = ResLayer(in_channels=self.res_layers[1], out_channels=self.res_layers[2], dsc=self.dsc)
            self.ResNet_4 = ResLayer(in_channels=self.res_layers[2], out_channels=self.res_layers[2], dsc=self.dsc)
        self.pooling2 = nn.AvgPool2d((2,2))

        # resnet layer 3
        if self.btn_dsc:
            self.ResNet_5 = BtnDSCLayer(in_channels=self.res_layers[2], out_channels=self.res_layers[3])
            self.ResNet_6 = BtnDSCLayer(in_channels=self.res_layers[3], out_channels=self.res_layers[3])
        else:
            self.ResNet_5 = ResLayer(in_channels=self.res_layers[2], out_channels=self.res_layers[3], dsc=self.dsc)
            self.ResNet_6 = ResLayer(in_channels=self.res_layers[3], out_channels=self.res_layers[3], dsc=self.dsc)
        if fps == 10:
            self.pooling3 = nn.AvgPool2d((1,2))
        else:
            self.pooling3 = nn.AvgPool2d((2,2))

        # resnet layer 4
        if not self.lightweight:
            if self.btn_dsc:
                self.ResNet_7 = BtnDSCLayer(in_channels=self.res_layers[3], out_channels=self.res_layers[4])
                self.ResNet_8 = BtnDSCLayer(in_channels=self.res_layers[4], out_channels=self.res_layers[4])
            else:
                self.ResNet_7 = ResLayer(in_channels=self.res_layers[3], out_channels=self.res_layers[4], dsc=self.dsc)
                self.ResNet_8 = ResLayer(in_channels=self.res_layers[4], out_channels=self.res_layers[4], dsc=self.dsc)

        # determining the bigru size
        gru_in = self.res_layers[-1]
        self.bigru = nn.GRU(input_size = gru_in, hidden_size = gru_size,
                            num_layers = 2, batch_first=True, bidirectional=True, dropout=0.3)

        # decoding layers
        self.fc1 = nn.Linear(in_features=gru_size * 2,
                             out_features=gru_size, bias=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.leaky = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=gru_size,
                             out_features=out_feat_shape[-1], bias=True)
        self.dropout2 = nn.Dropout(p=0.2)


    def forward(self, x):

        x = self.stem(x)
        x = self.stempool(x)

        x = self.ResNet_1(x)
        x = self.ResNet_2(x)
        x = self.pooling1(x)
        if self.verbose:
            print("After R1 : {}".format(x.shape))

        x = self.ResNet_3(x)
        x = self.ResNet_4(x)
        x = self.pooling2(x)
        if self.verbose:
            print("After R2 : {}".format(x.shape))

        x = self.ResNet_5(x)
        x = self.ResNet_6(x)
        x = self.pooling3(x)
        if self.verbose:
            print("After R3 : {}".format(x.shape))

        if not self.lightweight:
            x = self.ResNet_7(x)
            x = self.ResNet_8(x)
            if self.verbose:
                print("After R4 : {}".format(x.shape))

        # Preparing for biGRU layers
        x = torch.mean(x, dim=3)
        x = x.transpose(1,2).contiguous()
        x , _ = self.bigru(x)

        # Fully connected decoding layers
        x = self.leaky(self.fc1(self.dropout1(x)))
        x = torch.tanh(self.fc2(self.dropout2(x)))

        return x



if __name__ == "__main__":
    input_feature_shape = (1, 7, 80, 191) # SALSA-Lite input shape
    output_feature_shape = (1, 2, 6)
    
    """
    ResNet Full     : 3.125G MACs, 13.706M Params
    ResNet DSC      : 0.982G MACs, 3.972M Params
    ResNet BTNDSC   : 0.767G MACs, 3.020M Params
    SELDNet         : 0.280G MACs, 3.952M Params
    ResNet Full/2   : 0.816G MACs, 4.911M Params
    ResNet DSC/2    : 0.284G MACs, 2.485M Params
    ResNet BTNDSC/2 : 0.226G MACs, 2.240M Params

    Light BTNDSC    : 0.717G MACs, 2.259M Params
    Light DSC       : 0.860G MACs, 2.509M Params
    Light Full      : 2.197G MACs, 4.919M Params
    """

    model = ResNet(in_feat_shape=(7, 80, 191),
                   out_feat_shape=(2, 6),
                   res_layers=[64, 64, 128, 256, 256],
                   use_dsc=False, verbose=True, btn_dsc=False,
                   lightweight=True, fps=2)
    # model = SELDNet(in_feat_shape=(7, 80, 191),
    #                 out_feat_shape=(10, 6))

    x = torch.rand((input_feature_shape), device=torch.device("cpu"), requires_grad=True)
    y = model(x)

    macs, params = profile(model, inputs=(torch.randn(input_feature_shape), ))
    macs, params = clever_format([macs, params], "%.3f")
    print("{} MACS and {} Params".format(macs, params))
    print("Output shape : {}".format(y.shape))
    print("Number of params : {:.3f}M".format(count_parameters(model)/(10**6)))
    
    # import torchinfo
    # model_profile = torchinfo.summary(model, input_size=input_feature_shape)
    # print('MACC:\t \t %.3f' %  (model_profile.total_mult_adds/1e9), 'G')
    # print('Memory:\t \t %.3f' %  (model_profile.total_params/1e6), 'M\n')
    
    # del x, y, model
