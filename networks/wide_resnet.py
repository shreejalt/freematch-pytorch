import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    
    def __init__(
        self,
        in_planes,
        out_planes,
        stride,
        drop_rate=0.0,
        activate_bf_res=False,
        bn_momentum=0.001,
        bn_eps=0.001,
        negative_slope=0.1
    ):
        super(BasicBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=bn_momentum, eps=bn_eps)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
                    
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.same_planes = (in_planes == out_planes)
        self.activate_bf_res = activate_bf_res
        self.conv_short = (not self.same_planes) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
    
    def forward(self, x):
        
        if not self.same_planes and self.activate_bf_res == True:
            x = self.act1(self.bn1(x))
        else:
            out = self.act1(self.bn1(x))
            
        out = self.act2(self.bn2(self.conv1(out if self.same_planes else x)))
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        
        return torch.add(x if self.same_planes else self.conv_short(x), out)

class NetworkBlock(nn.Module):
    
    def __init__(
        self,
        nb_layers,
        block,
        in_planes,
        out_planes,
        stride,
        drop_rate=0.0,
        activate_bf_res=False
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer_(
            nb_layers,
            block,
            in_planes,
            out_planes,
            stride,
            drop_rate,
            activate_bf_res
        )
    
    def forward(self, x):
        return self.layer(x)
    
    def _make_layer_(
        self, 
        nb_layers,
        block,
        in_planes,
        out_planes,
        stride,
        drop_rate,
        activate_bf_res
    ):
        layers = list()
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    in_planes=in_planes if i == 0 else out_planes,
                    out_planes=out_planes,
                    stride=stride if i == 0 else 1,
                    drop_rate=drop_rate,
                    activate_bf_res=activate_bf_res
                )
            )
        return nn.Sequential(*layers)


class WideResNet(nn.Module):
    
    def __init__(
        self,
        num_classes,
        first_stride=1,
        depth=28,
        widen_factor=2,
        drop_rate=0.0
    ):
        super(WideResNet, self).__init__()
        
        self.num_classes = num_classes
        self.first_stride = first_stride
        
        assert ((depth - 4) % 6 == 0)
        num_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nb_layers =  int((depth - 4) / 6)
        
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3, stride=1,
                                       padding=1, bias=True)
        
        self.block1 = NetworkBlock(
            nb_layers=nb_layers,
            block=BasicBlock,
            stride=first_stride,
            in_planes=num_channels[0],
            out_planes=num_channels[1],
            drop_rate=drop_rate,
            activate_bf_res=True
        )
        self.block2 = NetworkBlock(
            nb_layers=nb_layers,
            block=BasicBlock,
            stride=2,
            in_planes=num_channels[1],
            out_planes=num_channels[2],
            drop_rate=drop_rate,
            activate_bf_res=False
        )
        self.block3 = NetworkBlock(
            nb_layers=nb_layers,
            block=BasicBlock,
            stride=2,
            in_planes=num_channels[2],
            out_planes=num_channels[3],
            drop_rate=drop_rate,
            activate_bf_res=False
        )
        self.bn1 = nn.BatchNorm2d(num_channels[3], momentum=0.001, eps=0.001)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc = nn.Linear(num_channels[3], num_classes)
        
        self.out_channels = num_channels[3]
        
        self.__init__weights__()
    
    def forward(self, x, fc_flag=False, feat_flag=False):
        
        if fc_flag:
            return self.fc(x)

        out = self.__feat__forward__(x)
        if feat_flag:
            return out
        
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.out_channels)
        logits = self.fc(out)
        
        return {'feats': out, 'logits': logits}

    def __feat__forward__(self, x):
        
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.act1(self.bn1(out))
        return out
        
    def __init__weights__(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
        

def wrn_28_2(num_classes, pretrained=False, pretrained_path=None):
    model = WideResNet(num_classes=num_classes, depth=28, widen_factor=2)
    if pretrained:
        model = model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model


def wrn_28_8(num_classes, pretrained=False, pretrained_path=None):
    model = WideResNet(num_classes=num_classes, depth=28, widen_factor=8)
    if pretrained:
        model = model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model
    

if __name__ == '__main__':
    
   
    torch.manual_seed(0)
    model = wrn_28_2(num_classes=10)

    torch.manual_seed(0)
    a = torch.randn((1, 3, 32, 32))
    print(a)
    print(model(a)['logits'])