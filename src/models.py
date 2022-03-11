from torchvision import models

import torch as t
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    #显式的继承自nn.Module
    #resnet是卷积的一种
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        #shortcut是直连，resnet和densenet的精髓所在
        #层的定义都在初始化里
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
                nn.BatchNorm2d(outchannel))
        self.right = shortcut
    
    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
    

class ResNet(nn.Module):
    #包括34，50，101等多种结构，可以按需实现，这里是Resnet34
    def __init__(self, num_classes=152):
        super(ResNet,self).__init__()
        self.pre = nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),
                                 nn.BatchNorm2d(64),#这个64是指feature_num
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(3,2,1) )
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        self.adp = nn.AdaptiveAvgPool2d(output_size = 1)
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        short_cut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
                nn.BatchNorm2d(outchannel)
                )
        layers = []
        layers.append(ResidualBlock(inchannel,outchannel,stride,short_cut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel,outchannel))#输入和输出要一致
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = F.avg_pool2d(x,7)#注意F和原生的区别
        x = self.adp(x)
        x = x.view(x.size(0), -1)
        # x = x.view(-1, x.size(0))
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = F.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }
        # return self.fc(x)


class ResNet_O(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=152): #num changed
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(
            pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = F.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }


def get_model(config: dict):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    if "resnet" in model_name:
        # model = ResNet(  # type: ignore
        #     base_model_name=model_name,
        #     pretrained=model_params["pretrained"],
        #     num_classes=model_params["n_classes"])
        
        model = ResNet(  # type: ignore
            num_classes=model_params["n_classes"])

        # model = ResNet_O(  # type: ignore
        #     base_model_name=model_name,
        #     pretrained=model_params["pretrained"],
        #     num_classes=model_params["n_classes"])
        return model
    else:
        raise NotImplementedError
