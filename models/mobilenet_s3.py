from torchvision import models
import torch.nn as nn

# change these manually for now
num_front_layers = 3
num_central_layers = 12
num_back_layers = 2
# remember unfreeze will be counted from the end of each model
num_unfrozen_front_layers = 0
num_unfrozen_center_layers = 2         #exp with 1 or 2(max)
num_unfrozen_back_layers = 2

def get_resnet18(pretrained: bool):
    model = models.mobilenet_v3_small(pretrained=pretrained)  # this will use cached model if available instead of downloadinig again
    return model


class front(nn.Module):
    def __init__(self, input_channels=3, pretrained=False):
        super(front, self).__init__()
        model = get_resnet18(pretrained)
        #model_children = list(model.children())[0][0:2]
        model_children = list(model.children())[0][0]
        self.input_channels = input_channels
        if self.input_channels == 1:
            self.conv_channel_change = nn.Conv2d(1,3,3,1,2)   #to keep the image size same as input image size to this conv layer
        self.front_model = nn.Sequential(*model_children)
        
        if pretrained:
            layer_iterator = iter(self.front_model)
            for i in range(num_front_layers-num_unfrozen_front_layers):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
                    #print("Front")
                    param.requires_grad = False

    def forward(self, x):

        if self.input_channels == 1:
            x = self.conv_channel_change(x)
        x = self.front_model(x)
        return x


class center(nn.Module):
    def __init__(self, pretrained=False):
        super(center, self).__init__()
        model = get_resnet18(pretrained)
        model_children = list(model.children())
        #center_model_length = len(model_children) - num_front_layers
        model_children = model_children[0][1:13]
        #model_children = model_children[0][1:13]nn.Sequential(model_children[1])
        center_model_length = len(model_children)
        self.center_model = nn.Sequential(*model_children)
        print(center_model_length)
        print(center_model_length-num_unfrozen_center_layers)
        if pretrained:
            layer_iterator = iter(self.center_model)
            for i in range(center_model_length-num_unfrozen_center_layers):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
                    #print("Center")
                    param.requires_grad = False


    def forward(self, x):
        x = self.center_model(x)
        return x


class back(nn.Module):
    def __init__(self, pretrained=False, output_dim=10):
        super(back, self).__init__()
        model = get_resnet18(pretrained)
        
        #model_length = len(model_children)
        #f=model_children[0][12]
        #c=model_children[1]
        model.classifier[3] = nn.Linear(1024, 10)
        #model_children = list(model.children())[2]
        #fc_layer = nn.Linear(512, output_dim)
        model_children = list(model.children())
        #model_children = model_children[1:3]
        model_children = model_children[1:2]+[nn.Flatten()]+model_children[2:3]
        back_model = nn.Sequential(*model_children)
        #model_children = model_children[:-1]
        self.back_model = nn.Sequential(*model_children)
        #Back_model_length = len(back_model)

        if pretrained:
            layer_iterator = iter(self.back_model)
            for i in range(4-4):
                layer = layer_iterator.__next__()
                for param in layer.parameters(num_back_layers-num_unfrozen_back_layers):
                    #print("Back")
                    param.requires_grad = False


    def forward(self, x):
        x = self.back_model(x)
        return x


if __name__ == '__main__':
    model = front(pretrained=True)
    print(">>>>>>>>>>>>>>> front >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(f'{model.front_model}\n\n')
    model = center(pretrained=True)
    print(">>>>>>>>>>>>>>> center >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(f'{model.center_model}\n\n')
    print(">>>>>>>>>>>>>>> back >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    model = back(pretrained=True)
    print(f'{model.back_model}')
    
