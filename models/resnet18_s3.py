from torchvision import models
import torch.nn as nn

# change these manually for now
num_front_layers = 4
num_back_layers = 4
# remember unfreeze will be counted from the end of each model
num_unfrozen_front_layers = 0
num_unfrozen_center_layers = 2   #exp with 1 or 2(max)
num_unfrozen_back_layers = 4

def get_resnet18(pretrained: bool):
    model = models.resnet18(pretrained=pretrained)  # this will use cached model if available instead of downloadinig again
    return model


class front(nn.Module):
    def __init__(self, input_channels=3, pretrained=False):
        super(front, self).__init__()
        model = get_resnet18(pretrained)
        model_children = list(model.children())
        self.input_channels = input_channels
        if self.input_channels == 1:
            self.conv_channel_change = nn.Conv2d(1,3,3,1,2)   #to keep the image size same as input image size to this conv layer
        self.front_model = nn.Sequential(*model_children[:num_front_layers])
        
        if pretrained:
            layer_iterator = iter(self.front_model)
            for i in range(num_front_layers-num_unfrozen_front_layers):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
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
        global center_model_length
        #center_model_length = len(model_children) - num_front_layers - num_back_layers
        #print(center_model_length)
        #model_children=model_children[num_front_layers:center_model_length+num_front_layers]+list(model_children[7][0])
        # Get the layers up to the 8th layer
        front_layers = model_children[num_front_layers:8]
        # Split the 8th layer into two BasicBlocks
        block1 = front_layers[-1][0]
        #block2 = front_layers[-1][1]

        # Remove the 8th layer from the front_layers list
        front_layers = front_layers[:-1]

        # Create the server model
        self.center_model = nn.Sequential(*front_layers, block1)
        center_model_length = len(list(self.center_model.children()))
        print(center_model_length)
        # Create the client model
        #client_model = nn.Sequential(*front_layers, block2)        
        #self.center_model = nn.Sequential(*model_children)
        
        #print(center_model_length-num_unfrozen_center_layers)
        if pretrained:
            
            layer_iterator = iter(self.center_model)
          
            for i in range(center_model_length-num_unfrozen_center_layers):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
                    param.requires_grad = False
        
    def freeze(self, epoch, pretrained=False):
        print("freezing the center model")
       
        num_unfrozen_center_layers=0
        if pretrained:
            
            layer_iterator = iter(self.center_model)
            for i in range(center_model_length-num_unfrozen_center_layers):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.center_model(x)
        return x



class back(nn.Module):
    def __init__(self, pretrained=False, output_dim=10):
        super(back, self).__init__()
        model = get_resnet18(pretrained)
        model_children = list(model.children())
        model_length = len(model_children)
        
        # Get the layers up to the 8th layer
        front_layers = model_children[:8]

        # Split the 8th layer into two BasicBlocks
        #block1 = front_layers[-1][0]
        block2 = model_children[8]
        block1 = front_layers[-1][1]

        # Remove the 8th layer from the front_layers list
        #front_layers = front_layers[:-1]

        # Create the client model
        #client_model = nn.Sequential(*front_layers, block2)

        fc_layer = nn.Linear(512, output_dim)
        #back_layers = model_children[:-1] + [nn.Flatten()] + [fc_layer]
        back_layers = [nn.Flatten()] + [fc_layer]
        #self.back_model = nn.Sequential(*model_children[model_length-num_back_layers:])
        self.back_model = nn.Sequential(block1, block2, *back_layers)
        num_back_layers = len(list(self.back_model.children()))
    
        if pretrained:
            layer_iterator = iter(self.back_model)
            for i in range(num_back_layers-num_unfrozen_back_layers):
                layer = layer_iterator.__next__()
                for param in layer.parameters():
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
    