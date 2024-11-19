import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class ConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x): 
        return self.relu(self.bn(self.conv(x)))
    

class ConvTBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x): 
        return self.relu(self.bn(self.convT(x)))
        

class Encoder(nn.Module): 
    def __init__(self, in_channels:int, out_channels:list): 
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ConvBlock(in_channels, out_channels[0]))
        
        for i in range(1, len(out_channels)): 
            self.layers.append(ConvBlock(out_channels[i - 1], out_channels[i]))
                    
        self.cache = {}
    
    def forward(self, x): 
        out = nn.Sequential(*self.layers)(x)
        out, idx = F.max_pool2d(out, 2, 2, return_indices = True)
        self.cache['idx'] = idx
        self.cache['dim'] = x.shape
        
        return out
    
class Decoder(nn.Module): 
    def __init__(self, in_channels:int, out_channels:list): 
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ConvTBlock(in_channels, out_channels[0]))
        
        for i in range(1, len(out_channels)): 
            self.layers.append(ConvTBlock(out_channels[i - 1], out_channels[i]))
        
    def forward(self, x, enc_cache): 
        idx, dim = enc_cache['idx'], enc_cache['dim']
        x = F.max_unpool2d(x, idx, 2, 2, output_size = dim)
        return nn.Sequential(*self.layers)(x)



class SegNet(nn.Module): 
    """
    SegNet for semantic segmentation implemented 
    
    """
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        encoder_dims = [ 64, 64, 
                         128, 128, 
                         256, 256, 256, 
                         512, 512, 512, 
                         512, 512, 512]
        
        self.vgg16 = torchvision.models.vgg16(pretrained = True)
        self.encoder = nn.ModuleList()
        
        self.encoder.append(Encoder(in_channels, encoder_dims[:2]))
        self.encoder.append(Encoder(encoder_dims[1], encoder_dims[2:4]))
        self.encoder.append(Encoder(encoder_dims[3], encoder_dims[4:7]))
        self.encoder.append(Encoder(encoder_dims[6], encoder_dims[7:10]))
        self.encoder.append(Encoder(encoder_dims[9], encoder_dims[10:]))
        self.sigmoid = nn.Sigmoid()
           
        self.init_vgg16()
        
        decoder_dims = [ 512, 512, 512,
                         512, 512, 512,
                         256, 256, 256, 
                         128, 128, 
                          64, 64]
        
        self.decoder = nn.ModuleList()
        self.decoder.append(Decoder(encoder_dims[-1], decoder_dims[:3]))
        self.decoder.append(Decoder(decoder_dims[3], decoder_dims[4:7]))
        self.decoder.append(Decoder(decoder_dims[6], decoder_dims[7:10]))
        self.decoder.append(Decoder(decoder_dims[9], decoder_dims[10:12]))
        self.decoder.append(Decoder(decoder_dims[11], [decoder_dims[12], out_channels]))
        
        
    def init_vgg16(self): 
        self.encoder[0].layers[0].conv.weight.data = self.vgg16.features[0].weight.data
        self.encoder[0].layers[0].conv.bias.data = self.vgg16.features[0].bias.data
        self.encoder[0].layers[1].conv.weight.data = self.vgg16.features[2].weight.data
        self.encoder[0].layers[1].conv.bias.data = self.vgg16.features[2].bias.data
        
        self.encoder[1].layers[0].conv.weight.data = self.vgg16.features[5].weight.data
        self.encoder[1].layers[0].conv.bias.data = self.vgg16.features[5].bias.data
        self.encoder[1].layers[1].conv.weight.data = self.vgg16.features[7].weight.data
        self.encoder[1].layers[1].conv.bias.data = self.vgg16.features[7].bias.data
        
        self.encoder[2].layers[0].conv.weight.data = self.vgg16.features[10].weight.data
        self.encoder[2].layers[0].conv.bias.data = self.vgg16.features[10].bias.data
        self.encoder[2].layers[1].conv.weight.data = self.vgg16.features[12].weight.data
        self.encoder[2].layers[1].conv.bias.data = self.vgg16.features[12].bias.data
        self.encoder[2].layers[2].conv.weight.data = self.vgg16.features[14].weight.data
        self.encoder[2].layers[2].conv.bias.data = self.vgg16.features[14].bias.data
        
        self.encoder[3].layers[0].conv.weight.data = self.vgg16.features[17].weight.data
        self.encoder[3].layers[0].conv.bias.data = self.vgg16.features[17].bias.data
        self.encoder[3].layers[1].conv.weight.data = self.vgg16.features[19].weight.data
        self.encoder[3].layers[1].conv.bias.data = self.vgg16.features[19].bias.data
        self.encoder[3].layers[2].conv.weight.data = self.vgg16.features[21].weight.data
        self.encoder[3].layers[2].conv.bias.data = self.vgg16.features[21].bias.data
        
        self.encoder[4].layers[0].conv.weight.data = self.vgg16.features[24].weight.data
        self.encoder[4].layers[0].conv.bias.data = self.vgg16.features[24].bias.data
        self.encoder[4].layers[1].conv.weight.data = self.vgg16.features[26].weight.data
        self.encoder[4].layers[1].conv.bias.data = self.vgg16.features[26].bias.data
        self.encoder[4].layers[2].conv.weight.data = self.vgg16.features[28].weight.data
        self.encoder[4].layers[2].conv.bias.data = self.vgg16.features[28].bias.data
        
    def forward(self, x): 
        out = nn.Sequential(*self.encoder)(x)
        for i in range(len(self.decoder)):
            out = self.decoder[i](out, self.encoder[len(self.encoder) - i - 1].cache)
        return out
    
