import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson
#Reference
#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html

def loadImage(sample=100):
    transform_list = [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    poslist=np.array(os.listdir("./images/train/happy/"))
    neglist=np.array(os.listdir("./images/train/disgust/"))
    posidx=np.arange(len(poslist))
    negidx=np.arange(len(neglist))
    np.random.shuffle(posidx)
    np.random.shuffle(negidx)
    poslist=poslist[posidx[:sample]]
    neglist=neglist[negidx[:sample]]
    samples=torch.zeros((sample,2,48,48))
    for i in range(30):
        posimg = Image.open(f'./images/train/happy/{polist[i]}')
        negimg = Image.open(f'./images/train/disgust/{neglist[i]}')
        posimg = transform(posimg)
        negimg = transform(negimg)
        sample= torch.concat([posimg,negimg],dim=0)
        samples[i]=sample
    for i,fname in enumerate(neglist):
        img = Image.open(f'./images/train/disgust/{fname}')
        img = transform(img)
        negsample[i]=img
    return possample,negsample

class Discriminator(nn.Module):
    def __init__( self):
        super(Discriminator, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.convnet =  torch.nn.Sequential(
            torch.nn.Conv2d(2,16,3,stride=2),            #   (48-(3-1)-1)/2 +1 = 23
            torch.nn.ReLU(), #(6,5),
            torch.nn.Conv2d(16,32,3,stride=2), #   (23-(3-1)-1)/2 +1 = 11
            torch.nn.ReLU(), #(6,5)
            torch.nn.Conv2d(32,64,3,stride=2) #   (11-(3-1)-1)/2 +1 = 5
        ).to(self.device)
        # 5*5*64
        self.discriminator =torch.nn.Linear(5*5*64,1)

    def forward(self, x):
        x=x.to(self.device)
        o= self.convnet(x)
        o= self.discriminator(o)
        return o



