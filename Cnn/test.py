import torch
from model.Cnn import TextCnn

x=torch.randn(20,1,60) #batch,通道数，序列长度
model=TextCnn(num_classes=2)
output=model(x)
print(output.shape)