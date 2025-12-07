import torch.nn as nn

class TextCnn(nn.Module):
    def __init__(self,num_classes): #mum_classes是分类数
        super(TextCnn, self).__init__()
        self.features = nn.Sequential( #做特征提取
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),#32*60
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),#32*30
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),#64*30
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)#64*15
        )
        #定义全连接层做分类
        self.classifier = nn.Sequential(
            nn.Linear(64*15, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        '''前向传播'''
        x = self.features(x) #先进行特征提取
        x = x.view(x.size(0), -1) #展平
        x = self.classifier(x)
        return x