import torch
import torch.nn as nn

class GestureModel(nn.Module): #creates a neural network class called GestureModel
    def __init__(self):
        super(GestureModel, self).__init__() #super() initializes the parent class (nn.Module) so PyTorch can manage the model.
        
        #This is the actual neural network.
        #nn.Sequential means the layers run one after another.
        self.model = nn.Sequential(
            nn.Linear(42, 64), #42 numbers in → 64 numbers out
            nn.ReLU(), #removes negative numbers.
            nn.Linear(64, 32), #64 numbers → 32 numbers
            nn.ReLU(),
            nn.Linear(32, 10)  # number of gestures, 32 numbers → 10 numbers
        )

    def forward(self, x):
        return self.model(x)