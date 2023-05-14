import torch.nn as nn


class models(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(5, 6)
        # self.linear2 = nn.Linear(6, 6)
        self.classifier = nn.Sigmoid()


    def forward(self, input):
        output = self.linear1(input)
        # output = self.linear2(output)
        output = self.classifier(output)
        return output
