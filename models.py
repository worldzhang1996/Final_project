import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, input_size, max_layer_size, output_size, is_more_layer=False):
        super(Block, self).__init__()

        self.block_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU6(),
            nn.Linear(64, 128),
            nn.ReLU6(),
            nn.Linear(128, max_layer_size),
            nn.ReLU6(),
            nn.Linear(max_layer_size, 64),
            nn.ReLU6(),
            nn.Linear(64, output_size),
            nn.ReLU6(),
        ) if is_more_layer else nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU6(),
            nn.Linear(64, max_layer_size),
            nn.ReLU6(),
            nn.Linear(max_layer_size, 64),
            nn.ReLU6(),
            nn.Linear(64, output_size),
            nn.ReLU6(),
        )

        self.short_cut_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Linear(64, output_size),
            nn.ReLU6(),
        )
        # 初始化
        self._initialize_weights()

    def forward(self, x):
        block_x = self.block_layer(x)
        short_cut_x = self.short_cut_layer(x)
        x = torch.add(block_x, short_cut_x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DistModule(nn.Module):
    def __init__(self):
        super(DistModule, self).__init__()
        self.block_layer1 = Block(5, 128, 16)
        self.block_layer2 = Block(5, 256, 16)
        self.block_layer3 = Block(5, 528, 16)
        self.output_layer = nn.Linear(16, 1)
        self._initialize_weights()

    def forward(self, x):
        block_layer1_x = self.block_layer1(x)
        block_layer2_x = self.block_layer2(x)
        block_layer3_x = self.block_layer3(x)

        x = torch.add(torch.add(block_layer1_x, block_layer2_x), block_layer3_x)
        output = self.output_layer(x)
        return output, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TimeModule(nn.Module):
    def __init__(self):
        super(TimeModule, self).__init__()
        self.layer1 = torch.nn.Linear(16 + 1, 64)
        self.layer2 = torch.nn.Linear(64, 128)
        self.layer3 = torch.nn.Linear(128, 20)
        self.layer4 = torch.nn.Linear(20, 1)

        # 初始化
        self._initialize_weights()

    def forward(self, x):
        x = F.sigmoid(self.layer1(x))
        x = F.sigmoid(self.layer2(x))
        x = F.sigmoid(self.layer3(x))
        return self.layer4(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

### quantization
class QBlock(nn.Module):
    def __init__(self, input_size, max_layer_size, output_size, is_more_layer=False):
        super(QBlock, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.add = nn.quantized.FloatFunctional()

        self.block_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU6(),
            nn.Linear(64, 128),
            nn.ReLU6(),
            nn.Linear(128, max_layer_size),
            nn.ReLU6(),
            nn.Linear(max_layer_size, 64),
            nn.ReLU6(),
            nn.Linear(64, output_size),
            nn.ReLU6(),
        ) if is_more_layer else nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU6(),
            nn.Linear(64, max_layer_size),
            nn.ReLU6(),
            nn.Linear(max_layer_size, 64),
            nn.ReLU6(),
            nn.Linear(64, output_size),
            nn.ReLU6(),
        )

        self.short_cut_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Linear(64, output_size),
            nn.ReLU6(),
        )
        # 初始化
        self._initialize_weights()

    def forward(self, x):
        block_x = self.block_layer(x)
        short_cut_x = self.short_cut_layer(x)
        x = self.add.add(block_x, short_cut_x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class QDistModule(nn.Module):
    def __init__(self):
        super(QDistModule, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.add = nn.quantized.FloatFunctional()
        self.block_layer1 = QBlock(5, 128, 16)
        self.block_layer2 = QBlock(5, 256, 16)
        self.block_layer3 = QBlock(5, 528, 16)
        self.output_layer = nn.Linear(16, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.quant(x)
        block_layer1_x = self.block_layer1(x)

        block_layer2_x = self.block_layer2(x)

        block_layer3_x = self.block_layer3(x)


        x = self.add.add(self.add.add(block_layer1_x, block_layer2_x), block_layer3_x)

        output = self.output_layer(x)
        x = self.dequant(x)
        output = self.dequant(output)
        return output, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)