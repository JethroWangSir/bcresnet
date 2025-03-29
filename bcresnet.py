# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import torch.nn.functional as F
from torch import nn

from subspectralnorm import SubSpectralNorm


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_plane,
        out_plane,
        idx,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()

        def get_padding(kernel_size, use_dilation):
            rate = 1  # dilation rate
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        self.idx = idx

        # padding and dilation rate
        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                temp_padding, temp_rate = get_padding(k_size, use_dilation)
                rate.append(temp_rate)
                padding.append(temp_padding)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        # convbnrelu block
        layers = []
        layers.append(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False)
        )
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(self, in_plane, out_plane, idx, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        # 2D part (f2)
        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        # 1D part (f1)
        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        # 2D part
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        # 1D part
        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = F.relu(x, True)
        return x


def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stage


class BCResNets(nn.Module):
    def __init__(self, base_c, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.n = [2, 2, 4, 4]  # identical modules repeated n times
        self.c = [
            base_c * 2,  # 1 * 8 * 2 = 16
            base_c,  # 1 * 8 = 8
            int(base_c * 1.5),  # int(1 * 8 * 1.5) = 12
            base_c * 2,  # 16
            int(base_c * 2.5),  # int(1 * 8 * 2.5) = 20
            base_c * 4,  # 1 * 8 * 4 = 32
        ]  # num channels
        self.s = [1, 2]  # stage using stride
        self._build_network()

    def _build_network(self):
        # Head: (Conv-BN-ReLU)
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        # Body: BC-ResBlocks
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        # Speech branch
        self.classifier1 = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], 1, 1),
        )

        # Keyword branch
        self.classifier2 = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], 1, 1),
        )

        # Keyword classification
        self.classifier3 = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes-2, 1),
        )

    def encode(self, x):
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)

        return x
    
    def speech_branch(self, x):
        x = self.classifier1(x)
        x = x.view(-1, x.shape[1])

        return x
    
    def keyword_branch(self, x):
        x = self.classifier2(x)
        x = x.view(-1, x.shape[1])

        return x
    
    def keyword_classification(self, x):
        x = self.classifier3(x)
        x = x.view(-1, x.shape[1])

        return x

    def forward(self, x):
        x = self.encode(x)

        # Step 1: Speech vs. Non-speech classification
        x1 = self.classifier1(x)
        x1 = x1.view(-1, x1.shape[1])  # [batch, 1] -> P(speech)
        P_speech = x1.sigmoid()
        P_non_speech = 1.0 - P_speech

        # Step 2: Keyword vs. Non-keyword classification (within speech)
        x2 = self.classifier2(x)
        x2 = x2.view(-1, x2.shape[1])  # [batch, 1] -> P(keyword | speech)
        P_keyword_given_speech = x2.sigmoid()
        P_non_keyword_given_speech = 1.0 - P_keyword_given_speech

        # Step 3: Keyword classification (only if keyword is detected)
        x3 = self.classifier3(x)
        x3 = x3.view(-1, x3.shape[1])  # [batch, 10] -> P(keyword_id | keyword)
        P_keyword_id_given_keyword = x3.softmax(dim=1)

        # Compute final probabilities
        P_keyword = P_speech * P_keyword_given_speech  # P(keyword) = P(speech) * P(keyword | speech)
        P_non_keyword_speech_final = P_speech * P_non_keyword_given_speech  # P(non-keyword speech) = P(speech) * (1 - P(keyword | speech))
        P_keyword_final = P_keyword * P_keyword_id_given_keyword  # P(keyword_id) = P(keyword) * P(keyword_id | keyword)

        # Concatenate to form the final probability distribution [batch, 12]
        P_total = torch.cat([P_non_speech, P_non_keyword_speech_final, P_keyword_final], dim=1)  # [batch, 12]

        return P_total
