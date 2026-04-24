# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
import torch
import torch.nn.functional as F

class CTC(torch.nn.Module):
    """CTC module"""

    def __init__(self,
        reduction = "sum",
        blank_id = 0,
        zero_infinity = False):
        """ Construct CTC module
        Args:
            reduction: reduce the CTC loss into a scalar
            blank_id: blank label.
        """
        super().__init__()

        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id,
                                         reduction=reduction,
                                         zero_infinity=zero_infinity)

    def forward(self, ys_hat, ys_pad, hlens, ys_lens):
        """Calculate CTC loss.

        Args:
            ys_hat: batch of padded hidden state sequences (B, L, Nvocab)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            hlens: batch of lengths of hidden state sequences (B)
            ys_lens: batch of lengths of character sequence (B)
        """
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss

    def log_softmax(self, ys_pad):
        """log_softmax of frame activations

        Args:
            Tensor ys_pad: 3d tensor (B, Tmax, Nvocab)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(ys_pad, dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor ys_pad: 3d tensor (B, Tmax, Nvocab)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(ys_pad, dim=2)
