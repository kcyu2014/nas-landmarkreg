#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/5 下午3:08
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================

# Overcoming Multi-model forgetting procedure.

# Major changes: Make the WPLModule like a DataParallel, i.e. wrapping a existing module and recursively
# wrap its sub-module
import torch.nn as nn


class WPLWrapper(nn.Module):

    def __init__(self, module):
        super(WPLWrapper, self).__init__()
        self


def train_procedure_omf():
    pass


