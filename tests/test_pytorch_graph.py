import unittest
import torch
from tensorboardX import SummaryWriter

class PytorchGraphTest(unittest.TestCase):
    def test_pytorch_graph(self):
        dummy_input = (torch.zeros(1, 3),)

        
        class myLinear(torch.nn.Module):
            def __init__(self):
                super(myLinear, self).__init__()
                self.l = torch.nn.Linear(3, 5)
            def forward(self, x):
                return self.l(x)

        with SummaryWriter(comment='LinearModel') as w:
            w.add_graph(myLinear(), dummy_input, True)

    def test_wrong_input_size(self):
        print('expect error here:')
        with self.assertRaises(RuntimeError) as e_info:
            dummy_input = torch.rand(1, 9)
            model = torch.nn.Linear(3, 5)
            with SummaryWriter(comment='expect_error') as w:
                w.add_graph(model, dummy_input)  # error
