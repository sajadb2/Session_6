import unittest
import torch
import sys
import os

# Add the parent directory to the path so we can import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Net

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Net()
        self.batch_size = 64
        self.input_shape = (1, 28, 28)  # MNIST image shape

    def test_output_shape(self):
        """Test if model output shape is correct"""
        x = torch.randn(self.batch_size, *self.input_shape)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, 10))  # 10 classes for MNIST

    def test_parameter_count(self):
        """Test if model has less than 20k parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total_params, 20000)

    def test_forward_pass(self):
        """Test if forward pass works without errors"""
        x = torch.randn(1, *self.input_shape)
        try:
            output = self.model(x)
            self.assertTrue(True)
        except:
            self.fail("Forward pass failed")

if __name__ == '__main__':
    unittest.main() 