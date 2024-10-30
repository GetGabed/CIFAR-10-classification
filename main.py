import torch
import numpy as np


def main():
    # Check availability of GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load data
