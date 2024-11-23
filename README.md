
# DeepTactile:Event-driven Tactile Sensing With Dense Spiking Graph Neural Networks

This repository contains the implementation of **DeepTactile**, Event-driven Tactile Sensing With Dense Spiking Graph Neural Networks. The project includes components for graph construction, tactile data processing, and model training on datasets like ST-MNIST.

![image](https://github.com/user-attachments/assets/7ea28064-6f07-450b-88c9-89439a019716)

## Features

- **Graph Construction**: Manual, KNN, and MST-based graph generation.
- **Spiking Neural Networks (SNN)**: Implements the Leaky Integrate-and-Fire (LIF) model for spiking dynamics.
- **Dataset Support**: Pre-configured support for ST-MNIST tactile dataset and custom tactile datasets.
- **Modular Design**: Components designed to be extensible for new tasks or datasets.


## File Structure

```plaintext
DEEPTACTILE-MASTER/
├── DeepTactile/
│   ├── main.py               # Main script for training and testing
│   ├── model.py              # DeepTactile network model
│   ├── tdlayer.py            # Tactile data layers and utilities
│   ├── to_graph.py           # General graph construction for tactile data
│   ├── to_STMnistgraph.py    # Graph construction for ST-MNIST tactile dataset
├── Ev-Containers/            # Example dataset: Event-based tactile containers
│   ├── train/
│   ├── test/
├── Ev-Objects/               # Example dataset: Event-based tactile objects
│   ├── train/
│   ├── test/
├── ST-MNIST-SPLIT/           # ST-MNIST tactile dataset (split into train/test)
│   ├── train/
│   ├── test/
```


## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- PyTorch 1.9+ (GPU support recommended)
- PyTorch Geometric

### Installation Steps

1. Clone this repository:
    ```bash
    git clone https://github.com/GuoFM/DeepTactile.git
    cd DeepTactile
    ```

2. Install required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install PyTorch Geometric:
    Follow the instructions for your environment at [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/).

    Example for CUDA 11.7:
    ```bash
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu117.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu117.html
    pip install torch-geometric
    ```


## Usage

### Training the Model

To train the model on the ST-MNIST tactile dataset:

```bash
python main.py --dataset ST-MNIST-SPLIT --epochs 100 --batch-size 16
```

### Evaluating the Model

To evaluate a pre-trained model:

```bash
python main.py --dataset ST-MNIST-SPLIT --evaluate --model-path ./path-to-model.pth
```

### Dataset Preparation

Ensure datasets are correctly placed in the respective directories:

- `ST-MNIST-SPLIT/`: Contains train/test splits for the ST-MNIST dataset.
- `Ev-Containers/` and `Ev-Objects/`: Placeholders for other tactile datasets.


## Example Code

Below is an example of generating a tactile graph using the KNN method:

```python
from to_STMnistgraph import TactileGraphSTMnist

# Initialize a graph generator with KNN (k=3)
graph_generator = TactileGraphSTMnist(k=3, useKNN=True)

# Generate a sample input tensor (example shape: 100 nodes, 39 features)
sample_input = torch.randn((100, 39))

# Generate the graph
graph = graph_generator(sample_input)

# Output the graph object
print(graph)
```


## Citation

If you use this code, please cite the following paper:

```
@article{guo2024eventdriven,
  title={Event-driven Tactile Sensing With Dense Spiking Graph Neural Networks},
  author={Guo, Fangming and Long, Xianlei and Yu, Fangwen and Li, Mingyan and Chen, Chao and Yan, Jinjin and Li, Yan and Gu, Fuqiang and Guo, Songtao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={XX},
  number={XX},
  pages={XXX--XXX},
  year={2024},
  publisher={IEEE}
}
```



## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.


## Acknowledgments

This project uses the following frameworks:

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [PyTorch](https://pytorch.org/)
