
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
    git clone https://github.com/CQU-UISC/DeepTactile.git
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

### Training and evaluating the Model

To train and evaluate the model on the ST-MNIST tactile dataset:

```bash
python main.py --dataset ST-MNIST-SPLIT --batch-size 32 --epochs 100 --lr 0.001 --save-dir ./models
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
@ARTICLE{10884798,
  author={Guo, Fangming and Yu, Fangwen and Li, Mingyan and Chen, Chao and Yan, Jinjin and Li, Yan and Gu, Fuqiang and Long, Xianlei and Guo, Songtao},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Event-Driven Tactile Sensing With Dense Spiking Graph Neural Networks}, 
  year={2025},
  volume={74},
  number={},
  pages={1-13},
  keywords={Sensors;Robot sensing systems;Event detection;Tactile sensors;Graph neural networks;Neurons;Manuals;Training;Spiking neural networks;Object recognition;Event-based learning;graph neural networks (GNNs);object recognition;spiking neural networks (SNNs);tactile sensing},
  doi={10.1109/TIM.2025.3541787}}
```



## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.


## Acknowledgments

This project uses the following frameworks:

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [PyTorch](https://pytorch.org/)
