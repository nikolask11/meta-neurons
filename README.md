
# 🧠 Meta-Neuron Network: Hybrid SNN-ANN Architecture

A novel neural architecture that replaces traditional perceptrons with small Spiking Neural Networks (SNNs) as "meta-neurons" for improved parameter efficiency and emergent intelligent behavior.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

## 📖 Overview

Inspired by research suggesting [biological neurons require complex networks to simulate](https://www.sciencedirect.com/science/article/pii/S0896627321005018), this project explores replacing simple artificial neurons with small, trainable SNNs. These "meta-neurons" enable more sophisticated computation while maintaining parameter efficiency through emergent specialization.

## 🎯 Key Findings

### 🚀 Parameter Efficiency
| Architecture | Parameters | Accuracy | Training Time |
|--------------|------------|----------|---------------|
| **Meta-Neuron** | **37** | **1.000** | 0.98s |
| Regular ANN | 45 | 1.000 | 0.16s |
| Pure SNN | 105 | 1.000 | 1.31s |

**18% fewer parameters than equivalent ANN • 65% fewer than pure SNN**

### 🧩 Emergent Specialization
Meta-neurons automatically develop specialized roles during training:
```python
Meta0: [1. 0. 0. 1.]  # "Same-value detector" (0,0 and 1,1)
Meta1: [0. 1. 1. 0.]  # "Different-value detector" (0,1 and 1,0)
```

The output layer learns to interpret these patterns, creating a collaborative decision-making system.

## 🏗️ Architecture

```
Input (2 features)
    ↓
[2 Meta-Neurons in parallel]
    ↓  
Output Neuron (1 output)
    ↓
Final Prediction
```

**Each Meta-Neuron contains:**
- 2-layer Spiking Neural Network
- 4 internal neurons → 1 output neuron
- Temporal processing over 5 time steps
- Surrogate gradient training

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/nikolask11/meta-neurons.git
cd meta-neurons
pip install -r requirements.txt
```

### Run Experiment
```bash
python meta_neurons.py
```

### Basic Usage
```python
from meta_neurons import MetaNeuronNetwork

# Create meta-neuron network
model = MetaNeuronNetwork(input_size=2, num_meta_neurons=2, output_size=1)

# Train on your data
# (See examples in meta_neurons.py)
```

## 📊 Results Analysis

### Performance Comparison
All architectures achieve 100% accuracy on XOR, but with different characteristics:

- **Pure SNN**: Perfect predictions but 184% more parameters
- **Regular ANN**: Fast training but no emergent specialization  
- **Meta-Neuron**: Best parameter efficiency with intelligent feature discovery

### Training Behavior
- **Faster convergence** than pure SNN
- **Automatic role allocation** between meta-neurons
- **Interpretable decision-making** process

## 🧪 Technical Details

### Surrogate Gradient Training
```python
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold=0.5):
        # Hard threshold for forward pass
        return (membrane >= threshold).float()
    
    @staticmethod  
    def backward(ctx, grad_output):
        # Smooth approximation for backward pass
        grad = 10 * torch.sigmoid(10 * (membrane - ctx.threshold)) * (
                    1 - torch.sigmoid(10 * (membrane - ctx.threshold)))
        return grad_output * grad, None
```

### Network Structure
```python
class MetaNeuronNetwork(nn.Module):
    def __init__(self, input_size=2, num_meta_neurons=2, output_size=1):
        self.meta_neurons = nn.ModuleList([
            SimpleMetaNeuron(input_size) for _ in range(num_meta_neurons)
        ])
        self.output_layer = nn.Linear(num_meta_neurons, output_size)
```

## 📁 Project Structure
```
meta-neuron-network/
├── meta_neurons.py          # Main implementation
├── requirements.txt          # Dependencies
├── README.md                # This file
├── experiment_results.md    # Detailed results
└── LICENSE                  # MIT License
```

## 🎓 Research Implications

This work demonstrates:
1. **Parameter-efficient hybrid architectures** are feasible
2. **Emergent specialization** occurs automatically in complex neurons
3. **SNN-ANN hybrids** can capture benefits of both approaches
4. **Interpretable AI** through observable neuron roles

## 🔮 Future Work

- [ ] Scale to larger datasets (MNIST, CIFAR-10)
- [ ] Integrate into CNN and transformer architectures
- [ ] Explore true 5-layer SNN meta-neurons
- [ ] Test on neuromorphic hardware
- [ ] Apply to real-world classification tasks

## 🤝 Contributing

This is an experimental research project. Ideas, feedback, and collaborations are welcome! Feel free to:
- Open an [issue](https://github.com/nikolask11/meta-neurons/issues) for discussion
- Submit [pull requests](https://github.com/nikolask11/meta-neurons/pulls) with improvements
- Share your own experimental results

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Nikolas Kallweit**  
*High School Student & AI Enthusiast*

- GitHub: [@nikolask11](https://github.com/nikolask11)
- Medium: [nikolaskallweit](https://medium.com/@nikolaskallweit_83151)


## 🙏 Acknowledgments

- Inspired by [Biological Neuron Complexity Research](https://www.sciencedirect.com/science/article/pii/S0896627321005018)
- Built with [PyTorch](https://pytorch.org/) and surrogate gradient methods
- Thanks to the open-source AI community for excellent tools and resources

---

**⭐ If you find this project interesting, please give it a star on [GitHub](https://github.com/nikolask11/meta-neurons)!**

