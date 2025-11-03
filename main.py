import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt


# ===== SURROGATE GRADIENT SETUP =====
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold=0.5):
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, = ctx.saved_tensors
        # Surrogate gradient: fast sigmoid derivative
        grad = 10 * torch.sigmoid(10 * (membrane - ctx.threshold)) * (
                1 - torch.sigmoid(10 * (membrane - ctx.threshold)))
        return grad_output * grad, None


spike_fn = SpikeFunction.apply


# ===== PURE SNN ARCHITECTURE =====
class PureSNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, output_size=1, time_steps=10):
        super().__init__()
        self.time_steps = time_steps
        self.threshold = 0.5

        # SNN layers - all layers use spiking neurons
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        # Initialize membrane potentials for all layers
        membrane_hidden1 = torch.zeros(batch_size, 8)
        membrane_hidden2 = torch.zeros(batch_size, 8)
        membrane_output = torch.zeros(batch_size, 1)

        total_output_spikes = 0

        for t in range(self.time_steps):
            # Input to first hidden layer
            current1 = self.input_to_hidden(x)
            membrane_hidden1 = 0.8 * membrane_hidden1 + current1
            spikes1 = spike_fn(membrane_hidden1, self.threshold)
            membrane_hidden1 = membrane_hidden1 * (1 - spikes1)

            # Hidden to hidden layer
            current2 = self.hidden_to_hidden(spikes1)
            membrane_hidden2 = 0.8 * membrane_hidden2 + current2
            spikes2 = spike_fn(membrane_hidden2, self.threshold)
            membrane_hidden2 = membrane_hidden2 * (1 - spikes2)

            # Hidden to output layer
            current_out = self.hidden_to_output(spikes2)
            membrane_output = 0.8 * membrane_output + current_out
            spikes_out = spike_fn(membrane_output, self.threshold)
            membrane_output = membrane_output * (1 - spikes_out)

            total_output_spikes += spikes_out

        # Return firing rate of output neuron
        return total_output_spikes / self.time_steps


# ===== META-NEURON ARCHITECTURE =====
class SimpleMetaNeuron(nn.Module):
    def __init__(self, input_size, time_steps=5):
        super().__init__()
        self.time_steps = time_steps
        self.threshold = 0.5
        self.layer1 = nn.Linear(input_size, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        membrane1 = torch.zeros(batch_size, 4)
        membrane2 = torch.zeros(batch_size, 1)
        total_spikes = 0

        for t in range(self.time_steps):
            # Layer 1 processing
            input_current = self.layer1(x)
            membrane1 = 0.8 * membrane1 + input_current
            spikes1 = spike_fn(membrane1, self.threshold)
            membrane1 = membrane1 * (1 - spikes1)

            # Layer 2 processing
            current2 = self.layer2(spikes1)
            membrane2 = 0.8 * membrane2 + current2
            spikes2 = spike_fn(membrane2, self.threshold)
            membrane2 = membrane2 * (1 - spikes2)

            total_spikes += spikes2

        return total_spikes / self.time_steps


class MetaNeuronNetwork(nn.Module):
    def __init__(self, input_size=2, num_meta_neurons=2, output_size=1):
        super().__init__()
        self.meta_neurons = nn.ModuleList([
            SimpleMetaNeuron(input_size) for _ in range(num_meta_neurons)
        ])
        self.output_layer = nn.Linear(num_meta_neurons, output_size)

    def forward(self, x):
        meta_outputs = [neuron(x) for neuron in self.meta_neurons]
        combined = torch.cat(meta_outputs, dim=1)
        return torch.sigmoid(self.output_layer(combined))


# ===== REGULAR ANN FOR COMPARISON =====
class RegularANN(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 4)
        self.hidden2 = nn.Linear(4, 4)
        self.hidden3 = nn.Linear(4, 2)
        self.output_layer = nn.Linear(2, output_size)

    def forward(self, x):
        h = torch.tanh(self.hidden1(x))
        h = torch.tanh(self.hidden2(h))
        h = torch.tanh(self.hidden3(h))
        return torch.sigmoid(self.output_layer(h))


# ===== ANALYSIS FUNCTIONS =====
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_meta_neurons(model, X, y):
    print("\n" + "=" * 60)
    print("META-NEURON ANALYSIS:")
    print("=" * 60)

    with torch.no_grad():
        # Get individual meta-neuron outputs
        meta_outputs = []
        for i, neuron in enumerate(model.meta_neurons):
            output = neuron(X)
            meta_outputs.append(output)
            print(f"Meta-Neuron {i} outputs: {output.squeeze().numpy()}")

        # See what patterns each meta-neuron detects
        print("\nPattern Analysis:")
        patterns = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
        for i, pattern in enumerate(patterns):
            print(f"{pattern}: ", end="")
            for j in range(len(model.meta_neurons)):
                print(f"Meta{j}={meta_outputs[j][i].item():.3f} ", end="")
            print()


def analyze_decision_making(meta_model, X):
    print("\n" + "=" * 60)
    print("DECISION MAKING ANALYSIS:")
    print("=" * 60)

    with torch.no_grad():
        # Get output layer weights
        output_weights = meta_model.output_layer.weight.data.numpy()[0]
        output_bias = meta_model.output_layer.bias.data.numpy()[0]

        print(f"Output layer weights: {output_weights}, bias: {output_bias:.3f}")

        # Show how decisions are made
        patterns = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
        for i, pattern in enumerate(patterns):
            meta0_out = meta_model.meta_neurons[0](X[i:i + 1]).item()
            meta1_out = meta_model.meta_neurons[1](X[i:i + 1]).item()

            # Calculate the raw output before sigmoid
            raw_output = output_weights[0] * meta0_out + output_weights[1] * meta1_out + output_bias
            final_output = torch.sigmoid(torch.tensor(raw_output)).item()

            print(f"{pattern}:")
            print(f"  Meta0: {meta0_out:.3f} × {output_weights[0]:.3f} = {meta0_out * output_weights[0]:.3f}")
            print(f"  Meta1: {meta1_out:.3f} × {output_weights[1]:.3f} = {meta1_out * output_weights[1]:.3f}")
            print(f"  Bias: {output_bias:.3f}")
            print(f"  Raw: {raw_output:.3f} → Final: {final_output:.3f}")
            print()


def multiple_runs_experiment(X, y, num_runs=5):
    print("\n" + "=" * 60)
    print(f"MULTIPLE RUNS EXPERIMENT ({num_runs} runs)")
    print("=" * 60)

    specialization_patterns = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1} ---")
        meta_model = MetaNeuronNetwork()

        # Quick training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(meta_model.parameters(), lr=0.01)

        for epoch in range(200):  # Shorter training
            optimizer.zero_grad()
            outputs = meta_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Analyze this run
        with torch.no_grad():
            meta0_outputs = meta_model.meta_neurons[0](X).squeeze().numpy()
            meta1_outputs = meta_model.meta_neurons[1](X).squeeze().numpy()
            specialization_patterns.append((meta0_outputs, meta1_outputs))

            print(f"Meta0: {meta0_outputs}")
            print(f"Meta1: {meta1_outputs}")

    print("\n" + "=" * 60)
    print("SPECIALIZATION SUMMARY:")
    print("=" * 60)
    for i, (meta0, meta1) in enumerate(specialization_patterns):
        print(f"Run {i + 1}:")
        print(f"  Meta0: {meta0}")
        print(f"  Meta1: {meta1}")


# ===== TRAINING AND COMPARISON =====
def train_model(model, X, y, model_name, epochs=1000):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 200 == 0:
            with torch.no_grad():
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == y).float().mean()
                print(f"{model_name} - Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.2f}")

    training_time = time.time() - start_time

    # Final evaluation
    with torch.no_grad():
        final_outputs = model(X)
        final_predictions = (final_outputs > 0.5).float()
        final_accuracy = (final_predictions == y).float().mean()
        final_loss = criterion(final_outputs, y)

    return {
        'model': model_name,
        'final_accuracy': final_accuracy.item(),
        'final_loss': final_loss.item(),
        'training_time': training_time,
        'parameters': count_parameters(model),
        'loss_history': losses
    }


# ===== MAIN COMPARISON =====
def main():
    # XOR dataset
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    print("=" * 60)
    print("COMPARING ALL THREE ARCHITECTURES")
    print("=" * 60)

    # Create models
    pure_snn = PureSNN()
    meta_model = MetaNeuronNetwork()
    ann_model = RegularANN()

    print(f"Pure SNN parameters: {count_parameters(pure_snn)}")
    print(f"Meta-neuron network parameters: {count_parameters(meta_model)}")
    print(f"Regular ANN parameters: {count_parameters(ann_model)}")
    print()

    # Train all models
    print("TRAINING PURE SNN:")
    snn_results = train_model(pure_snn, X, y, "Pure SNN", epochs=1000)

    print("\nTRAINING META-NEURON NETWORK:")
    meta_results = train_model(meta_model, X, y, "Meta-Neuron", epochs=1000)

    print("\nTRAINING REGULAR ANN:")
    ann_results = train_model(ann_model, X, y, "Regular ANN", epochs=1000)

    # Print comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON RESULTS:")
    print("=" * 60)

    results = [snn_results, meta_results, ann_results]
    for result in results:
        print(f"\n{result['model']}:")
        print(f"  Final Accuracy: {result['final_accuracy']:.3f}")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Training Time: {result['training_time']:.2f} seconds")
        print(f"  Parameters: {result['parameters']}")

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(snn_results['loss_history'], label='Pure SNN', alpha=0.7)
    plt.plot(meta_results['loss_history'], label='Meta-Neuron Network', alpha=0.7)
    plt.plot(ann_results['loss_history'], label='Regular ANN', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss: All Three Architectures')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Show final predictions
    print("\n" + "=" * 60)
    print("FINAL PREDICTIONS:")
    print("=" * 60)

    with torch.no_grad():
        print("\nPure SNN:")
        snn_preds = pure_snn(X)
        for i in range(4):
            print(f"  Input: {X[i].numpy()} -> Target: {y[i].item()} -> Prediction: {snn_preds[i].item():.3f}")

        print("\nMeta-Neuron Network:")
        meta_preds = meta_model(X)
        for i in range(4):
            print(f"  Input: {X[i].numpy()} -> Target: {y[i].item()} -> Prediction: {meta_preds[i].item():.3f}")

        print("\nRegular ANN:")
        ann_preds = ann_model(X)
        for i in range(4):
            print(f"  Input: {X[i].numpy()} -> Target: {y[i].item()} -> Prediction: {ann_preds[i].item():.3f}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)

    # Find best accuracy
    accuracies = [snn_results['final_accuracy'], meta_results['final_accuracy'], ann_results['final_accuracy']]
    best_acc = max(accuracies)
    best_model = ['Pure SNN', 'Meta-Neuron', 'Regular ANN'][accuracies.index(best_acc)]

    # Find fewest parameters
    params = [snn_results['parameters'], meta_results['parameters'], ann_results['parameters']]
    fewest_params = min(params)
    most_efficient = ['Pure SNN', 'Meta-Neuron', 'Regular ANN'][params.index(fewest_params)]

    print(f"✓ {best_model} achieved the HIGHEST accuracy ({best_acc:.3f})")
    print(f"✓ {most_efficient} has the FEWEST parameters ({fewest_params})")

    if meta_results['final_accuracy'] > ann_results['final_accuracy'] and meta_results['parameters'] < ann_results[
        'parameters']:
        print("🎯 Meta-Neuron Network achieved BOTH higher accuracy AND fewer parameters!")

    # Deep analysis of meta-neurons
    analyze_meta_neurons(meta_model, X, y)
    analyze_decision_making(meta_model, X)

    # Multiple runs experiment
    multiple_runs_experiment(X, y, num_runs=5)


if __name__ == "__main__":
    main()
