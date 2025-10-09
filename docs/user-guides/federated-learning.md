# Federated Learning Guide

This guide covers how to use federated learning with Flower to train models across multiple decentralized clients without centralizing data.

## Overview

**Federated Learning (FL)** enables training machine learning models across multiple devices or organizations while keeping data decentralized. Each client trains locally on its private data, and only model updates (not raw data) are shared with a central server for aggregation.

### Key Benefits

- **Privacy**: Data never leaves the client's device
- **Compliance**: Meets regulatory requirements (HIPAA, GDPR)
- **Scalability**: Leverages distributed compute resources
- **Data Ownership**: Organizations maintain control of their data

### Use Cases

✅ **Good for:**

- Medical imaging across hospitals (data can't leave premises)
- Mobile devices training personal models
- Financial data across banks (regulatory constraints)
- Cross-organization collaboration without data sharing

❌ **Not recommended for:**

- Single organization with centralized data access
- Small datasets (< 1000 images total)
- When centralized training is feasible and faster

---

## Installation

Install the Flower package as an optional dependency:

```bash
uv pip install -e ".[flower]"
# or
pip install -e ".[flower]"
```

This installs Flower (>=1.7.0) which provides the federated learning infrastructure.

---

## Key Concept: Composition Over Inheritance

The FL implementation **wraps existing trainers** in Flower clients rather than creating a new trainer type. This architectural decision provides maximum flexibility:

```python
# Inside FlowerClient.__init__():
self.trainer = get_trainer(
    config=config,
    model=self.model,
    criterion=self.criterion,
    optimizer=self.optimizer,
    # ... all standard trainer args
)

# Inside FlowerClient.fit():
trained_model, losses, accs = self.trainer.train()  # Reuses existing training!
```

### Benefits

- **Heterogeneous clients**: Each client can use a different trainer type
  - Client 0: `standard` trainer (CPU)
  - Client 1: `dp` trainer (privacy-sensitive data with differential privacy)
  - Client 2: `mixed_precision` trainer (GPU with AMP)
  - Client 3: `accelerate` trainer (multi-GPU)
- **Zero code changes**: Existing training infrastructure works as-is
- **Full feature support**: Callbacks, EMA, metrics, checkpointing all work
- **FL + DP synergy**: Privacy-preserving federated learning out of the box

---

## Quick Start

### 1. Prepare Federated Data Splits

Use `ml-split` with the `--federated` flag to partition data across clients:

```bash
ml-split --raw_data data/medical_images/raw \\
  --federated \\
  --num-clients 10 \\
  --partition-strategy non-iid
```

**Output:**
```
data/medical_images/splits/
├── client_0_train.txt  # Client 0's training data
├── client_0_val.txt    # Client 0's validation data
├── client_1_train.txt
├── client_1_val.txt
...
└── test.txt            # Shared global test set
```

### 2. Create Configuration

Copy and edit the federated config template:

```bash
cp ml_src/federated_config_template.yaml configs/my_fl_experiment.yaml
```

Edit the config to specify:
- Number of clients
- FL strategy (FedAvg, FedProx, etc.)
- Number of rounds
- Client profiles (heterogeneous configurations)

### 3. Run Federated Training

Launch federated training with a single command:

```bash
ml-fl-run --config configs/my_fl_experiment.yaml
```

### 4. Monitor Training

View training progress with TensorBoard:

```bash
# View all clients combined
tensorboard --logdir runs/simulation/

# Or view individual client
tensorboard --logdir runs/simulation/fl_client_0/tensorboard
```

---

## Data Partitioning Strategies

The data partitioning strategy determines how data is distributed across clients, which significantly impacts FL performance.

### IID (Independent and Identically Distributed)

**Description:** Each client receives uniform random data with balanced class distribution.

**Use case:** Ideal baseline, all clients have similar data

**Command:**
```bash
ml-split --raw_data data/my_dataset/raw \\
  --federated \\
  --num-clients 10 \\
  --partition-strategy iid
```

**Characteristics:**
- ✅ Fastest convergence
- ✅ Stable training
- ❌ Unrealistic for many real-world scenarios

---

### Non-IID (Dirichlet Distribution)

**Description:** Creates realistic heterogeneous data distributions using a Dirichlet distribution. Each client has different class proportions.

**Use case:** Most realistic for real-world scenarios (hospitals, devices)

**Command:**
```bash
ml-split --raw_data data/my_dataset/raw \\
  --federated \\
  --num-clients 10 \\
  --partition-strategy non-iid \\
  --alpha 0.5
```

**Alpha parameter:**
- `alpha=0.1`: **Very heterogeneous** (each client highly skewed toward certain classes)
- `alpha=0.5`: **Moderately heterogeneous** (realistic imbalance)
- `alpha=10.0`: **Nearly IID** (approaching uniform distribution)

**Characteristics:**
- ✅ Realistic real-world distribution
- ✅ Tests robustness of FL algorithms
- ⚠️ Slower convergence than IID
- ⚠️ May require FedProx strategy

---

### Label-Skew

**Description:** Each client sees only a subset of classes (e.g., Hospital A only sees certain diseases).

**Use case:** Extreme non-IID scenarios

**Command:**
```bash
ml-split --raw_data data/my_dataset/raw \\
  --federated \\
  --num-clients 5 \\
  --partition-strategy label-skew \\
  --classes-per-client 2
```

**Characteristics:**
- ✅ Models extreme data heterogeneity
- ✅ Useful for specialized clients
- ⚠️ Most challenging scenario
- ⚠️ Requires careful strategy selection (FedProx recommended)

---

## Execution Modes

The framework supports two execution modes: **simulation** and **deployment**.

### Simulation Mode

**Purpose:** Testing and development on a single machine

**How it works:**
- Uses Flower's simulation engine
- All clients run as separate processes on one machine
- GPU resources are shared automatically
- Perfect for prototyping and debugging

**Configuration:**
```yaml
federated:
  mode: 'simulation'

  server:
    strategy: 'FedAvg'
    num_rounds: 100

  clients:
    num_clients: 10
```

**Command:**
```bash
ml-fl-run --config configs/federated_config.yaml
```

**Advantages:**
- ✅ Single command launch
- ✅ Easy debugging
- ✅ Quick iteration
- ✅ GPU sharing handled automatically

**Limitations:**
- ❌ Limited by single machine resources
- ❌ Not representative of network latency
- ❌ All clients must fit in memory

---

### Deployment Mode

**Purpose:** Real distributed federated learning across multiple machines

**How it works:**
- Server and clients run as separate processes
- Can span multiple physical machines
- True parallelism and realistic network conditions

**Configuration:**
```yaml
federated:
  mode: 'deployment'

  server:
    address: '10.0.0.1:8080'  # Server IP and port
    strategy: 'FedAvg'
    num_rounds: 200

  clients:
    manifest:
      - id: 0
        config_override: 'configs/client_overrides/hospital_1.yaml'
      - id: 1
        config_override: 'configs/client_overrides/hospital_2.yaml'
```

**Commands:**

**Option 1: Automated (single machine)**
```bash
ml-fl-run --config configs/federated_deployment.yaml --mode deployment
```

**Option 2: Manual (distributed across machines)**
```bash
# Machine 1 (Server)
ml-fl-server --config configs/federated_deployment.yaml

# Machine 2 (Hospital 1)
ml-fl-client --config configs/federated_deployment.yaml --client-id 0

# Machine 3 (Hospital 2)
ml-fl-client --config configs/federated_deployment.yaml --client-id 1
```

**Advantages:**
- ✅ True distributed setup
- ✅ Realistic network conditions
- ✅ Scales to many clients
- ✅ Production-ready

**Considerations:**
- ⚠️ Requires network configuration
- ⚠️ More complex orchestration
- ⚠️ Need failure recovery mechanisms

---

## Heterogeneous Clients

One of the most powerful features is support for **heterogeneous clients** - different clients can use different trainer types, batch sizes, and devices.

### Configuration

Use **profiles** (simulation mode) or **manifest** (deployment mode) to specify per-client configurations:

#### Simulation Mode: Profiles

```yaml
federated:
  mode: 'simulation'

  clients:
    num_clients: 10

    # Define client groups with different capabilities
    profiles:
      # Clients 0-5: Standard GPU training
      - id: [0, 1, 2, 3, 4, 5]
        trainer_type: 'standard'
        batch_size: 32
        device: 'cuda:0'

      # Clients 6-7: Mixed precision (faster)
      - id: [6, 7]
        trainer_type: 'mixed_precision'
        batch_size: 64
        device: 'cuda:0'

      # Client 8: Privacy-sensitive (uses DP)
      - id: [8]
        trainer_type: 'dp'
        batch_size: 16
        device: 'cuda:0'
        dp:
          noise_multiplier: 1.1
          max_grad_norm: 1.0
          target_epsilon: 3.0
          target_delta: 1e-5

      # Client 9: CPU only
      - id: [9]
        trainer_type: 'standard'
        batch_size: 16
        device: 'cpu'
```

#### Deployment Mode: Manifest with Overrides

**Main config:**
```yaml
federated:
  mode: 'deployment'

  clients:
    manifest:
      - id: 0
        config_override: 'configs/client_overrides/hospital_1.yaml'
      - id: 1
        config_override: 'configs/client_overrides/hospital_2_dp.yaml'
      - id: 2
        config_override: 'configs/client_overrides/hospital_3_multigpu.yaml'
```

**Client override (configs/client_overrides/hospital_2_dp.yaml):**
```yaml
training:
  trainer_type: 'dp'
  batch_size: 16
  device: 'cuda:0'

  dp:
    noise_multiplier: 1.1
    max_grad_norm: 1.0
    target_epsilon: 3.0
    target_delta: 1e-5
```

### Example: Mixed Capabilities

```yaml
# Hospital 1: Standard GPU
- id: 0
  trainer_type: 'standard'
  batch_size: 32

# Hospital 2: Privacy-sensitive (DP)
- id: 1
  trainer_type: 'dp'
  batch_size: 16

# Hospital 3: High-performance (Multi-GPU)
- id: 2
  trainer_type: 'accelerate'
  batch_size: 128

# Hospital 4: CPU only (limited resources)
- id: 3
  trainer_type: 'standard'
  device: 'cpu'
  batch_size: 8
```

All clients train together and contribute to the global model!

---

## Federated Learning Strategies

The framework supports multiple FL strategies for different scenarios.

### FedAvg (Default)

**Description:** Federated Averaging - the standard FL algorithm

**Use case:** Default choice, works well for IID data

**Configuration:**
```yaml
server:
  strategy: 'FedAvg'
  strategy_config:
    fraction_fit: 0.8              # Use 80% of clients per round
    fraction_evaluate: 0.5         # Evaluate on 50% of clients
    min_fit_clients: 8
    min_evaluate_clients: 4
    min_available_clients: 10
```

**Characteristics:**
- ✅ Simple and well-tested
- ✅ Good baseline performance
- ⚠️ Struggles with high data heterogeneity

**Reference:** McMahan et al. (2017)

---

### FedProx

**Description:** Federated Proximal - handles heterogeneous clients better

**Use case:** Non-IID data, clients with different computational capabilities

**Configuration:**
```yaml
server:
  strategy: 'FedProx'
  strategy_config:
    fraction_fit: 0.8
    min_fit_clients: 8
    min_available_clients: 10
    proximal_mu: 0.01  # Regularization term (0.001-0.1 typical)
```

**Characteristics:**
- ✅ Better for non-IID data
- ✅ More robust to stragglers
- ✅ Handles system heterogeneity
- ⚠️ Requires tuning `proximal_mu`

**Proximal mu parameter:**
- `proximal_mu=0.001`: Light regularization
- `proximal_mu=0.01`: Moderate regularization (recommended)
- `proximal_mu=0.1`: Strong regularization

**Reference:** Li et al. (2020)

---

### FedAdam

**Description:** Adaptive optimizer on the server side

**Use case:** When server-side optimization helps (similar to using Adam in centralized training)

**Configuration:**
```yaml
server:
  strategy: 'FedAdam'
  strategy_config:
    fraction_fit: 0.8
    min_fit_clients: 8
    min_available_clients: 10
    eta: 0.01      # Server learning rate
    eta_l: 0.01    # Client learning rate
    beta_1: 0.9    # First moment decay
    beta_2: 0.99   # Second moment decay
```

**Characteristics:**
- ✅ Adaptive learning rates
- ✅ Faster convergence in some scenarios
- ⚠️ More hyperparameters to tune

**Reference:** Reddi et al. (2020)

---

### FedAdagrad

**Description:** Adagrad-style server optimizer

**Use case:** Sparse gradients, varying feature scales

**Configuration:**
```yaml
server:
  strategy: 'FedAdagrad'
  strategy_config:
    fraction_fit: 0.8
    min_fit_clients: 8
    min_available_clients: 10
    eta: 0.01      # Server learning rate
    eta_l: 0.01    # Client learning rate
```

**Characteristics:**
- ✅ Adapts to feature scales
- ✅ Good for sparse data
- ⚠️ Learning rate may decay too quickly

**Reference:** Reddi et al. (2020)

---

## Privacy-Preserving FL (FL + Differential Privacy)

Combine federated learning with differential privacy for **formal privacy guarantees**.

### Why FL + DP?

- **FL alone**: Prevents direct data sharing but vulnerable to inference attacks
- **DP alone**: Provides privacy guarantees but requires centralized data
- **FL + DP**: Best of both worlds - decentralized training with provable privacy

### Configuration

```yaml
federated:
  clients:
    profiles:
      # Privacy-sensitive clients (e.g., hospitals with sensitive data)
      - id: [0, 1, 2]
        trainer_type: 'dp'
        batch_size: 16
        dp:
          noise_multiplier: 1.1    # Privacy noise level
          max_grad_norm: 1.0       # Gradient clipping
          target_epsilon: 3.0      # Privacy budget
          target_delta: 1e-5       # Privacy parameter

      # Non-sensitive clients (can train faster)
      - id: [3, 4, 5]
        trainer_type: 'standard'
        batch_size: 32
```

### Privacy Budget (Epsilon)

- **ε < 1**: Very strong privacy (high utility loss)
- **ε = 1-10**: Good privacy-utility tradeoff (recommended)
- **ε > 10**: Weaker privacy guarantees

### Key Points

- Only sensitive clients need to use DP
- Non-sensitive clients contribute more effectively (no DP overhead)
- Server aggregates both DP and non-DP updates
- Global model benefits from all clients

**Example scenario:**
- Hospital A, B, C have highly sensitive patient data → use `dp` trainer
- Clinics D, E, F have less sensitive data → use `standard` trainer
- All contribute to federated model training!

---

## Hyperparameter Search + FL

### Recommended Approach: Pre-FL Optimization

Run hyperparameter search **before** starting federated training:

```bash
# 1. Server-side global search (recommended)
ml-search --config configs/federated_base.yaml --n-trials 50

# 2. Use best config for FL training
ml-fl-run --config runs/optuna_studies/my_study/best_config.yaml
```

**Why this works:**
- ✅ Finds good global hyperparameters
- ✅ All clients benefit from optimization
- ✅ No synchronization issues
- ✅ Efficient use of compute

### Alternative: Per-Client Optimization

For heterogeneous clients with very different data/compute:

```bash
# Each client finds its own optimal hyperparameters
ml-search --config configs/client_0_base.yaml --n-trials 20
ml-search --config configs/client_1_base.yaml --n-trials 20

# Then join federation with optimized configs
ml-fl-run --config configs/federated_deployment.yaml
```

### NOT Recommended: During FL Rounds

❌ **Do not run Optuna during FL rounds**

Why:
- Breaks client synchronization
- Massive communication overhead
- Each trial requires multiple FL rounds
- Very slow and inefficient

---

## Monitoring and Logging

### TensorBoard Logging

Each client logs to its own directory:

```bash
# View all clients combined
tensorboard --logdir runs/simulation/

# View specific client
tensorboard --logdir runs/simulation/fl_client_0/tensorboard
```

**Per-client metrics:**
- Training loss and accuracy
- Validation loss and accuracy
- Learning rate
- EMA metrics (if enabled)
- Confusion matrices
- Classification reports

### Server-Side Metrics

Server logs aggregated metrics to console:

```
[Round 1] Aggregated fit metrics: {'train_loss': 0.532, 'train_acc': 0.812, ...}
[Round 1] Aggregated eval metrics: {'val_acc': 0.795}
[Round 2] Aggregated fit metrics: {'train_loss': 0.421, 'train_acc': 0.854, ...}
```

### Output Structure

```
runs/simulation/
├── fl_client_0/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── logs/
│   │   ├── train.log
│   │   └── classification_report_*.txt
│   └── tensorboard/
├── fl_client_1/
│   └── ...
└── fl_client_N/
    └── ...
```

---

## CLI Reference

### ml-fl-run

Unified launcher for simulation or deployment mode.

```bash
# Simulation mode (default)
ml-fl-run --config configs/federated_config.yaml

# Override number of rounds
ml-fl-run --config configs/federated_config.yaml --num-rounds 200

# Override number of clients (simulation only)
ml-fl-run --config configs/federated_config.yaml --num-clients 15

# Deployment mode
ml-fl-run --config configs/federated_deployment.yaml --mode deployment
```

---

### ml-fl-server

Start federated learning server manually.

```bash
# Basic usage
ml-fl-server --config configs/federated_config.yaml

# Custom server address
ml-fl-server --config configs/federated_config.yaml \\
  --server-address 0.0.0.0:9000

# Override number of rounds
ml-fl-server --config configs/federated_config.yaml --num-rounds 150
```

---

### ml-fl-client

Start federated learning client manually.

```bash
# Basic usage
ml-fl-client --config configs/federated_config.yaml --client-id 0

# Custom server address
ml-fl-client --config configs/federated_config.yaml \\
  --client-id 1 \\
  --server-address 10.0.0.1:8080

# Override trainer type
ml-fl-client --config configs/federated_config.yaml \\
  --client-id 2 \\
  --trainer-type dp

# Custom run directory
ml-fl-client --config configs/federated_config.yaml \\
  --client-id 3 \\
  --run-dir /custom/path/client_3
```

---

### ml-split (Federated Mode)

Generate federated data splits.

```bash
# IID partitioning
ml-split --raw_data data/my_dataset/raw \\
  --federated \\
  --num-clients 10 \\
  --partition-strategy iid

# Non-IID (Dirichlet)
ml-split --raw_data data/my_dataset/raw \\
  --federated \\
  --num-clients 10 \\
  --partition-strategy non-iid \\
  --alpha 0.5

# Label-skew
ml-split --raw_data data/my_dataset/raw \\
  --federated \\
  --num-clients 5 \\
  --partition-strategy label-skew \\
  --classes-per-client 2

# Custom seed
ml-split --raw_data data/my_dataset/raw \\
  --federated \\
  --num-clients 10 \\
  --seed 123
```

---

## Advanced Topics

### Callbacks in FL

All existing callbacks work with federated learning:

```yaml
training:
  callbacks:
    - type: 'early_stopping'
      monitor: 'val_acc'
      patience: 3

    - type: 'swa'
      swa_start_epoch: 3

    - type: 'gradient_clipping'
      value: 1.0
```

Each client applies callbacks independently during local training.

---

### EMA in FL

Exponential Moving Average works seamlessly with FL:

```yaml
training:
  ema:
    enabled: true
    decay: 0.9999
    warmup_steps: 0
```

Each client maintains its own EMA model during local training. The EMA weights are aggregated along with the regular weights.

---

### Resuming FL Training

FL training can be resumed from checkpoints:

1. **Client-side**: Each client saves `weights/last.pt` after every round
2. **Server-side**: Use `--num-rounds` to extend training

```bash
# Initial training (50 rounds)
ml-fl-run --config configs/fl_config.yaml --num-rounds 50

# Extend training (50 more rounds)
ml-fl-run --config configs/fl_config.yaml --num-rounds 100
```

---

## Troubleshooting

### Issue: Clients can't connect to server

**Symptoms:**
```
[Client 0] Connection refused: localhost:8080
```

**Solutions:**
1. Ensure server is running first
2. Check firewall settings
3. Verify server address in config
4. Try `0.0.0.0` instead of `localhost` for server address

---

###Issue: Out of memory in simulation mode

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce `num_clients`
2. Reduce batch size in client profiles
3. Set `num_gpus` to fraction in client resources:
   ```yaml
   # In config or via Flower simulation settings
   client_resources:
     num_gpus: 0.2  # Each client uses 20% of GPU
   ```

---

### Issue: Slow convergence

**Possible causes and solutions:**

1. **High data heterogeneity**
   - Switch to FedProx strategy
   - Increase `proximal_mu` parameter
   - Use more local epochs per round

2. **Too few clients per round**
   - Increase `fraction_fit` (e.g., 0.8 → 1.0)
   - Increase `min_fit_clients`

3. **Learning rate too low**
   - Run `ml-lr-finder` on representative client data
   - Adjust server LR (`eta`) for FedAdam/FedAdagrad

---

### Issue: Client stragglers

**Symptoms:** Some clients much slower than others

**Solutions:**
1. Use FedProx (more tolerant to stragglers)
2. Reduce `fraction_fit` to exclude slowest clients
3. Use heterogeneous profiles with appropriate batch sizes
4. Consider asynchronous FL (future feature)

---

## Best Practices

### 1. Start with Simulation

Always prototype in simulation mode before deployment:

```bash
ml-fl-run --config configs/fl_config.yaml  # Simulation first
```

### 2. Use FedProx for Non-IID Data

If your data is heterogeneous, use FedProx:

```yaml
server:
  strategy: 'FedProx'
  strategy_config:
    proximal_mu: 0.01
```

### 3. Run Optuna Before FL

Optimize hyperparameters before starting FL training:

```bash
ml-search --config configs/federated_base.yaml --n-trials 50
ml-fl-run --config runs/optuna_studies/my_study/best_config.yaml
```

### 4. Monitor Per-Client Performance

Use TensorBoard to identify problematic clients:

```bash
tensorboard --logdir runs/simulation/
```

Look for:
- Clients with unusually high loss
- Clients with poor validation accuracy
- Diverging training curves

### 5. Use DP Selectively

Only apply DP to clients that truly need privacy:

```yaml
profiles:
  - id: [0, 1]  # Sensitive data
    trainer_type: 'dp'
  - id: [2, 3]  # Less sensitive
    trainer_type: 'standard'
```

### 6. Balance Local Epochs vs Rounds

**More local epochs, fewer rounds:**
- ✅ Less communication overhead
- ❌ More client drift

**Fewer local epochs, more rounds:**
- ✅ Better convergence (more aggregation)
- ❌ More communication overhead

Typical: 5-10 local epochs per round

---

## Configuration Templates

### Basic Simulation (IID)

```yaml
federated:
  mode: 'simulation'

  server:
    strategy: 'FedAvg'
    num_rounds: 50
    strategy_config:
      fraction_fit: 1.0
      min_fit_clients: 5
      min_available_clients: 5

  clients:
    num_clients: 5

  partitioning:
    strategy: 'iid'
```

---

### Production Heterogeneous (Non-IID)

```yaml
federated:
  mode: 'simulation'

  server:
    strategy: 'FedProx'
    num_rounds: 100
    strategy_config:
      fraction_fit: 0.8
      min_fit_clients: 8
      min_available_clients: 10
      proximal_mu: 0.01

  clients:
    num_clients: 10
    profiles:
      - id: [0, 1, 2, 3, 4]
        trainer_type: 'standard'
        batch_size: 32

      - id: [5, 6]
        trainer_type: 'mixed_precision'
        batch_size: 64

      - id: [7, 8]
        trainer_type: 'dp'
        batch_size: 16
        dp:
          noise_multiplier: 1.1
          target_epsilon: 3.0

  partitioning:
    strategy: 'non-iid'
    alpha: 0.5
```

---

### Distributed Deployment

```yaml
federated:
  mode: 'deployment'

  server:
    address: '10.0.0.1:8080'
    strategy: 'FedAvg'
    num_rounds: 200
    strategy_config:
      fraction_fit: 0.8
      min_fit_clients: 8
      min_available_clients: 10

  clients:
    manifest:
      - id: 0
        config_override: 'configs/client_overrides/hospital_1.yaml'
      - id: 1
        config_override: 'configs/client_overrides/hospital_2_dp.yaml'
      - id: 2
        config_override: 'configs/client_overrides/hospital_3.yaml'

  partitioning:
    strategy: 'non-iid'
    alpha: 0.3
```

---

## Next Steps

- See [Advanced Training](advanced-training.md) for trainer types
- See [Hyperparameter Tuning](hyperparameter-tuning.md) for Optuna integration
- See [Configuration Reference](../configuration/training.md) for all options
- Check `ml_src/federated_config_template.yaml` for annotated examples

---

## References

- McMahan et al. (2017): "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Li et al. (2020): "Federated Optimization in Heterogeneous Networks (FedProx)"
- Reddi et al. (2020): "Adaptive Federated Optimization"
- Flower Documentation: https://flower.ai/docs/
