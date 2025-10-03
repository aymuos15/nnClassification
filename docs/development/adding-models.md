# Adding Custom Models

Guide to creating and integrating custom model architectures.

## For Torchvision Models

**No code needed!** Just update config:

```yaml
model:
  type: 'base'
  architecture: 'efficientnet_b0'  # Any torchvision model
  num_classes: 10
  weights: 'DEFAULT'
```

All torchvision models are automatically supported.

## For Custom Models

### Step 1: Define Model Class

Edit `ml_src/network/custom.py`:

```python
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes, input_size=224, **kwargs):
        super().__init__()
        
        # Your architecture here
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... more layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 112 * 112, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### Step 2: Register Model

Add to `MODEL_REGISTRY` in `ml_src/network/custom.py`:

```python
MODEL_REGISTRY = {
    'simple_cnn': SimpleCNN,
    'tiny_net': TinyNet,
    'my_custom_model': MyCustomModel,  # Add here
}
```

### Step 3: Use in Config

```yaml
model:
  type: 'custom'
  custom_architecture: 'my_custom_model'
  num_classes: 10
  input_size: 224
```

### Step 4: Train

```bash
python train.py
```

## Example: ResNet-like Model

```python
class MyResNet(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

## Best Practices

1. **Inherit from `nn.Module`**
2. **Accept `num_classes` parameter**
3. **Return logits** (no softmax)
4. **Test forward pass** before training
5. **Document architecture**

## Testing Your Model

```python
# Test script
from ml_src.network import get_model
import torch

config = {
    'model': {
        'type': 'custom',
        'custom_architecture': 'my_custom_model',
        'num_classes': 10
    }
}

model = get_model(config, 'cpu')
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(f"Output shape: {y.shape}")  # Should be (2, 10)
```

## Related

- [Model Configuration](../configuration/models.md)
- [Network Package](../architecture/ml-src-modules.md)
