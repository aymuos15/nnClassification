"""
Comprehensive tests for torchvision architecture support.

Tests validate the claim: "All torchvision models are automatically supported
with proper final layer adaptation."

Tests verify:
1. Model instantiation via get_model()
2. Final layer replacement for custom num_classes
3. Forward pass functionality
4. Correct output shape [batch_size, num_classes]
5. Both pretrained and random initialization
"""

import pytest
import torch

from ml_src.core.network import get_model

# Test architectures organized by family
# Using representative models from each family to ensure comprehensive coverage
# while keeping test execution time reasonable

RESNET_FAMILY = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

VGG_FAMILY = [
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]

EFFICIENTNET_FAMILY = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
]

MOBILENET_FAMILY = [
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
]

VIT_FAMILY = [
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
]

SWIN_FAMILY = [
    "swin_t",
    "swin_s",
    "swin_b",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]

CONVNEXT_FAMILY = [
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]

DENSENET_FAMILY = [
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]

REGNET_FAMILY = [
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_1_6gf",
    "regnet_x_3_2gf",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
    "regnet_y_3_2gf",
]

MNASNET_FAMILY = [
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
]

SHUFFLENET_FAMILY = [
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]

OTHER_ARCHITECTURES = [
    "alexnet",
    "squeezenet1_0",
    "squeezenet1_1",
    "googlenet",
    "maxvit_t",
]

# Special case: Inception v3 requires 299x299 input
INCEPTION_FAMILY = [
    "inception_v3",
]

# Combine all standard 224x224 architectures
ALL_STANDARD_ARCHITECTURES = (
    RESNET_FAMILY
    + VGG_FAMILY
    + EFFICIENTNET_FAMILY
    + MOBILENET_FAMILY
    + VIT_FAMILY
    + SWIN_FAMILY
    + CONVNEXT_FAMILY
    + DENSENET_FAMILY
    + REGNET_FAMILY
    + MNASNET_FAMILY
    + SHUFFLENET_FAMILY
    + OTHER_ARCHITECTURES
)


class TestArchitectureInstantiation:
    """Test that all architectures can be instantiated."""

    @pytest.mark.parametrize("architecture", ALL_STANDARD_ARCHITECTURES)
    def test_standard_architecture_loads(
        self, architecture, device, num_classes_small, base_config_template
    ):
        """Test that standard 224x224 architectures load successfully."""
        config = base_config_template.copy()
        config["model"]["architecture"] = architecture
        config["model"]["num_classes"] = num_classes_small
        config["model"]["weights"] = None

        # Should not raise any exceptions
        model = get_model(config, device)
        assert model is not None

    @pytest.mark.parametrize("architecture", INCEPTION_FAMILY)
    def test_inception_architecture_loads(
        self, architecture, device, num_classes_small, base_config_template
    ):
        """Test that Inception v3 (299x299) loads successfully."""
        config = base_config_template.copy()
        config["model"]["architecture"] = architecture
        config["model"]["num_classes"] = num_classes_small
        config["model"]["weights"] = None

        # Should not raise any exceptions
        model = get_model(config, device)
        assert model is not None


class TestFinalLayerReplacement:
    """Test that final layers are correctly replaced for custom num_classes."""

    @pytest.mark.parametrize(
        "architecture",
        [
            "resnet18",
            "vgg16",
            "efficientnet_b0",
            "mobilenet_v2",
            "densenet121",
            "convnext_tiny",
            "swin_t",
            "vit_b_16",
        ],
    )
    @pytest.mark.parametrize("num_classes", [2, 10, 100, 1000])
    def test_num_classes_replacement(self, architecture, num_classes, device, base_config_template):
        """Test that final layer is adapted to match num_classes."""
        config = base_config_template.copy()
        config["model"]["architecture"] = architecture
        config["model"]["num_classes"] = num_classes
        config["model"]["weights"] = None

        model = get_model(config, device)

        # Create dummy input and get output shape
        if architecture == "inception_v3":
            dummy_input = torch.randn(1, 3, 299, 299).to(device)
        else:
            dummy_input = torch.randn(1, 3, 224, 224).to(device)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        # Verify output shape matches num_classes
        assert output.shape == (
            1,
            num_classes,
        ), f"Expected output shape (1, {num_classes}), got {output.shape}"


class TestForwardPass:
    """Test that forward passes work correctly."""

    @pytest.mark.parametrize("architecture", ALL_STANDARD_ARCHITECTURES)
    def test_standard_forward_pass(
        self,
        architecture,
        device,
        num_classes_small,
        base_config_template,
        sample_input_224,
        batch_size,
    ):
        """Test forward pass with standard 224x224 input."""
        config = base_config_template.copy()
        config["model"]["architecture"] = architecture
        config["model"]["num_classes"] = num_classes_small
        config["model"]["weights"] = None

        model = get_model(config, device)
        model.eval()

        with torch.no_grad():
            output = model(sample_input_224)

        # Check output shape
        assert output.shape == (
            batch_size,
            num_classes_small,
        ), f"Expected shape ({batch_size}, {num_classes_small}), got {output.shape}"

        # Check output is valid (no NaN or Inf)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    @pytest.mark.parametrize("architecture", INCEPTION_FAMILY)
    def test_inception_forward_pass(
        self,
        architecture,
        device,
        num_classes_small,
        base_config_template,
        sample_input_299,
        batch_size,
    ):
        """Test forward pass with 299x299 input for Inception v3."""
        config = base_config_template.copy()
        config["model"]["architecture"] = architecture
        config["model"]["num_classes"] = num_classes_small
        config["model"]["weights"] = None

        model = get_model(config, device)
        model.eval()

        with torch.no_grad():
            # Inception v3 returns InceptionOutputs during training, tensor during eval
            output = model(sample_input_299)

        # Check output shape
        assert output.shape == (
            batch_size,
            num_classes_small,
        ), f"Expected shape ({batch_size}, {num_classes_small}), got {output.shape}"

        # Check output is valid
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"


class TestPretrainedWeights:
    """Test that pretrained weights can be loaded."""

    @pytest.mark.parametrize(
        "architecture",
        [
            "resnet18",
            "resnet50",
            "vgg16",
            "efficientnet_b0",
            "mobilenet_v2",
            "densenet121",
        ],
    )
    def test_pretrained_loading(
        self, architecture, device, num_classes_small, base_config_template
    ):
        """Test that models load successfully with pretrained weights."""
        config = base_config_template.copy()
        config["model"]["architecture"] = architecture
        config["model"]["num_classes"] = num_classes_small
        config["model"]["weights"] = "DEFAULT"

        # Should not raise exceptions
        model = get_model(config, device)
        assert model is not None

        # Verify forward pass still works after final layer replacement
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, num_classes_small)


class TestBatchSizes:
    """Test that models handle various batch sizes correctly."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("architecture", ["resnet18", "efficientnet_b0", "mobilenet_v2"])
    def test_various_batch_sizes(
        self, architecture, batch_size, device, num_classes_small, base_config_template
    ):
        """Test that models handle different batch sizes."""
        config = base_config_template.copy()
        config["model"]["architecture"] = architecture
        config["model"]["num_classes"] = num_classes_small
        config["model"]["weights"] = None

        model = get_model(config, device)
        model.eval()

        # Create input with specific batch size
        input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (
            batch_size,
            num_classes_small,
        ), f"Expected shape ({batch_size}, {num_classes_small}), got {output.shape}"


class TestArchitectureFamilies:
    """Test coverage across all documented architecture families."""

    def test_resnet_family_coverage(self, device, num_classes_small, base_config_template):
        """Test that ResNet family is supported."""
        for arch in ["resnet18", "resnext50_32x4d", "wide_resnet50_2"]:
            config = base_config_template.copy()
            config["model"]["architecture"] = arch
            config["model"]["num_classes"] = num_classes_small
            model = get_model(config, device)
            assert model is not None

    def test_vgg_family_coverage(self, device, num_classes_small, base_config_template):
        """Test that VGG family is supported."""
        for arch in ["vgg16", "vgg16_bn"]:
            config = base_config_template.copy()
            config["model"]["architecture"] = arch
            config["model"]["num_classes"] = num_classes_small
            model = get_model(config, device)
            assert model is not None

    def test_efficientnet_family_coverage(self, device, num_classes_small, base_config_template):
        """Test that EfficientNet family is supported."""
        for arch in ["efficientnet_b0", "efficientnet_v2_s"]:
            config = base_config_template.copy()
            config["model"]["architecture"] = arch
            config["model"]["num_classes"] = num_classes_small
            model = get_model(config, device)
            assert model is not None

    def test_mobilenet_family_coverage(self, device, num_classes_small, base_config_template):
        """Test that MobileNet family is supported."""
        for arch in ["mobilenet_v2", "mobilenet_v3_large"]:
            config = base_config_template.copy()
            config["model"]["architecture"] = arch
            config["model"]["num_classes"] = num_classes_small
            model = get_model(config, device)
            assert model is not None

    def test_transformer_family_coverage(self, device, num_classes_small, base_config_template):
        """Test that Transformer families (ViT, Swin) are supported."""
        for arch in ["vit_b_16", "swin_t"]:
            config = base_config_template.copy()
            config["model"]["architecture"] = arch
            config["model"]["num_classes"] = num_classes_small
            model = get_model(config, device)
            assert model is not None

    def test_convnext_family_coverage(self, device, num_classes_small, base_config_template):
        """Test that ConvNeXt family is supported."""
        config = base_config_template.copy()
        config["model"]["architecture"] = "convnext_tiny"
        config["model"]["num_classes"] = num_classes_small
        model = get_model(config, device)
        assert model is not None

    def test_densenet_family_coverage(self, device, num_classes_small, base_config_template):
        """Test that DenseNet family is supported."""
        config = base_config_template.copy()
        config["model"]["architecture"] = "densenet121"
        config["model"]["num_classes"] = num_classes_small
        model = get_model(config, device)
        assert model is not None


class TestErrorHandling:
    """Test error handling for invalid configurations."""

    def test_invalid_architecture_raises_error(
        self, device, num_classes_small, base_config_template
    ):
        """Test that invalid architecture name raises appropriate error."""
        config = base_config_template.copy()
        config["model"]["architecture"] = "nonexistent_model_12345"
        config["model"]["num_classes"] = num_classes_small

        with pytest.raises(ValueError, match="not found in torchvision.models"):
            get_model(config, device)

    def test_invalid_model_type_raises_error(self, device, num_classes_small, base_config_template):
        """Test that invalid model type raises appropriate error."""
        config = base_config_template.copy()
        config["model"]["type"] = "invalid_type"
        config["model"]["num_classes"] = num_classes_small

        with pytest.raises(ValueError, match="Invalid model type"):
            get_model(config, device)
