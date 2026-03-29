import matplotlib.pyplot as plt

def plot_feature_map(feature, num_channels=6):
    '''
    Plot feature map safely
    '''
    # Handle shape
    if feature.dim() == 4:
        feature = feature[0]  # [1, C, H, W] -> [C, H, W]
    elif feature.dim() == 3:
        pass  # already [C, H, W]
    else:
        raise ValueError(f"Invalid feature shape: {feature.shape}")

    feature = feature.cpu()

    C = feature.shape[0]
    num_channels = min(num_channels, C)

    plt.figure(figsize=(15, 5))

    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)

        fmap = feature[i].detach().numpy()

        # normalize
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)

        plt.imshow(fmap, cmap='viridis')
        plt.axis('off')
        plt.title(f"Ch {i}")

    plt.tight_layout()
    plt.show()