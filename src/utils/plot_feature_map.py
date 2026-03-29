import matplotlib.pyplot as plt

def plot_feature_map(feature, num_channels=6):
    '''
    Util function for ploting feature map
    Args:
        feature: The feature map we want to plot
        num_channels: Number of channels
    '''
    feature = feature[0].cpu()  # [C, H, W]

    plt.figure(figsize=(15, 5))

    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        fmap = feature[i]

        # normalized to [0,1]
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)

        plt.imshow(fmap, cmap='viridis')
        plt.axis('off')
        plt.title(f"Ch {i}")

    plt.show()