import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_feature_map_on_image(image, feature_map, alpha=0.5):
    # Resize the feature map to match the size of the raw image
    feature_map = cv2.resize(feature_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    # Get the number of feature maps
    num_maps = feature_map.shape[-1]
    # Set the number of rows and columns in the plot
    num_rows = int(np.ceil(np.sqrt(num_maps)))
    num_cols = int(np.ceil(num_maps / num_rows))
    # Create a figure and axis
    fig, ax = plt.subplots(num_rows, num_cols)
    # Plot each feature map on top of the image
    for i, axi in enumerate(ax.flat):
        if i < num_maps:
            axi.imshow(image, cmap="gray")
            axi.imshow(feature_map[:, :, i], cmap="jet", alpha=alpha)
            axi.axis("off")
    # plt.show()
    plt.savefig('./vis.jpg')

# Example usage
image_3d = np.random.rand(128, 128, 128)
feature_map_3d = np.random.rand(8, 8, 8, 2)
# 0 plane
image_2d = image_3d[64,:,:]
feature_map_2d = feature_map_3d[4,:,:,:]
visualize_feature_map_on_image(image_2d, feature_map_2d)