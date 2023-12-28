import numpy as np
np.random.seed(42)

def convolve2d(image, kernel, stride=1):
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = (img_height - kernel_height) // stride +1
    output_width = (img_width - kernel_width) // stride +1
    output = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(img[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width] * kernel)
    
    #normalize
    output_min, output_max = output.min(), output.max()
    output = (output - output_min) / (output_max - output_min) * 255

    return output
    
    
img = np.random.randint(0, 255, size=(20, 20)) 
kernel = np.random.rand(5, 5) 
feature_map = convolve2d(img, kernel)
print(feature_map)
