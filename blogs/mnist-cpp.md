---
layout: default
title:
permalink: /blogs/mnist-cpp/
---

* A quick workaround to get started with `mnist` dataset in C++. 
* Next: Train a small MLP from scratch in CUDA/C++.

## Get the data: 

Sadly you can't download it from Yann Lecun's website anymore, it is hosted by Hugging Face now.

```bash
wget https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet
wget https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet
```


## Decoding the images with Python:

* Some requirements and a check-up of the files:

```bash
pip install pandas pyarrow
```


```python
import pandas as pd

# Load the MNIST Parquet files
train_df = pd.read_parquet('path/to/train_mnist.parquet')
test_df = pd.read_parquet('path/to/test_mnist.parquet')

# Save to CSV (simple text format)
train_df.to_csv('train_mnist.csv', index=False)
test_df.to_csv('test_mnist.csv', index=False)
```

Giving the `train_df.head()`:

```bash
                                               image  label
0  {'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHD...      5
1  {'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHD...      0
2  {'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHD...      4
3  {'bytes': b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHD...      1
4  {'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHD...      9
```

The `.parquet` files are contains the PNG images (encoded in bytes) with the associated labels. Some more magic, we'll basically decode them in raw pixels, then store as bytes.


```python
import pandas as pd
from PIL import Image
import io
import numpy as np

# Load the DataFrame
train = "train-00000-of-00001.parquet"
test  = "test-00000-of-00001.parquet"

train_df = pd.read_parquet(f'../mlp/{train}')
test_df  = pd.read_parquet(f'../mlp/{test}')

def save_images_and_labels_to_binary(df, output_path):
    # Open binary files for writing
    with open(output_path + '_images.bin', 'wb') as img_file, \
         open(output_path + '_labels.bin', 'wb') as lbl_file:
        
        for _, row in df.iterrows():
            # Decode PNG bytes to an image
            image_data = row['image']['bytes']
            image = Image.open(io.BytesIO(image_data))

            # Convert image to numpy array and normalize the pixel values
            image_array = np.array(image, dtype=np.uint8)  # 28 x 28 (and not flattened)
            
            # Write the raw image data and the label to binary files
            img_file.write(image_array.tobytes())
            lbl_file.write(np.array([row['label']], dtype=np.uint8))
            break
            

# Save data to binary files
save_images_and_labels_to_binary(train_df, 'train_mnist')
save_images_and_labels_to_binary(test_df, 'test_mnist')
```

By now you should have something like this in your working directory:

```bash
$ coding $ ls | grep ".bin$"
test_mnist_images.bin
test_mnist_labels.bin
train_mnist_images.bin
train_mnist_labels.bin
```

3. Loading in C++:

```cpp
#include <iostream>
#include <fstream>
#include <vector>

const int IMAGE_SIZE = 28 * 28; // Total pixels per image
const int NUM_IMAGES = 60000;   // Adjust based on the number of images you saved

int main() {

    std::vector<std::vector<uint8_t>> images(NUM_IMAGES, std::vector<uint8_t>(IMAGE_SIZE));
    std::vector<uint8_t> labels(NUM_IMAGES);

    std::ifstream imageFile("train_mnist_images.bin", std::ios::binary);
    std::ifstream labelFile("train_mnist_labels.bin", std::ios::binary);

    if (!imageFile || !labelFile) {
        std::cerr << "Error opening one or both files." << std::endl;
        return 1;
    }

    // Read the label /  image data
    labelFile.read(reinterpret_cast<char*>(labels.data()), NUM_IMAGES);

    for (int i = 0; i < NUM_IMAGES; ++i) {
        imageFile.read(reinterpret_cast<char*>(images[i].data()), IMAGE_SIZE);
    }

    imageFile.close();
    labelFile.close();

    int image_id;
    std::cout << "Image id = " ;
    std::cin >> image_id;

    std::cout << "Label of the first image: " << static_cast<int>(labels[image_id]) << std::endl;
    std::cout << "Image data:" << std::endl;
    for (int i = 0; i < 28; ++i) { // Print the first 28 bytes as the first row
        for (int j = 0; j < 28; ++j) {
            std::cout << (images[image_id][i * 28 + j] > 0 ? '#' : ' ');
        }
        std::cout << std::endl;
    }

    return 0;
}
```

Now you get this cool illustration from your terminal:

![mnist01](/src/mnist-blog-cpp/mnist01.png)








