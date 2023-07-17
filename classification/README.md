# Usage

    mkdir build 
    cd build
    cmake ..
    make

After the compilation is completed, two executable programs will appear in the bin folder, one is used to test precision, and the other is used to test FPS.

- accuracy

```
    ./bin/test_classification_accuracy <model_path> <accuracy_image_list_path> <threads>
```
- fps

```
    ./bin/test_classification_fps <model_path> <image_list_path> <threads>
```

Note: <accuracy_image_list_path> need label