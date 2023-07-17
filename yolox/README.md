# Usage

    mkdir build 
    cd build
    cmake ..
    make

After the compilation is completed, two executable programs will appear in the bin folder, one is used to test precision, and the other is used to test FPS.

- accuracy

```
    ./bin/test_yoloxaccuracy <model_path> <accuracy_image_list_path> <threads> <output_json_file>
```
- fps

```
    ./bin/test_yolox_fps <model_path> <image_list_path> <threads>
```

Note: <accuracy_image_list_path> need label

