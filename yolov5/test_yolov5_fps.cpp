//./bin/test_demo ../yolov5_nano_pt/yolov5_nano_pt.xmodel name_id.txt 1
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include "./common/common.hpp"
#include "./yolov5/yolov5.hpp"


using  namespace std;
using  namespace cv;
int main(int argc, char* argv[]) {
    
    //判断输入参数是否正确
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_list> <threads_num>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::ifstream fs(argv[2]);
    int threads_num = atoi(argv[3]);

    std::string line;
    std::vector<vitis::ai::YOLOv3Result> output;
    std::vector<std::shared_ptr<YOLOv3>> models; 
    std::vector<std::string> ImageLines;
    while (getline(fs, line))
    {
        ImageLines.emplace_back(line);
    }
    // cout << "aaaa" << endl;
    models = model_init(model_path,threads_num);
    //time start
    auto exe_start = std::chrono::system_clock::now();
    output = run_xmodel(models,ImageLines,threads_num);
    //time end
    auto act_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - exe_start).count();
    std::cout << "e2e time: " << act_time << std::endl;
    std::cout << "fps:" << 1 / (act_time / 1e6 / ImageLines.size()) << std::endl;
    return 0;
} 
