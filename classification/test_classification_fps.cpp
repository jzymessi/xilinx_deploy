#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include "./common/common.hpp"
#include "./classification/classification_v1.hpp"



using  namespace std;

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
    std::vector<std::string> ImageLines;
    ImageLines.clear();
    while (getline(fs, line))
    {
        ImageLines.emplace_back(line);
    }

    std::vector<vitis::ai::ClassificationResult> output;
    std::vector<std::shared_ptr<Classification>> models;
    models = model_init(model_path,threads_num);
    auto e2e_start = std::chrono::system_clock::now();
    output = run_xmodel(models,ImageLines,threads_num);
    auto act_time_load = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - e2e_start).count();
    std::cout << "microseconds: " << act_time_load << std::endl;
    std::cout << "Include load fps:" <<  1 / (act_time_load / 1e6 / ImageLines.size()) << std::endl;
    return 0;
} 
