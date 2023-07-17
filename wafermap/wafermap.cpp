#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include "./common/common.hpp"
#include "./wafermap/wafermap.hpp"



using  namespace std;

int main(int argc, char* argv[]) {
    
    std::string model_path = argv[1];

    std::vector<std::string> ImageLines;
    std::ifstream fs(argv[2]);
    std::string line;
    while (getline(fs, line))
    {
        ImageLines.emplace_back(line);
    }
    int threads_num = atoi(argv[3]);
    std::string output_txt = argv[4];

    std::vector<std::vector<int>> output;
    std::vector<std::shared_ptr<Classification>> models;
    std::ofstream outfile(output_txt);
    models = model_init(model_path,threads_num);
    auto exe_start = std::chrono::system_clock::now();
    // cout << "Test......" << endl;
    output = run_xmodel(models,ImageLines,threads_num);
    auto act_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - exe_start).count();
    cout << "output size:  " << output.size() << endl;
    // cout << "output[0] size:  " << output[0].size() << endl;

    for(int j=0 ; j<output.size();j++)
    {
        //outfile << ImageLines[j] << " " ;
        for(int i=0;i<output[0].size();i++)
        {
            outfile << output[j][i] << " " ;
        }
        outfile << endl;
    }
    outfile.close();
    
    std::cout << "e2e time: " << act_time << std::endl;
    std::cout << "fps:" << 1 / (act_time / 1e6 / ImageLines.size()) << std::endl;

    return 0;
} 
