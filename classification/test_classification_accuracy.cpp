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
      std::vector<vitis::ai::ClassificationResult> output;
      std::vector<std::shared_ptr<Classification>> models;
      std::vector<std::string> ImageLines;
      ImageLines.clear();
      std::string line;
      int one_hit_count  = 0;
      string image_list = argv[2];
      int threads_num = atoi(argv[3]);
      std::vector<struct feature_param> id_features;
      id_features.clear();
      LoadFeatureNames(image_list, id_features);
      for (int i = 0; i < id_features.size(); i++)
      {
        ImageLines.emplace_back(id_features[i].image_path);
        //cout << ImageLines[i] << "   " << id_features[i].image_path << endl;
      }
      // std::vector<vitis::ai::ClassificationResult> output;
      // std::vector<std::shared_ptr<Classification>> models;
      models = model_init(model_path,threads_num);
      auto e2e_start = std::chrono::system_clock::now();
      output = run_xmodel(models,ImageLines,threads_num);
      auto act_time_load = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - e2e_start).count();
      // std::cout << "microseconds: " << act_time_load << std::endl;
      // std::cout << "Include load fps:" <<  1 / (act_time_load / 1e6 / ImageLines.size()) << std::endl;
      // std::cout << "output size: " << output.size() << std::endl;
      for(int i = 0; i < id_features.size(); i++)
      {
        bool hit_or_not = false;
        if(id_features[i].feature_id == output[i].scores[0].index)
            {
              hit_or_not = true;
              one_hit_count++;
            }	        
            else
            {    
              ;    
            //   cout << "image id:"<< id_features[i].feature_id << ", inference id:" << output[i].scores[0].index << endl;
            //   cout << "image path:" << id_features[i].image_path << endl;
            //   for(int j=0;j<3;j++)
            //   {
            //   cout << "index = " << output[i].scores[j].index  << ", scores = " << output[i].scores[j].score << endl;
            //  }
              //Mat image = imread(id_features[i].image_path);
              //string save_path = "./error_image"+ id_features[i].image_path.erase(0,5);
              //cout << save_path << endl;
              //imwrite(save_path,image);
            }
      }
      float accuracy = 0.0;
      accuracy  =  float(one_hit_count) / id_features.size();
      //cout <<  accuracy   <<  one_hit_count  <<  id_features.size() ;
      accuracy   = accuracy  * 100;
      cout <<" accuracy: "<< accuracy <<"%" << endl;
      cout << "output_size:" <<  output.size() << endl;
    return 0;
} 
