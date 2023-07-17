#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include "./common/common.hpp"
#include "./yolovx/yolox_v1.hpp"



using  namespace std;

int main(int argc, char* argv[]) {
    
    //judge the input parameters
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_list> <output_feature_list> <threads_num>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    string id_image_list = argv[2];
    string output_feature_list = argv[3];
    int threads_num = atoi(argv[4]);

    std::vector<vitis::ai::YOLOvXResult> output;
    std::vector<std::shared_ptr<YOLOvX>> models; 
    
    std::vector<std::string> ImageLines;
    ImageLines.clear();
    std::vector<struct feature_param> id_features;
    id_features.clear();

    LoadFeatureNames(id_image_list, id_features);
    for (int i = 0; i < id_features.size(); i++)
    {
      ImageLines.emplace_back(id_features[i].image_path);
      
    }
    
    models = model_init(model_path,threads_num);
    std::vector<struct input_image_param> input_params;
    LoadImageNames(id_image_list, input_params);
    output = run_xmodel(models,ImageLines,threads_num);
    ofstream out_id(output_feature_list);
    out_id << "[" ;
    for (size_t id =0; id < output.size(); id++)
    {
        // auto results = yolo->run(image);  //int label, flaot score, vector<float> box 
        for(auto& result: output[id].bboxes)
        {

            int label = result.label;
            auto& box = result.box;
           
            if (id != (input_params.size() - 1))
            {
                out_id<< "{" << "\"image_id\": " <<input_params[id].image_id << ", " << "\"category_id\": " << label << ", \"bbox\":" << "["  << \
                box[0] << "," << box[1] << "," << box[2]-box[0] << "," << box[3]-box[1] << "]," << "\"score\": " << result.score << ", \"segmentation\": []}, ";
            }
            else
            {
                out_id<< "{" << "\"image_id\": " <<input_params[id].image_id << ", " << "\"category_id\": " << label << ", \"bbox\":" << "["  << \
                box[0] << "," << box[1] << "," << box[2]-box[0] << "," << box[3]-box[1] << "]," << "\"score\": " << result.score << ", \"segmentation\": []} ";
            }
        }
    }
    out_id << "]";

    return 0;

} 
