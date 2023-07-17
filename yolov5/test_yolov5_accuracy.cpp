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
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_list> <threads_num> <output_feature_list>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    string id_image_list = argv[2];
    int threads_num = atoi(argv[3]);
    string output_feature_list = argv[4];
    
    std::vector<vitis::ai::YOLOv3Result> output;
    std::vector<std::shared_ptr<YOLOv3>> models; 
    
    std::vector<std::string> ImageLines;
    ImageLines.clear();
    std::vector<struct feature_param> id_features;
    id_features.clear();

    LoadFeatureNames(id_image_list, id_features);
    for (int i = 0; i < id_features.size(); i++)
    {
      ImageLines.emplace_back(id_features[i].image_path);
      //cout << ImageLines[i] << "   " << id_features[i].image_path << endl;
    }
    // cout << "aaaa" << endl;
    models = model_init(model_path,threads_num);
    std::vector<struct input_image_param> input_params;
    LoadImageNames(id_image_list, input_params);
    output = run_xmodel(models,ImageLines,threads_num);
    // cout << "output size: " << output.size() << endl;
    ofstream out_id(output_feature_list);
    out_id << "[" ;
    for (size_t id =0; id < output.size(); id++)
    {
    //     // auto results = yolo->run(image);  //int label, flaot score, vector<float> box 
        cv::Mat image = cv::imread(ImageLines[id]);
        cv::Mat original_image = image;

        for(auto& bbox: output[id].bboxes)
        {
            // cout << output[id].bboxes.size() << endl;
            int label = bbox.label;
            float xmin = bbox.x * image.cols ;
            float ymin = bbox.y * image.rows ;
            float xmax = xmin + bbox.width * image.cols;
            float ymax = ymin + bbox.height * image.rows;
            float confidence = bbox.score;
            // if (confidence < 0.3) continue;
            if (xmin < 0) xmin = 0;
            if (ymin < 0) ymin = 0;
            if (xmax > image.cols) xmax = image.cols;
            if (ymax > image.rows) ymax = image.rows;

            out_id<< "{" << "\"image_id\": " <<input_params[id].image_id << ", " << "\"category_id\": " << label << ", \"bbox\":" << "["  << \
            xmin << "," << ymin << "," << xmax-xmin << "," << ymax-ymin << "]," << "\"score\": " << confidence << ", \"segmentation\": []}, ";       
            
        }
    
    }
    out_id.seekp(-2,std::ios::end);
    out_id.put(' ');
    out_id << "]";

        // string str_jpg = ".jpg";
    //     string save_output_imgpath = "./output/" + to_string(id) + str_jpg;
    //     cout << save_output_imgpath << endl;
    //     imwrite(save_output_imgpath,original_image);
    // }
    // out_id << "]";
    
    return 0;
} 
