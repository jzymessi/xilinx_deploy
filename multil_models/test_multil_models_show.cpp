#include <glog/logging.h>
#include <array>
#include <iostream>
#include <memory>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/yolovx.hpp>
#include <vitis/ai/segmentation.hpp>
#include <vitis/ai/demo.hpp>
#include <sys/time.h>
#include "./process_result.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>

using  namespace std;
using  namespace  cv; 
static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}

void LoadImageNames(std::string const &filename,
                    std::vector<std::string> &images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE *fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }
  
  fclose(fp);
}

int main(int argc, char* argv[]) {

    if (argc != 6) 
    {
        std::cerr << "usage: " << argv[0] << "\n yolox model name: " << argv[1]  << "\n segmentation model name: " << argv[2] << "\n image_list_file "  << argv[3] << "\n yolox result path: " << argv[4] << "\n segmentation result path: " <<  argv[5] << std::endl;
        return -1;
    }

    //coco class vector
    vector<string> v_class{};
    v_class.push_back("person");
    v_class.push_back("bicycle");
    v_class.push_back("car");
    v_class.push_back("motorcycle");
    v_class.push_back("airplane");
    v_class.push_back("bus");
    v_class.push_back("train");
    v_class.push_back("truck");
    v_class.push_back("boat");
    v_class.push_back("traffic light");
    v_class.push_back("fire hydrant");
    v_class.push_back("stop sign");
    v_class.push_back("parking meter");
    v_class.push_back("bench");
    v_class.push_back("bird");
    v_class.push_back("cat");
    v_class.push_back("dog");
    v_class.push_back("horse");
    v_class.push_back("sheep");
    v_class.push_back("cow");
    v_class.push_back("elephant");
    v_class.push_back("bear");
    v_class.push_back("zebra");
    v_class.push_back("giraffe");
    v_class.push_back("backpack");
    v_class.push_back("umbrella");
    v_class.push_back("handbag");
    v_class.push_back("tie");
    v_class.push_back("suitcase");
    v_class.push_back("frisbee");
    v_class.push_back("skis");
    v_class.push_back("snowboard");
    v_class.push_back("sports ball");
    v_class.push_back("kite");
    v_class.push_back("baseball bat");
    v_class.push_back("baseball glove");
    v_class.push_back("skateboard");
    v_class.push_back("surfboard");
    v_class.push_back("tennis racket");
    v_class.push_back("bottle");
    v_class.push_back("wine glass");
    v_class.push_back("cup");
    v_class.push_back("fork");
    v_class.push_back("knife");
    v_class.push_back("spoon");
    v_class.push_back("bowl");
    v_class.push_back("banana");
    v_class.push_back("apple");
    v_class.push_back("sandwich");
    v_class.push_back("orange");
    v_class.push_back("broccoli");
    v_class.push_back("carrot");
    v_class.push_back("hot dog");
    v_class.push_back("pizza");
    v_class.push_back("donut");
    v_class.push_back("cake");
    v_class.push_back("chair");
    v_class.push_back("couch");
    v_class.push_back("potted plant");
    v_class.push_back("bed");
    v_class.push_back("dining table");
    v_class.push_back("toilet");
    v_class.push_back("tv");
    v_class.push_back("laptop");
    v_class.push_back("mouse");
    v_class.push_back("remote");
    v_class.push_back("keyboard");
    v_class.push_back("cell phone");
    v_class.push_back("microwave");
    v_class.push_back("oven");
    v_class.push_back("toaster");
    v_class.push_back("sink");
    v_class.push_back("refrigerator");
    v_class.push_back("book");
    v_class.push_back("clock");
    v_class.push_back("vase");
    v_class.push_back("scissors");
    v_class.push_back("teddy bear");
    v_class.push_back("hair drier");
    v_class.push_back("toothbrush");

    //color
    std::array<float,240> array_color= {
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    };
    
    std::string yolo_model_path = argv[1];
    std::string segmentation_model_path = argv[2];
    double feature_time  = 0.0;

    auto yolo = vitis::ai::YOLOvX::create(yolo_model_path,true);
    int model_width_yolo = yolo->getInputWidth();
    int model_height_yolo = yolo->getInputHeight();
    std::cout << "yolo model width = " <<  model_width_yolo << std::endl;
    std::cout << "yolo model height = " << model_height_yolo << std::endl;


    auto segmentation = vitis::ai::Segmentation::create(segmentation_model_path,true);
    int model_width_segmentation = segmentation->getInputWidth();
    int model_height_segmentation = segmentation->getInputHeight();
    std::cout << "segmentation model width = " <<  model_width_segmentation << std::endl;
    std::cout << "segmentation model height = " << model_height_segmentation << std::endl;

    vector<string> names;
    LoadImageNames(argv[3], names);
    string yolox_result = argv[4];
    string segmentation_result = argv[5];

    for (auto name : names) 
    {

        cv::Mat image  = cv::imread(name);

        // if(image.rows != model_width_yolo || image.cols != model_height_yolo)
        // {
        //     cv::resize(image, image, cv::Size(model_width_yolo, model_height_yolo), 0, 0, cv::INTER_LINEAR);
        // }

        double tstart = get_current_time();
        auto results = yolo->run(image);  //int label, flaot score, vector<float> box 
        double tend = get_current_time();
        feature_time = tend - tstart;
        std::cout << "feature_time = " << feature_time << "ms" << std::endl;
        int i=0;
        for(auto& result: results.bboxes)
        {
            i++;
            int label = result.label;
            auto& box = result.box;
            int color_b = int(array_color[label*3] * 255);
            int color_g = int(array_color[label*3+1] * 255);
            int color_r = int(array_color[label*3+2] * 255);
            cout << color_b << " "<< color_g << " " <<color_r << endl;
            cout << "RESULT: " << label << "\t" << v_class[label] << "\t"<< std::fixed << std::setprecision(2) << box[0] << "\t"  << box[1] << "\t" << box[2] << "\t" << box[3] << "\t" <<std::setprecision(6) << result.score << "\n";
            // cout << "results size: " << results.size() << endl; 
            putText(image,v_class[label],Point(box[0],box[1]-2),FONT_HERSHEY_SIMPLEX,0.5,Scalar(color_b,color_g,color_r),1);
            rectangle(image, Point(box[0],box[1]),Point(box[2],box[3]),Scalar(color_b,color_g,color_r),2,2,0);
            //  mosaic picture
            
            

        }
        //cut picture 
        // Rect rect(box[0], box[1], box[2]-box[0], box[3]-box[1]);
        // cv::Mat cut_image = image(rect);
        // cv::Mat segmentation_image;
        // cout << "cut_image cols:" << cut_image.cols << endl;
        // cout << "cut_image rows:" << cut_image.rows << endl;
        // cv::imwrite("test.jpg",cut_image);
        // //resize picture
        // cv::resize(cut_image,segmentation_image,cv::Size(model_width_segmentation, model_height_segmentation), 0, 0, cv::INTER_LINEAR);
        // // use segmentation picture

        auto result_channel_3 = segmentation->run_8UC3(image);
        cout << "result.rows:" << result_channel_3.segmentation.rows << endl;
        cout << "result.cols:" << result_channel_3.segmentation.cols << endl;
        // for (auto y = 0; y < result_channel_3.segmentation.rows; y++) 
        // {
        //   for (auto x = 0; x < result_channel_3.segmentation.cols; x++) 
        //   {
        //     result_channel_3.segmentation.at<uchar>(y, x) *= 10;
        //   }              
        // }
        auto namesp = split(name, "/");
        cv::Mat segmentation_result_image = result_channel_3.segmentation;
        
        cv::resize(segmentation_result_image,segmentation_result_image,cv::Size(image.cols, image.rows), 0, 0, cv::INTER_LINEAR);

        
        // cv::imwrite(segmentation_result + "/" + to_string(i) + namesp[namesp.size() - 1], segmentation_result_image);
        cv::Mat  result_image;
        cv::addWeighted(image, 0.6, segmentation_result_image, 0.4, 0, result_image);
        cv::imwrite("output_show.jpg",result_image);
        // cv::Rect roi(box[0], box[1], segmentation_result_image.cols, segmentation_result_image.rows);
        // segmentation_result_image.copyTo(image(roi));
  // cv::imwrite(yolox_result + "/" + namesp[namesp.size() - 1], image);
    
    
    }

    return 0;

} 
