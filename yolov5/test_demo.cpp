/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/demo_accuracy.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <vitis/ai/yolov3.hpp>


using  namespace std;
using  namespace cv;
int main(int argc, char* argv[]) {
    
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

    std::string model_path = argv[1];
    string image_list = argv[2];
    auto yolo = vitis::ai::YOLOv3::create(model_path, true);
    cv::Mat image = cv::imread(image_list);
    auto results = yolo->run(image);

   
    //     // auto results = yolo->run(image);  //int label, flaot score, vector<float> box 
        // cv::Mat image = cv::imread(ImageLines[id]);
        cv::Mat original_image = image;
          for (const auto bbox : results.bboxes) 
          {
            int label = bbox.label;
            float xmin = bbox.x * image.cols ;
            float ymin = bbox.y * image.rows ;
            float xmax = xmin + bbox.width * image.cols;
            float ymax = ymin + bbox.height * image.rows;
            float confidence = bbox.score;
            if (xmax > image.cols) xmax = image.cols;
            if (ymax > image.rows) ymax = image.rows;
            int color_b = int(array_color[label*3] * 255);
            int color_g = int(array_color[label*3+1] * 255);
            int color_r = int(array_color[label*3+2] * 255);
            cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax << "\t" << ymax << "\t" << confidence << endl;
            cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax),Scalar(color_b,color_g,color_r),2,2,0);
            }

        // for(auto& box: results.bboxes)
        // {
        //     int label = box.label;
        //     float confidence = box.score;

        //     float xmin = box.x * img.cols;
        //     float ymin = box.y * img.rows;
        //     float xmax = (box.x + box.width) * img.cols;
        //     float ymax = (box.y + box.height) * img.rows;
        //     if (xmin < 0) xmin = 1;
        //     if (ymin < 0) ymin = 1;
        //     if (xmax > img.cols) xmax = img.cols;
        //     if (ymax > img.rows) ymax = img.rows;

        //     int color_b = int(array_color[label*3] * 255);
        //     int color_g = int(array_color[label*3+1] * 255);
        //     int color_r = int(array_color[label*3+2] * 255);
            
        //     cout  << ", \"bbox\":[" << xmin << ", "
        //         << ymin << ", " << xmax - xmin << ", " << ymax - ymin
        //         << "], \"score\":" << confidence << "}" << endl;

        //     putText(original_image,v_class[label],Point(xmin,ymin-2),FONT_HERSHEY_SIMPLEX,0.5,Scalar(color_b,color_g,color_r),1);
        //     rectangle(original_image, Point(xmin,ymin),Point((xmax - xmin),(ymax - ymin)),Scalar(color_b,color_g,color_r),2,2,0);
        // }
        
        string save_output_imgpath = "./result.jpg"; 
        imwrite(save_output_imgpath,original_image);
    
    // out_id << "]";
    
    return 0;
} 
