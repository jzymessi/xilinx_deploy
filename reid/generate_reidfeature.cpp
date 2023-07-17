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

#include <stdio.h>
#include <sstream>  
#include <fstream>
#include <sys/time.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <vitis/ai/reid.hpp>
using namespace std;

struct input_image_param{
	
    std::string image_path;
    int feature_id;
    std::string camera_id;
    input_image_param()
    {
        feature_id = 0;
        camera_id = "0";
        image_path.clear();
    }
};

static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void LoadImageNames(std::string const& filename,
                    std::vector<struct input_image_param>& images_param ) {
  images_param.clear();

  /*Check if path is a valid directory path. */	
  ifstream input( filename.c_str());  
  if ( !input ) {   
	fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);  
  }
  
  std::string line;
  while ( getline(input, line) ) {

	stringstream ss(line); 
	struct input_image_param param;
	ss >> param.image_path;
  ss >> param.feature_id; 	
	ss >> param.camera_id; 
	images_param.push_back(param);
  }
  
  input.close();		
}

int main(int argc, char* argv[]) 
{
    if (argc < 3) {
      std::cout << "usage : " << argv[0] << " .xmodel"
                << " <image_list_file> <output_feature_list> "
                << std::endl;
      return -1;
    }
    string id_image_list = argv[2];
    string output_feature_list = argv[3];

    bool preprocess = !(getenv("PRE") != nullptr);
    auto facefeature = vitis::ai::Reid::create(argv[1], true);
    int width = facefeature->getInputWidth();
    int height = facefeature->getInputHeight();
    std::cout<<"width: "<<width<<"height: "<<height<<std::endl;
    std::vector<struct input_image_param> input_params;
    LoadImageNames(id_image_list, input_params);
    double total_feature_time  =0.0;  
    int id_num = 0;
    ofstream out_id(output_feature_list);
    for (size_t id =0; id < input_params.size(); id++) 
    {
      cv::Mat image = cv::imread(input_params[id].image_path);
      //std::cout << "PIC path = " << input_params[id].image_path << endl;
      if (image.empty()) {
        std::cout << "cannot load " << input_params[id].image_path << std::endl;
        continue;
    }
      
    cv::Mat img_resize;

    cv::resize(image, img_resize, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

    //cv::imshow("1111",img_resize);
    //cv::waitKey(0);
    double tstart = get_current_time(); 
    auto result = facefeature->run(img_resize);
    cv::Mat feat = result.feat;
    double tend = get_current_time();
    double costtime = tend - tstart;

    total_feature_time = total_feature_time + costtime;
    
    out_id  << input_params[id].image_path << " " <<  input_params[id].feature_id  << " " << input_params[id].camera_id << " "; 
    //std::cout<<"************" << feat.rows<<std::endl;
	  //std::cout<<"************" << feat.cols<<std::endl;
    for (int i = 0 ; i < feat.rows ; i ++ )
    {
      for (int j = 0 ; j < feat.cols ; j++ )
      {
        out_id<<feat.at<float>(i,j)<<" ";
	
	//std::cout<<feat.at<float>(i,j)<<std::endl;
      } 

    } 
    out_id << std::endl;
	
  }
  std::cout << "avg_feature_time: "<< (total_feature_time/(int)input_params.size())<< " ms"<<std::endl;
	
  return 0;
}
