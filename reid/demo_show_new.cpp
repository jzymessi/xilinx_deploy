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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <vitis/ai/reid.hpp>
#include <sys/time.h>
#include <algorithm>

//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
vector<Mat> imageVector;

#define DEMO_SHOW 1

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

struct feature_param{
	
    std::string image_path;
    int feature_id;
    std::string camera_id;
    float feature_value[2048];
  
    feature_param()
    {
        feature_id = 0;
        camera_id = "0";
        image_path.clear();
	    memset(feature_value,0,sizeof(feature_value));
    }
};

struct feature_info{
	
    std::string image_path;
    int feature_id;
    std::string camera_id;
    float feature_score;
  
    feature_info()
    {
       image_path ;
       feature_score = 0;
       feature_id = 0;
       camera_id = "0";
    }
};

//升序排列
bool LessSort (struct feature_info a, struct feature_info b) 
{ 
    return (a.feature_score < b.feature_score); 
}

//降序排列
bool GreaterSort (struct feature_info a, struct feature_info b) 
{ 
    return (a.feature_score > b.feature_score); 
}

bool Judge_Hit_Or_Not(vector<struct feature_info> feature_vec, int expected_index, int top_k)
{
	bool hit_or_not = false;
	if(feature_vec.size() < top_k){
		std::cout << "The top_k is out of range!" << std::endl;
		return hit_or_not;
	}

  for(int i =0; i < top_k; i++){
	   if(expected_index == feature_vec[i].feature_id){
		   hit_or_not = true;
		   break;
	   }	   
   }
   return hit_or_not;
}

void LoadImageNames(std::string const& filename,
                    std::vector<struct input_image_param>& images_param) {
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

void LoadFeatureNames(std::string const& filename, std::vector<struct feature_param>& id_features) 
{
	id_features.clear();
		
	/*Check if path is a valid directory path. */	
	ifstream input( filename.c_str());  
    if ( !input ) {   
		fprintf(stdout, "open file: %s  error\n", filename.c_str());
        exit(1);  
    }
	
    std::string line;
	while ( getline(input, line) ) 
    {
  	    //std::cout<<line<<std::endl;
		struct feature_param param;
  	
       stringstream ss(line);  
       ss >> param.image_path;
       ss >> param.feature_id; 
       ss >> param.camera_id;
       for(int i =0; i!=2048; i++){
           ss >> param.feature_value[i];
       }
  	   id_features.push_back(param);
	}	
    input.close();	
}

					
float calculCosineSimilar(float *feature1, float *feature2, int feature_dim) {
    float ret = 0.0, mod1 = 0.0, mod2 = 0.0;
    for (std::vector<double>::size_type i = 0; i != feature_dim; i++) {
        ret += feature1[i] * feature2[i];
        mod1 += feature1[i] * feature1[i];
        mod2 += feature2[i] * feature2[i];
    }
    return ret / (sqrt(mod1) * sqrt(mod2) );
}

float calculEuclideanDistance(float *feature1, float *feature2, int feature_dim) {
    float diff = 0.0, sum = 0.0;
    for (std::vector<double>::size_type i = 0; i < feature_dim; i++) {
        diff = feature1[i] - feature2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}



static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void multipleImage(vector<Mat> imgVector, Mat& dst, int imgCols) 
{
    const int MAX_PIXEL=300;
    int imgNum = imgVector.size();
    //选择图片最大的一边 将最大的边按比例变为300像素
    Size imgOriSize = imgVector[0].size();
    int imgMaxPixel = max(imgOriSize.height, imgOriSize.width);
    //获取最大像素变为MAX_PIXEL的比例因子
    double prop = imgMaxPixel < MAX_PIXEL ?  (double)imgMaxPixel/MAX_PIXEL : MAX_PIXEL/(double)imgMaxPixel;
    Size imgStdSize(imgOriSize.width * prop, imgOriSize.height * prop); //窗口显示的标准图像的Size

    Mat imgStd; //标准图片
    Point2i location(0, 0); //坐标点(从0,0开始)
    //构建窗口大小 通道与imageVector[0]的通道一样
    Mat imgWindow(imgStdSize.height*((imgNum-1)/imgCols + 1), imgStdSize.width*imgCols, imgVector[0].type());
    for (int i=0; i<imgNum; i++)
    {
        location.x = (i%imgCols)*imgStdSize.width;
        location.y = (i/imgCols)*imgStdSize.height;
        resize(imgVector[i], imgStd, imgStdSize, prop, prop, INTER_LINEAR); //设置为标准大小
        //string text = to_string(input_params[search_i].feature_id);
        //cv::Point origin;
        //origin.x = location.x;
        //origin.y = location.y+30;
        imgStd.copyTo( imgWindow( Rect(location, imgStdSize) ) );
        //cv::putText(imgWindow,text,origin,cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,255,255),2,8);
    }
    dst = imgWindow;
}


int main(int argc, char* argv[]) 
{
    if (argc < 4) {
      std::cout << "usage : " 
	    << argv[0] 
		<< " .xmodel"
        << " imagefile"
  		<< " featurefile"
        << std::endl;
      return -1;
    }
    
    string id_image_list_file = "search.txt";
    string id_feature_list_file = argv[3];
  
    

    bool preprocess = !(getenv("PRE") != nullptr);
    auto facefeature = vitis::ai::Reid::create(argv[1], preprocess );
    int modelwidth = facefeature->getInputWidth();
    int modelheight = facefeature->getInputHeight();
  
    int search_i = atoi(argv[2]);
    std::vector<struct input_image_param> input_params;
    LoadImageNames(id_image_list_file, input_params);
    cv::Mat img = cv::imread(input_params[search_i].image_path,1);
    //cv::imshow("PIC",img);
    cv::Mat dst;
    std::vector<struct feature_param> id_features;
    LoadFeatureNames(id_feature_list_file, id_features);
	
	  //count precision 
	int one_hit_count =0;
    int three_hit_count = 0;
    int five_hit_count = 0;
    int ten_hit_count = 0;
    int thirty_hit_count = 0;
    int fifty_hit_count = 0;
    
    std::stringstream str2int;


    double calcul_time_total = 0; 
    vector<string> Text_id;
    vector<struct feature_info> feature_vec;      

        cv::Mat image = cv::imread(input_params[search_i].image_path);
        std::cout << input_params[search_i].image_path << std::endl;
        std::cout << input_params[search_i].feature_id << std::endl;
        
        if (image.empty()) {
            std::cout << "cannot load " << input_params[search_i].image_path << std::endl;
            //continue;
        }
      
        if(image.rows != modelwidth || image.cols != modelheight)
      	{
      	    cv::resize(image, image, cv::Size(modelwidth, modelheight), 0, 0, cv::INTER_LINEAR);
      	}
         
      	double tstart = get_current_time(); 
        auto result = facefeature->run(image);
        
        cv::Mat features = result.feat;

        //get feature val in Mat
        float feat_at[2048] = {0};
        for (int i = 0 ; i < features.rows ; i ++ )
        {
            for (int j = 0 ; j < features.cols ; j++ )
            {
                feat_at[j] = features.at<float>(i,j);
            } 

        } 
        
		float* feature_qur = &feat_at[0];
        feature_vec.clear();
        for(size_t i =0; i< id_features.size(); i++)
        {
            double start = get_current_time(); 
        	float score = calculCosineSimilar(id_features[i].feature_value, feature_qur, 2048);
            double end = get_current_time();
            double cur = end - start;
            calcul_time_total += cur;
            struct feature_info param;
            param.image_path = id_features[i].image_path;
            param.feature_id = id_features[i].feature_id;
            param.camera_id = id_features[i].camera_id;
            param.feature_score = score;
            feature_vec.push_back(param);
            //}
        }
        std::sort (feature_vec.begin(), feature_vec.end(), GreaterSort); 
 
        		
   	 int k = 0;
	 int i=0;
     
	cv::Mat search_img = cv::imread(input_params[search_i].image_path,1); 
    //cout << input_params[search_i].image_path << endl;
    string text = to_string(input_params[search_i].feature_id);
    cv::Point origin;
    origin.x = 0;
    origin.y = 30;
    cv::namedWindow("search",0);
    cv::resizeWindow("search",128,256);
    cv::putText(search_img,text,origin,cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,255,255),2,8);
	cv::imshow("search", search_img);
    cv::waitKey(0);
    int Text[6] = {0};
         while(1)
         {  
            std::cout << "Show "<< (i+1) << std::endl;
            std::cout << feature_vec[i].camera_id << std::endl;
            std::cout << input_params[search_i].camera_id<< std::endl;
            if(feature_vec[i].camera_id==input_params[search_i].camera_id)
            {
               ;
            }
            else
	        {
                std::cout<<feature_vec[i].feature_score<<"	"<<feature_vec[i].feature_id << "  " <<feature_vec[i].image_path << std::endl;
                std::cout<<feature_vec[i].image_path << std::endl;
                string show_path = feature_vec[i].image_path;
                cv::Mat img_show = cv::imread(show_path,1);
                std::cout<<"Path"<< show_path <<std::endl;
                if (img_show.empty())
                {
                    std::cout<<"Empty"<<std::endl;
  
                    break;
                }
                string Ranking =to_string(k) + "  ID:" +to_string(feature_vec[i].feature_id);
                //Text1 = to_string(feature_vec[i].feature_id)
                imageVector.push_back(img_show);
                //Text_id.push_back(feature_vec[i].feature_id);
                Text[k] = feature_vec[i].feature_id;
		        k++ ;
		
            } 
		i = i +1 ;
		if(k >= 6)
		{
		  break;
		}
        }
        cv::Mat dst_new;
        multipleImage(imageVector, dst, 3);
        cv::resize(dst,dst_new,Size(384,512),0,0,INTER_LINEAR);
        cv::namedWindow("multipleWindow",0);
        cv::resizeWindow("multipleWindow",384,512);
        
        cv::Point origin0;
        origin0.x = 0;
        origin0.y = 30;
        cv::Point origin1;
        origin1.x = 128;
        origin1.y = 30;
        cv::Point origin2;
        origin2.x = 256;
        origin2.y = 30;
        cv::Point origin3;
        origin3.x = 0;
        origin3.y = 286;
        cv::Point origin4;
        origin4.x = 128;
        origin4.y = 286;
        cv::Point origin5;
        origin5.x = 256;
        origin5.y = 286;

        // cout << "text" << Text[0] << endl;
        // cout << "text" << Text[1] << endl;
        // cout << "text" << Text[2] << endl;
        // cout << "text" << Text[3] << endl;
        // cout << "text" << Text[4] << endl;
        // cout << "text" << Text[5] << endl;
        
        // cout << "dstsize" << dst_new.cols << endl;
        // cout << "dstsize" << dst_new.rows << endl;

        cv::putText(dst_new,to_string(Text[0]),origin0,cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,255,255),2,8);
        cv::putText(dst_new,to_string(Text[1]),origin1,cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,255,255),2,8);
        cv::putText(dst_new,to_string(Text[2]),origin2,cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,255,255),2,8);
        cv::putText(dst_new,to_string(Text[3]),origin3,cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,255,255),2,8);
        cv::putText(dst_new,to_string(Text[4]),origin4,cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,255,255),2,8);
        cv::putText(dst_new,to_string(Text[5]),origin5,cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,255,255),2,8);

        cv::imshow("multipleWindow", dst_new);
        cv::waitKey(0);    
        cv::destroyAllWindows();

  return 0;
}
