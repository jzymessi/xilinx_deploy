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
#include <emmintrin.h>
#include <math.h>
#include <stdlib.h>
#include <memory>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
vector<Mat> imageVector;

#define DEMO_SHOW 1

void* MallocAlign16(size_t size)
{
    int ptrSize = sizeof(void*);


    byte* ptr =(byte*) malloc(size + 16 + ptrSize);
    byte* alignedPtr = (byte*) ( ((size_t) ptr) + 15 & ~15);
    if( (alignedPtr - ptr) < ptrSize)
    {
        alignedPtr += 16;
    }

    *((size_t*) (alignedPtr - ptrSize)) = (size_t) ptr;
    return (void*) alignedPtr;
}

float bytesTofloat(byte bytes[])
{

    return *((float*)bytes);
}

void FreeAlign16(void* ptr)
{
    int ptrSize = sizeof(void*);
    free( (void *) *((size_t *) (( (byte *) ptr ) - ptrSize)) );

}
float calculCosineSimilar_2048(byte* feature1, byte* feature2) {
   	float partial[4];
    //byte temp[12];
     
    size_t length = sizeof(float);
    byte temp1[4];
    byte temp2[4];
    
    for (int i = 0; i != 2048; i+=4) {
        //__builtin_prefetch(feature1+i+2048, 0, 3); 
		//__builtin_prefetch(feature2+i+2048, 0, 3); 
		
		memcpy(temp1, &feature1[i], 4);
		memcpy(temp2, &feature2[i], 4);
		
		float data1 = bytesTofloat(temp1);
		float data2 = bytesTofloat(temp2);
		
        partial[0] += data1 * data2;
        partial[1] += data1 * data1;
        partial[2] += data2 * data2;    
    }
  
    return partial[0] / (sqrt(partial[1]) * sqrt(partial[2]) );
}

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
    
    //string id_image_list_file = "search.txt";
    string id_feature_list_file = argv[2];
    string search_image_path = argv[3];
    string search_image_camera_id = argv[4];
    string output_result_list = argv[5];
    
    ofstream out_id(output_result_list);
    bool preprocess = !(getenv("PRE") != nullptr);
    auto facefeature = vitis::ai::Reid::create(argv[1], preprocess );
    int modelwidth = facefeature->getInputWidth();
    int modelheight = facefeature->getInputHeight();
  
    int search_i = atoi(argv[2]);

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

    cv::Mat image = cv::imread(search_image_path,1);
    std::cout << search_image_path << std::endl;
        
    if (image.empty()) {
        std::cout << "cannot load " << search_image_path << std::endl;
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

        //SSE optimizer
        /*
        void* pfeature1 = MallocAlign16(2048);
        void* pfeature2 = MallocAlign16(2048);
        memcpy(pfeature1, &id_features[i].feature_value[0], 2048);
        memcpy(pfeature2, &feature_qur, 2048);

        double start = get_current_time(); 
        float score = calculCosineSimilar_2048((byte*)pfeature1, (byte*)pfeature2);
        double end = get_current_time();
        FreeAlign16(pfeature1);
        FreeAlign16(pfeature2);
        */

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
    int Text[6] = {0};
         while(1)
         {  
            std::cout << "Show "<< (i+1) << std::endl;
            std::cout << feature_vec[i].camera_id << std::endl;
            std::cout << search_image_camera_id << std::endl;
            if(feature_vec[i].camera_id==search_image_camera_id)
            {
               ;
            }
            else
	        {
                std::cout << setprecision(5) << feature_vec[i].feature_score<<"	"<<feature_vec[i].feature_id << "  " <<feature_vec[i].image_path << std::endl;
                out_id<< setprecision(5) << feature_vec[i].feature_score<<"	"<<feature_vec[i].feature_id << " " <<  feature_vec[i].camera_id <<" "<<feature_vec[i].image_path << std::endl; 
                k++;	
            } 
		i = i +1 ;
		if(k >= 6)
		{
		  break;
		}
        }
     

       

  return 0;
}
