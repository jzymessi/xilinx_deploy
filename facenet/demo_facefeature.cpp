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
#include <vitis/ai/facefeature.hpp>
#include <sys/time.h>
#include <algorithm>
//#include <intrin.h>
#include <emmintrin.h>
#include <math.h>
#include <stdlib.h>
#include <memory>
 
using namespace std;

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


void FreeAlign16(void* ptr)
{
    int ptrSize = sizeof(void*);
    free( (void *) *((size_t *) (( (byte *) ptr ) - ptrSize)) );

}

/*
void* aligned_malloc(size_t size, size_t alignment)
{
    size_t offset = alignment - 1 + sizeof(void*);
    void * originalP = malloc(size + offset);
    size_t originalLocation = reinterpret_cast<size_t>(originalP);
    size_t realLocation = (originalLocation + offset) & ~(alignment - 1);
    void * realP = reinterpret_cast<void*>(realLocation);
    size_t originalPStorage = realLocation - sizeof(void*);
    *reinterpret_cast<void**>(originalPStorage) = originalP;
    return realP;
}
 
void aligned_free(void* p)
{
    size_t originalPStorage = reinterpret_cast<size_t>(p) - sizeof(void*);
    free(*reinterpret_cast<void**>(originalPStorage));
}
*/
struct input_search_param{
	
    std::string image_path;
    int feature_id;
    float feature_value[512];
    //void* pfeature = MallocAlign16(2048);
    //void* paf = aligned_malloc(2048,16);
    
    input_search_param()
    {
        feature_id = 0;
        image_path.clear();
        memset(feature_value,0,sizeof(feature_value));
    }
};

struct targert_feature_param{
	
    std::string image_path;
    int feature_id;
    float feature_value[512];
    //void* pfeature = MallocAlign16(2048);
    
    targert_feature_param()
    {
        feature_id = 0;
        image_path.clear();
	    memset(feature_value,0,sizeof(feature_value));
    }
};

struct feature_result{
	
    int feature_id;
    float feature_score;
  
    feature_result()
    {
       feature_score = 0;
       feature_id = 0;
    }
};



bool LessSort (struct feature_result a, struct feature_result b) 
{ 
    return (a.feature_score < b.feature_score); 
}

bool GreaterSort (struct feature_result a, struct feature_result b) 
{ 
    return (a.feature_score > b.feature_score); 
}

bool Judge_Hit_Or_Not(vector<struct feature_result> feature_vec, int expected_index, int top_k)
{
	bool hit_or_not = false;
	if(feature_vec.size() < top_k)
	{
		std::cout << "The top_k is out of range!" << std::endl;
		return hit_or_not;
	}

    for(int i =0; i < top_k; i++)
    {
	   if(expected_index == feature_vec[i].feature_id)
	   {
		   hit_or_not = true;
		   break;
	   }	   
   }
   return hit_or_not;
}

void LoadImageNames(std::string const& filename,
                    std::vector<struct input_search_param>& images_param) {
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
	struct input_search_param param;
	ss >> param.image_path;
    ss >> param.feature_id; 	
	
	images_param.push_back(param);
  }
  
  input.close();	
}

void LoadFeatureNames(std::string const& filename, std::vector<struct targert_feature_param>& id_features) 
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
		struct targert_feature_param param;
  	
       stringstream ss(line);  
       ss >> param.image_path;
       ss >> param.feature_id; 

       for(int i =0; i!=512; i++){
           ss >> param.feature_value[i];
       }  
       //memcpy( param.pfeature, &param.feature_value[0], 2048);
       
  	   id_features.push_back(param);
	}	
    input.close();	
}

float simd_dot(const float* x, const float* y, const long& len)
{
    float inner_prob =0.0f;
    __m128 X,Y;
    __m128 acc =_mm_setzero_ps();
    float temp[4];
    
    long i;
    for(i = 0; i+4 <len; i += 4) {
        X = _mm_loadu_ps (x + i);
        Y = _mm_loadu_ps (y + i);
        acc = _mm_add_ps (acc, _mm_mul_ps(X,Y));
    }
    _mm_storeu_ps(&temp[0], acc);
    inner_prob = temp[0] + temp[1] + temp[2] +temp[3];
    
    for(; i < len; ++i) {
        inner_prob +=x[i] * y[i];
    }
    return inner_prob;
}


float calculCosineSimilar_acc(float *fc1, float *fc2, int dim) {
 
    return simd_dot(fc1, fc2, dim) 
           / sqrt(simd_dot(fc1, fc1, dim)) 
           * sqrt(simd_dot(fc2, fc2, dim));
}


float calculCosineSimilar_acc_opt(float *fc1, float *fc2, int dim) {
	float partial[4];
    float temp[12];
    
	__m128 X,Y;
    __m128 acc =_mm_setzero_ps();	
	__m128 acc1 =_mm_setzero_ps();	
	__m128 acc2 =_mm_setzero_ps();	
	  
	for(int i =0; i+4 < dim; i += 4) {
		__builtin_prefetch(fc1+i+2048, 0, 3); 
		__builtin_prefetch(fc2+i+2048, 0, 3); 
		
		X = _mm_loadu_ps (fc1 + i); 
        Y = _mm_loadu_ps (fc2 + i);
        
        acc = _mm_add_ps (acc, _mm_mul_ps(X,Y));
		acc1 = _mm_add_ps (acc1, _mm_mul_ps(X,X));
		acc2 = _mm_add_ps (acc2, _mm_mul_ps(Y,Y));
	}
    
	_mm_storeu_ps(&temp[0], acc);
    _mm_storeu_ps(&temp[4], acc1);
    _mm_storeu_ps(&temp[8], acc2);
	
	partial[0] = temp[0] + temp[1] + temp[2] +temp[3];
	partial[1] = temp[4] + temp[5] + temp[6] +temp[7];
	partial[2] = temp[8] + temp[9] + temp[10] +temp[11];
	 
    return partial[0] / (sqrt(partial[1]) * sqrt(partial[2]) );
}
				
float calculCosineSimilar(float *feature1, float *feature2, int feature_dim) {
   	float partial[4];
    float temp[12];
    for (std::vector<double>::size_type i = 0; i != feature_dim; i++) {
        //__builtin_prefetch(feature1+i+2048, 0, 3); 
		//__builtin_prefetch(feature2+i+2048, 0, 3); 
        partial[0] += feature1[i] * feature2[i];
        partial[1] += feature1[i] * feature1[i];
        partial[2] += feature2[i] * feature2[i];
    }
    return partial[0] / (sqrt(partial[1]) * sqrt(partial[2]) );
}

float bytesTofloat(byte bytes[])
{

    return *((float*)bytes);
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
    
    string id_image_list_file = argv[2];
    string id_feature_list_file = argv[3];
  
    bool preprocess = !(getenv("PRE") != nullptr);
    auto facefeature = vitis::ai::FaceFeature::create(argv[1], preprocess);
    int modelwidth = facefeature->getInputWidth();
    int modelheight = facefeature->getInputHeight();
    std::cout << "modelwidth = " <<  modelwidth << std::endl;
    std::cout << "modelheight = " << modelheight << std::endl;
    std::vector<struct input_search_param> input_params;
    LoadImageNames(id_image_list_file, input_params);
     
    std::vector<struct targert_feature_param> id_features;
    LoadFeatureNames(id_feature_list_file, id_features);
	
	//count precision 
	int one_hit_count =0;
    int three_hit_count = 0;
    int five_hit_count = 0;
    int ten_hit_count = 0;
    int thirty_hit_count = 0;
    int fifty_hit_count = 0;

    double calcul_time_total = 0; 
    double total_feature_time  =0.0;   
    for (size_t id =0; id < input_params.size(); id++) 
    {
        cv::Mat image = cv::imread(input_params[id].image_path);
        if (image.empty()) {
            std::cout << "cannot load " << input_params[id].image_path << std::endl;
            break;
        }
      
        if(image.rows != modelwidth || image.cols != modelheight)
      	{
      	    cv::resize(image, image, cv::Size(modelwidth, modelheight), 0, 0, cv::INTER_LINEAR);
      	}
        
        //std::cout << "imagewidth = " << image.cols << std::endl;
        //std::cout << "imagehight = " << image.rows << std::endl;
         
      	double tstart = get_current_time(); 
        auto result = facefeature->run(image);
      	auto features = *result.feature;
        double tend = get_current_time();
      	  
      	double costtime = tend - tstart;
      	//std::cout << "extract feature time :"<< costtime << " ms"<<std::endl;
      	total_feature_time = total_feature_time + costtime;
  
      	//std::cout << "float features :";  //
        int i =0;
      	for (float f : features) {
  
      		//std::cout <<i<<" "<< f << " "; 
      		//std::cout << f << ","; 
      		
      		input_params[id].feature_value[i] = f;     		
      		i++;
      	}       
      	//memcpy(input_params[id].pfeature, &input_params[id].feature_value[0], 2048);
      	
    }
    
    for(size_t id =0; id < input_params.size();id++)
    {
		vector<struct feature_result> feature_vec;
        feature_vec.clear();

        for(size_t i =0; i< id_features.size(); i++)
        {
           /*  
        	 double start = get_current_time(); 
           float score = calculCosineSimilar(input_params[id].feature_value, id_features[i].feature_value, 512);
            double end = get_current_time();
        	 */
           
            void* pfeature1 = MallocAlign16(2048);
        	  void* pfeature2 = MallocAlign16(2048);
        	  memcpy(pfeature1, &input_params[id].feature_value[0], 2048);
        	  memcpy(pfeature2, &id_features[i].feature_value[0], 2048);
        	  double start = get_current_time(); 
        	  float score = calculCosineSimilar_2048((byte*)pfeature1, (byte*)pfeature2);
        	  double end = get_current_time();
        	  FreeAlign16(pfeature1);
        	  FreeAlign16(pfeature2);
        	  
        	  //float score = calculCosineSimilar(input_params[id].pfeature, id_features[i].pfeature, 512);
        	   //float score = calculEuclideanDistance(id_features[i].feature_value, tmpfeature, 512);
           
             double cur = end - start;
             calcul_time_total += cur;
         
             struct feature_result param;
             param.feature_id = id_features[i].feature_id;
             param.feature_score = score;

             feature_vec.push_back(param);           
        }
         
        std::sort (feature_vec.begin(), feature_vec.end(), GreaterSort); 
   
        bool one_hit = Judge_Hit_Or_Not(feature_vec, input_params[id].feature_id, 1);
    	bool three_hit = Judge_Hit_Or_Not(feature_vec, input_params[id].feature_id, 3);
    	bool five_hit = Judge_Hit_Or_Not(feature_vec, input_params[id].feature_id, 5);
    	bool ten_hit = Judge_Hit_Or_Not(feature_vec, input_params[id].feature_id, 10);
    	bool thirty_hit = Judge_Hit_Or_Not(feature_vec, input_params[id].feature_id, 30);
  	    bool fifty_hit = Judge_Hit_Or_Not(feature_vec, input_params[id].feature_id, 50);
    	if(one_hit) one_hit_count++;
    	if(three_hit) three_hit_count++;
    	if(five_hit) five_hit_count++;
    	if(ten_hit) ten_hit_count++;
    	if(thirty_hit) thirty_hit_count++;
    	if(fifty_hit) fifty_hit_count++;		
   }//end of for
   
    std::cout << "avg_feature_time: "<< (total_feature_time/(int)input_params.size())<< " ms"<<std::endl;
    std::cout <<"total number: "<< input_params.size() <<" total time: "<< calcul_time_total<< " ms"<< std::endl;
	  std::cout <<"hit@1 accuracy: "<< ((float)one_hit_count/(float)input_params.size())*100 <<"%" << std::endl;
	  std::cout <<"hit@3 accuracy: "<< ((float)three_hit_count/(float)input_params.size())*100 <<"%" << std::endl;
	  std::cout <<"hit@5 accuracy: "<< ((float)five_hit_count/(float)input_params.size())*100 <<"%" << std::endl;
	  std::cout <<"hit@10 accuracy: "<< ((float)ten_hit_count/(float)input_params.size())*100 <<"%" << std::endl;
    std::cout <<"hit@30 accuracy: "<< ((float)thirty_hit_count/(float)input_params.size() )*100 <<"%" << std::endl;
    std::cout <<"hit@50 accuracy: "<< ((float)fifty_hit_count/(float)input_params.size() )*100 <<"%" << std::endl;
  return 0;
}
