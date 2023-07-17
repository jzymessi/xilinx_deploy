#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <cstring>
struct feature_param{
	
    std::string image_path;
    int feature_id;
    float feature_value[2048];
  
    feature_param()
    {
        feature_id = 0;
        image_path.clear();
	    memset(feature_value,0,sizeof(feature_value));
    }
};

struct input_image_param{
	
    std::string image_path;
    std::string image_id;
    input_image_param()
    {
        image_id = "0";
        image_path.clear();
    }
};

extern void LoadFeatureNames(std::string const& filename, std::vector<struct feature_param>& id_features);
extern void LoadImageNames(std::string const& filename, std::vector<struct input_image_param>& images_param);

#endif