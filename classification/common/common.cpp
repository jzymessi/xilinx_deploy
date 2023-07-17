#include <iostream>
#include <vector>
#include <fstream>
#include "common.hpp"
#include <sstream>
using namespace std;

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
	   struct feature_param param;
  	
       stringstream ss(line);  
       ss >> param.image_path;
       ss >> param.feature_id; 

  	   id_features.emplace_back(param);
	}	
    input.close();	
}


void LoadImageNames(std::string const& filename, std::vector<struct input_image_param>& images_param ) {
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
    ss >> param.image_id; 
	images_param.push_back(param);
  }
  
  input.close();		
}
