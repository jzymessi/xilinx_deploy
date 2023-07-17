#include <fstream>
#include <future>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/image_list.hpp>
#include <vitis/ai/nnpp/yolovx.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/time_measure.hpp>
#include <boost/type_index.hpp>
#include <stdlib.h>
using namespace std;
using namespace cv;
std::mutex g_mtx;

struct BenchMarkResult {
  int image_start;
  std::vector<vitis::ai::YOLOvXResult> res;
};

class YOLOvX: public vitis::ai::ConfigurableDpuTaskBase {
public:
    static std::unique_ptr<YOLOvX> create(const std::string& model_name, bool need_preprocess = true);
    static std::unique_ptr<YOLOvX> create(const std::string& model_name, xir::Attrs* attrs, bool need_preprocess = true);
    ~YOLOvX();

    virtual vector<vitis::ai::YOLOvXResult> run(const vector<vector<char>>& input_images) =0;

protected:
    explicit YOLOvX(const std::string& model_name, bool need_preprocess);
    YOLOvX(const YOLOvX&) = delete;
};

YOLOvX::YOLOvX(const std::string& model_name, bool need_preprocess): vitis::ai::ConfigurableDpuTaskBase(model_name, need_preprocess) {}
YOLOvX::~YOLOvX(){}

class YOLOvXImp : public YOLOvX {
public:
    YOLOvXImp(const std::string& model_name, bool need_preprocess = true);
    virtual ~YOLOvXImp();

private:
    virtual vector<vitis::ai::YOLOvXResult> run(const vector<vector<char>>& input_images) override;
};
std::unique_ptr<YOLOvX> YOLOvX::create(const std::string& model_name, bool need_preprocess) {
    return std::unique_ptr<YOLOvX>(new YOLOvXImp(model_name, need_preprocess));
}

YOLOvXImp::YOLOvXImp(const std::string& model_name, bool need_preprocess) : YOLOvX(model_name, need_preprocess) {}
YOLOvXImp::~YOLOvXImp() {}


std::vector<std::shared_ptr<YOLOvX>> yolox_model_init(std::string model_name,int num_of_threads)
{
    std::shared_ptr<YOLOvX> model = YOLOvX::create(model_name);
  
    std::vector<std::shared_ptr<YOLOvX>> models;
    for (int i = 0; i < num_of_threads; ++i) {
        if (i == 0)
        {      
            models.emplace_back(std::move(model));
        } else {
            models.emplace_back(YOLOvX::create(model_name));
        }
    }
  return models;
}

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

void letterbox(const cv::Mat& im, int w, int h, cv::Mat& om, float& scale) {
  scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);
  cv::Mat img_res;
  if (im.size() != cv::Size(w, h)) {
    cv::resize(im, img_res, cv::Size(im.cols * scale, im.rows * scale), 0, 0,
               cv::INTER_LINEAR);
    auto dw = w - img_res.cols;
    auto dh = h - img_res.rows;
    if (dw > 0 || dh > 0) {
      om = cv::Mat(cv::Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
      copyMakeBorder(img_res, om, 0, dh, 0, dw, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
    } else {
      om = img_res;
    }
  } else {
    om = im;
    scale = 1.0;
  }
}

vector<vitis::ai::YOLOvXResult> YOLOvXImp::run(const vector<vector<char>>& input_images) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  vector<cv::Mat> images(input_images.size());
  vector<float> scale(input_images.size());


  for (auto i = 0u; i < input_images.size(); i++) {
    auto input_image = cv::imdecode(cv::Mat(input_images[i]), 1);
    letterbox(input_image, sWidth, sHeight, images[i], scale[i]);
  }

  configurable_dpu_task_->setInputImageBGR(images);

  configurable_dpu_task_->run(0);

  auto ret = vitis::ai::yolovx_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), scale);

  return ret;
}

std::vector<std::string> Cutline(std::vector<std::string>& Arrs, int begin, int end)
{
  std::vector<std::string> result;
  std::vector<std::string>::const_iterator First = Arrs.begin() + begin;  // 找到第  2 个迭代器 （idx=1）
  std::vector<std::string>::const_iterator Second = Arrs.begin() + end; // 找到第  6 个迭代器 （idx=5）的下一个位置 
  result.assign(First,Second);
  return result;
}


template <typename T>
inline BenchMarkResult threads_run(std::vector<std::string> Lines ,std::vector<vitis::ai::YOLOvXResult> res , std::shared_ptr<T> &&model, int image_start ,int image_end) 
{
    std::unique_lock<std::mutex> lock_t(g_mtx);
    lock_t.unlock();
    auto ret_temp = std::vector<std::vector<char>>();  
    for (int i=0;i < (image_end-image_start);i++)
    {     
      std::vector<char> data;
      std::ifstream file(Lines[i]);
      file >> std::noskipws;        
      std::copy(std::istream_iterator<char>(file), std::istream_iterator<char>(), std::back_inserter(data));
      ret_temp.emplace_back(data); 
      if(ret_temp[i].empty())
      {
        cout << "copy data error" << endl;
      }   
    }
    int count = (image_end-image_start) / 6;
    int remainder = (image_end-image_start) % 6;

    for (int i = 0; i < (count+1); i++) 
    {
      std::vector<std::vector<char>> imgs;
      imgs.reserve(6);
      auto res_tmp = std::vector<vitis::ai::YOLOvXResult>();
      if(i != count)
      {
          for (int n = 0; n < 6; n++) 
        {
          imgs.emplace_back(ret_temp[(i*6 + n)]); 
        }
        if(model == NULL)
        {
          cout << "model is empty" << endl;
        }

        res_tmp = model->run(imgs);
        for (auto n = 0u; n < 6; n++) 
        {
            res.emplace_back(res_tmp[n]);
        }
      }
      else
      {
        for (int n = 0; n < remainder; n++) 
        {
          imgs.emplace_back(ret_temp[(i*6 + n)]); 
        }
        res_tmp = model->run(imgs); 
        for (auto n = 0u; n < remainder; n++) 
        {
            res.emplace_back(res_tmp[n]);
        }
      }    
  }   
  return BenchMarkResult{image_start,res};
}
  

std::vector<vitis::ai::YOLOvXResult> run_yolox_xmodel(std::vector<std::shared_ptr<YOLOvX>>& models, std::vector<std::string> vectorLines, int num_of_threads) 
{
    auto ret = std::vector<std::vector<char>>();
    auto res = std::vector<vitis::ai::YOLOvXResult>();
    auto res_img = std::vector<vitis::ai::YOLOvXResult>();
    auto res_img_new = std::vector<vitis::ai::YOLOvXResult>();
    auto res_data = std::vector<std::vector<char>>();
    auto res_data_new = std::vector<std::vector<char>>();
    int thread_image_num = vectorLines.size() / num_of_threads;
    int thread_image_remainder = vectorLines.size() % num_of_threads;
    //std::unique_lock<std::mutex> lock_main(g_mtx);
    auto load_image_start = std::chrono::system_clock::now();
    std::vector<std::future<BenchMarkResult>> results;
    results.reserve(num_of_threads);
    //cout << "vectorLines size:" << vectorLines.size() << endl;
    //use threads copy image from ssd to ddr
    int start_copy = 0;
    int end_copy = 0;
    std::unique_lock<std::mutex> lock_main(g_mtx);
    for(int i=0; i < num_of_threads ; ++i)
    {
        start_copy = thread_image_num*(i);
        if(i==(num_of_threads-1))
        {
         end_copy = thread_image_num*(i+1)+thread_image_remainder;
        }
        else
        {
          end_copy = thread_image_num*(i+1);
        }
        std::vector<std::string> result_line = Cutline(vectorLines,start_copy,end_copy);
        //cout << "result_line size:" << result_line.size() << endl;
        //cout << "start copy:" << start_copy << " end_copy: " << end_copy << endl;
        results.emplace_back(std::async(std::launch::async, threads_run<YOLOvX>,result_line, res, models[i], start_copy, end_copy));
    }

    lock_main.unlock();
    
    for (auto &r : results) {
      auto result = r.get();
      int index = result.image_start;
      res_img = result.res;
      for(int i=0;i<res_img.size();i++)
      {
        res_img_new.emplace_back(res_img[i]);
      }
    }

    return res_img_new;
} 

int main(int argc, char* argv[])
{

    vector<string> v_class{};
    v_class.push_back("TPDPD");
    v_class.push_back("TPDPS");
    v_class.push_back("TPDPC");
    v_class.push_back("TGUNO");
    v_class.push_back("TPFP0");

    //color
    std::array<float,15> array_color= {
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
    };

    std::string model_path = argv[1];
    string id_image_list = argv[2];
    std::vector<vitis::ai::YOLOvXResult> output;
    std::vector<std::shared_ptr<YOLOvX>> models; 
    int threads_num = atoi(argv[3]);
    std::vector<std::string> ImageLines;
    ImageLines.clear();
    std::vector<struct feature_param> id_features;
    id_features.clear();

    LoadFeatureNames(id_image_list, id_features);
    for (int i = 0; i < id_features.size(); i++)
    {
      ImageLines.emplace_back(id_features[i].image_path);
    }
    
    models = yolox_model_init(model_path,threads_num);
    // std::vector<struct input_image_param> input_params;
    
    output = run_yolox_xmodel(models,ImageLines,threads_num);
    for(int i=0;i<output.size();i++)
    {
        cv::Mat image = cv::imread(ImageLines[i]);
        cv::Mat original_image = image;
        for(auto& result: output[i].bboxes)
        {
            // cv::Mat image = cv::imread(ImageLines[i]);
            // cv::Mat original_image = image;
            int label = result.label;
            cout << "label: " << label << "score: " << result.score << endl;
            auto& box = result.box;
            int color_b = int(array_color[label*3] * 255);
            int color_g = int(array_color[label*3+1] * 255);
            int color_r = int(array_color[label*3+2] * 255);
        //     float scale = min(640.0 / original_image.rows, 640.0 / original_image.cols);
        //     for (int i=0; i<4; i++){
        //        box[i] /= scale;
        //    }
            putText(original_image,v_class[label],Point(box[0],box[1]-2),FONT_HERSHEY_SIMPLEX,0.5,Scalar(color_b,color_g,color_r),1);
            rectangle(original_image, Point(box[0],box[1]),Point(box[2],box[3]),Scalar(color_b,color_g,color_r),2,2,0);
        }
        string str_jpg = ".jpg";
        string save_output_imgpath = "./output/" + to_string(i) + str_jpg;
        cout << save_output_imgpath << endl;
        imwrite(save_output_imgpath,original_image);
        // cout << "original_image.rows:" << original_image.rows << " original_image.cols:" << original_image.cols << endl;
    }

}
