#include <fstream>
#include <future>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/image_list.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/time_measure.hpp>
#include <boost/type_index.hpp>
#include "./classification_v1.hpp"
#include <stdlib.h>
using namespace std;
std::mutex g_mtx;


std::vector<std::shared_ptr<Classification>> model_init(std::string model_name,int num_of_threads)
{
  std::shared_ptr<Classification> model = Classification::create(model_name);
  
  std::vector<std::shared_ptr<Classification>> models;
  for (int i = 0; i < num_of_threads; ++i) {
      if (i == 0) {
          models.emplace_back(std::move(model));
      } else {
          models.emplace_back(Classification::create(model_name));
      }
  }
  return models;
}

void globalAvePool(int8_t *src, int channel, int width, int height, int8_t *dst,int num) 
{
  float sum;
  for (int i = 0; i < channel; i++) {
    sum = 0.0f;
    for (int j = 0; j < width * height; j++) {
      sum += src[i + channel * j];
    }
    int temp = round(((sum / (width * height)) * num));
    dst[i] = (int8_t)std::min(temp, 127);
  }
}

static void croppedImage(const cv::Mat& image, int height, int width,cv::Mat& cropped_img) 
{
  int offset_h = abs((image.rows - height)) / 2;
  int offset_w = abs((image.cols - width)) / 2;  
  cv::Rect box(offset_w, offset_h, width, height);
  cropped_img = image(box).clone();
}


static void vgg_preprocess(const cv::Mat& image, int height, int width,cv::Mat& pro_res) 
{
  float smallest_side = 256;
  float scale =smallest_side / ((image.rows > image.cols) ? image.cols : image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image,cv::Size(image.cols * scale, image.rows * scale));
  croppedImage(resized_image, height, width, pro_res);
}

static void vgg_preprocess_512(const cv::Mat& image, int height, int width,cv::Mat& pro_res) 
{
  float smallest_side = 530;
  float scale =smallest_side / ((image.rows > image.cols) ? image.cols : image.rows);
  cv::Mat resized_image;
  cv::resize(image, resized_image,cv::Size(image.cols * scale, image.rows * scale));
  croppedImage(resized_image, height, width, pro_res);
}

std::vector<vitis::ai::ClassificationResult> ClassificationImp::run(const vector<vector<char>>& input_images) 
{
  std::vector<cv::Mat> images;
  int width = getInputWidth();
  int height = getInputHeight();
  auto size = cv::Size(width, height);
  for (auto i = 0u; i < input_images.size(); i++) 
  {
    if(input_images[i].empty())
    {
      cout << "image" << i <<" input_images is empty" << endl;
    }
    auto input_image = cv::imdecode(cv::Mat(input_images[i]), 1);
    if (size == input_image.size() && (preprocess_type != 8)) 
    {
      images.emplace_back(input_image);
    } 
    else 
    {
      cv::Mat image;
      switch (preprocess_type) 
      {
        case 0:
        {
          // auto resize_start = std::chrono::system_clock::now();
          cv::resize(input_image, image, size); 
          // auto resize_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - resize_start).count();
          // std::cout << "microseconds: " << resize_time << std::endl;
        }
          
          break;
        case 1:
          if (test_accuracy) 
          {
            croppedImage(input_image, height, width, image);
          } 
          else 
          {
            cv::resize(input_image, image, size);
          }
          break;
        case 2:
          vgg_preprocess(input_image, height, width, image);
          break;
        case 3:
          vgg_preprocess_512(input_image, height, width, image);
          break;
        default:
          break;
      }
      images.emplace_back(image);
    }
  }

  
  if (preprocess_type == 0 || preprocess_type == 2 || preprocess_type == 3 || preprocess_type == 4 ||
      preprocess_type == 6 || preprocess_type == 7 || preprocess_type == 8 ||
      preprocess_type == 9 || preprocess_type == 10) {
        // cout << "111" << endl;
    configurable_dpu_task_->setInputImageRGB(images);
  } else {
    // cout << "2222" << endl;
    configurable_dpu_task_->setInputImageBGR(images);
  }
  

  auto postprocess_index = 0;
  if (configurable_dpu_task_->getConfig()
          .classification_param()
          .has_avg_pool_param()) {
    
    configurable_dpu_task_->run(0);
    

    auto avg_scale = configurable_dpu_task_->getConfig()
                         .classification_param()
                         .avg_pool_param()
                         .scale();
    auto batch_size = configurable_dpu_task_->getInputTensor()[0][0].batch;

    for (auto batch_idx = 0u; batch_idx < batch_size; batch_idx++) {
      globalAvePool(
          (int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(
              batch_idx),
          // 1024, 9, 3,
          configurable_dpu_task_->getOutputTensor()[0][0].channel,
          configurable_dpu_task_->getOutputTensor()[0][0].width,
          configurable_dpu_task_->getOutputTensor()[0][0].height,
          (int8_t*)configurable_dpu_task_->getInputTensor()[1][0].get_data(
              batch_idx),
          avg_scale);
    }

    configurable_dpu_task_->run(1); 
    postprocess_index = 1;
  } else {  
    configurable_dpu_task_->run(0);
  }
  
  auto rets = vitis::ai::classification_post_process(
      configurable_dpu_task_->getInputTensor()[postprocess_index],
      configurable_dpu_task_->getOutputTensor()[postprocess_index],
      configurable_dpu_task_->getConfig());

  for (auto& ret : rets) {
    if (configurable_dpu_task_->getOutputTensor()[postprocess_index][0]
            .channel == 1001) {
      for (auto& s : ret.scores) {
        s.index--;
      }
    }
    ret.type = 0;
  }
  return rets;
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
inline BenchMarkResult threads_run(std::vector<std::string> Lines ,std::vector<vitis::ai::ClassificationResult> res , std::shared_ptr<T> &&model, int image_start ,int image_end) 
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
    //cout << "images size :" << ret_temp.size() << endl;
    for (int i = 0; i < (count+1); i++) 
    {
      std::vector<std::vector<char>> imgs;
      imgs.reserve(6);
      auto res_tmp = std::vector<vitis::ai::ClassificationResult>();
      if(i != count)
      {
          for (int n = 0; n < 6; n++) 
        {
          imgs.emplace_back(ret_temp[(i*6 + n)]); 
        }
        //cout << "11111" << endl;
        if(model == NULL)
        {
          cout << "model is empty" << endl;
        }

        res_tmp = model->run(imgs);
        //cout << "2222" << endl;
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
  

std::vector<vitis::ai::ClassificationResult> run_xmodel(std::vector<std::shared_ptr<Classification>>& models, std::vector<std::string> vectorLines, int num_of_threads) 
{
    auto ret = std::vector<std::vector<char>>();
    auto res = std::vector<vitis::ai::ClassificationResult>();
    auto res_img = std::vector<vitis::ai::ClassificationResult>();
    auto res_img_new = std::vector<vitis::ai::ClassificationResult>();
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
        results.emplace_back(std::async(std::launch::async, threads_run<Classification>,result_line, res, models[i], start_copy, end_copy));
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

