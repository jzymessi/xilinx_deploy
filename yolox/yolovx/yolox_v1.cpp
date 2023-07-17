//./zheta_threads_demo zheta_0.95_new_resnet50/resnet50.xmodel val_3cls_accuracy.txt 20
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
#include "./yolox_v1.hpp"
#include <stdlib.h>
using namespace std;
std::mutex g_mtx;

std::vector<std::shared_ptr<YOLOvX>> model_init(std::string model_name,int num_of_threads)
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
    //cout << "images size :" << ret_temp.size() << endl;
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
  

std::vector<vitis::ai::YOLOvXResult> run_xmodel(std::vector<std::shared_ptr<YOLOvX>>& models, std::vector<std::string> vectorLines, int num_of_threads) 
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
