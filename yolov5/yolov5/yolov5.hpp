#ifndef YOLOV5_HPP
#define YOLOV5_HPP
#include <fstream>
#include <future>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/image_list.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <vitis/ai/env_config.hpp>
using namespace std;

struct BenchMarkResult {
  int image_start;
  std::vector<vitis::ai::YOLOv3Result> res;
};

class YOLOv3 : public vitis::ai::ConfigurableDpuTaskBase {
 public:
  static std::unique_ptr<YOLOv3> create(const std::string& model_name, bool need_preprocess = true);
  virtual ~YOLOv3();
  virtual std::vector<vitis::ai::YOLOv3Result> run(const vector<vector<char>>& input_images) = 0;
 protected:
  explicit YOLOv3(const std::string& model_name, bool need_preprocess);
  YOLOv3(const YOLOv3&) = delete;  
};

YOLOv3::YOLOv3(const std::string& model_name, bool need_preprocess)
    : vitis::ai::ConfigurableDpuTaskBase(model_name, need_preprocess) {}
YOLOv3::~YOLOv3() {}



class YOLOv3Imp : public YOLOv3 {
 public:
  YOLOv3Imp(const std::string& model_name, bool need_preprocess = true);
  virtual ~YOLOv3Imp();

 private:
  virtual std::vector<vitis::ai::YOLOv3Result> run(const vector<vector<char>>& input_images) override;
  bool tf_flag_;
};

std::unique_ptr<YOLOv3> YOLOv3::create(const std::string& model_name,bool need_preprocess) {
  return std::unique_ptr<YOLOv3>(new YOLOv3Imp(model_name, need_preprocess));
}

YOLOv3Imp::YOLOv3Imp(const std::string& model_name, bool need_preprocess)
    : YOLOv3(model_name, need_preprocess),
      tf_flag_(configurable_dpu_task_->getConfig().is_tf()) {}

YOLOv3Imp::~YOLOv3Imp() {}

//extren func
extern std::vector<vitis::ai::YOLOv3Result> run_xmodel(std::vector<std::shared_ptr<YOLOv3>>& models, std::vector<std::string> vectorLines, int num_of_threads);
extern std::vector<std::shared_ptr<YOLOv3>> model_init(std::string model_name,int num_of_threads);
#endif



