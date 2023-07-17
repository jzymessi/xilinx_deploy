#ifndef CLASSIFICATION_THREADS_CUT_HPP
#define CLASSIFICATION_THREADS_CUT_HPP

#include <fstream>
#include <future>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/image_list.hpp>
#include <vitis/ai/nnpp/classification.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <vitis/ai/env_config.hpp>
using namespace std;

struct BenchMarkResult {
  int image_start;
  std::vector<vitis::ai::ClassificationResult> res;
};

class Classification : public vitis::ai::ConfigurableDpuTaskBase {
 public:
  static std::unique_ptr<Classification> create(const std::string& model_name,bool need_preprocess = true);
  explicit Classification(const std::string& model_name, bool need_preprocess);
  Classification(const Classification&) = delete;
  virtual ~Classification();
  virtual std::vector<vitis::ai::ClassificationResult> run(const vector<vector<char>>& input_chars) = 0;
};

Classification::Classification(const std::string& model_name,bool need_preprocess): vitis::ai::ConfigurableDpuTaskBase(model_name, need_preprocess) {}
Classification::~Classification() {}

class ClassificationImp : public Classification {
public:
  ClassificationImp(const std::string& model_name, bool need_preprocess = true);
  virtual ~ClassificationImp();

private:
  virtual std::vector<vitis::ai::ClassificationResult> run(const vector<vector<char>>& input_chars) override;
private:
  int preprocess_type;
  const int TOP_K;
  const bool test_accuracy;

};

std::unique_ptr<Classification> Classification::create(const std::string& model_name, bool need_preprocess) 
{
  return std::unique_ptr<Classification>(new ClassificationImp(model_name, need_preprocess));
}

ClassificationImp::ClassificationImp(const std::string& model_name,bool need_preprocess)
    : Classification(model_name, need_preprocess),preprocess_type{configurable_dpu_task_->getConfig().classification_param().preprocess_type()},
      TOP_K{configurable_dpu_task_->getConfig().classification_param().top_k()},
      test_accuracy{configurable_dpu_task_->getConfig().classification_param().test_accuracy()} {}

ClassificationImp::~ClassificationImp() {}

//func
extern std::vector<std::shared_ptr<Classification>> model_init(std::string model_name,int num_of_threads);
extern std::vector<vitis::ai::ClassificationResult> run_xmodel(std::vector<std::shared_ptr<Classification>>& models, std::vector<std::string> vectorLines, int num_of_threads);

#endif



