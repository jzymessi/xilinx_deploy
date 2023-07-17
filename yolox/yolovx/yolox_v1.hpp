#ifndef YOLOVX_V1_HPP
#define YOLOVX_V1_HPP
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
using namespace std;

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

//extren func
extern std::vector<vitis::ai::YOLOvXResult> run_xmodel(std::vector<std::shared_ptr<YOLOvX>>& models, std::vector<std::string> vectorLines, int num_of_threads);
extern std::vector<std::shared_ptr<YOLOvX>> model_init(std::string model_name,int num_of_threads);
#endif



