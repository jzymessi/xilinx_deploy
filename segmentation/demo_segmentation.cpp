//source /workspace/setup/vck5000/setup.sh DPUCVDX8H_6pe_dwc
//./demo_segmentation SemanticFPN_cityscapes_pt/SemanticFPN_cityscapes_pt.xmodel test_segmentation_1.txt ./test_images/
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <vitis/ai/segmentation.hpp>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/profiling.hpp>
#include <vector>
#include <vitis/ai/image_list.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/nnpp/segmentation.hpp>
#include <boost/type_index.hpp>
#include <vitis/ai/dpu_task.hpp>
using namespace std;
using namespace cv;

struct SegmentationResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Segmentation result. The cv::Mat type is CV_8UC1 or CV_8UC3.
  cv::Mat segmentation;
};

class Segmentation 
{
 public:
  static std::unique_ptr<Segmentation> create(const std::string& model_name, bool need_preprocess = true);
  explicit Segmentation();
  Segmentation(const Segmentation&) = delete;
  virtual ~Segmentation();
  virtual int getInputWidth() const = 0;
  virtual int getInputHeight() const = 0;
  virtual size_t get_input_batch() const = 0;
  virtual vitis::ai::SegmentationResult new_run_8UC1(const cv::Mat& image) = 0;
};

Segmentation::Segmentation() {}
Segmentation::~Segmentation() {}

class SegmentationImp : public vitis::ai::TConfigurableDpuTask<Segmentation> 
{
 public:
  SegmentationImp(const std::string& model_name, bool need_preprocess = true);
  virtual ~SegmentationImp();

 private:
  virtual vitis::ai::SegmentationResult new_run_8UC1(const cv::Mat& image) override;
};

SegmentationImp::SegmentationImp(const std::string& model_name,bool need_preprocess)
  : vitis::ai::TConfigurableDpuTask<Segmentation>(model_name,need_preprocess) {}
SegmentationImp::~SegmentationImp() {}


std::unique_ptr<Segmentation> Segmentation::create(
    const std::string& model_name, bool need_preprocess) {
  return std::unique_ptr<Segmentation>(
      new SegmentationImp(model_name, need_preprocess));
}


static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
  std::vector<std::string> elems;
  size_t pos = 0;
  size_t len = s.length();
  size_t delim_len = delim.length();
  if (delim_len == 0) return elems;
  while (pos < len) {
    int find_pos = s.find(delim, pos);
    if (find_pos < 0) {
      elems.push_back(s.substr(pos, len - pos));
      break;
    }
    elems.push_back(s.substr(pos, find_pos - pos));
    pos = find_pos + delim_len;
  }
  return elems;
}

void LoadImageNames(std::string const &filename,
                    std::vector<std::string> &images) {
  images.clear();

  /*Check if path is a valid directory path. */
  FILE *fp = fopen(filename.c_str(), "r");
  if (NULL == fp) {
    fprintf(stdout, "open file: %s  error\n", filename.c_str());
    exit(1);
  }

  char buffer[256] = {0};
  while (fgets(buffer, 256, fp) != NULL) {
    int n = strlen(buffer);
    buffer[n - 1] = '\0';
    std::string name = buffer;
    images.push_back(name);
  }
  
  fclose(fp);
}


void max_index_c(int8_t *d, int c, int g, uint8_t *results) {
  for (int i = 0; i < g; ++i) {
    auto it = std::max_element(d, d + c);
    results[i] = it - d;
    // cout << (int)results[i] << endl;
    d += c;
    // std::cout << "it results type is " << boost::typeindex::type_id_with_cvr<decltype(it)>().pretty_name() << std::endl;
    // std::cout << "d results type is " << boost::typeindex::type_id_with_cvr<decltype(d)>().pretty_name() << std::endl;
    // cout << "*it = " << *it << endl;
    // cout << "*d = " << *d << endl;
    // cout << "c = " << c << endl;
  }
  // cout << "111" << endl;
  // cout << (int)(it-d) << endl;
}

std::vector<uint8_t> max_index(int8_t *feature_map, int width, int height,
                               int channel) {
  const auto g = width * height;
  std::vector<uint8_t> ret(g);
  max_index_c(feature_map, channel, g, &ret[0]);
  return ret;
}

vitis::ai::SegmentationResult segmentation_post_process_8UC1(
    const vitis::ai::library::InputTensor& input_tensors,
    const vitis::ai::library::OutputTensor& output_layer,
    size_t batch_idx) {

  std::vector<uint8_t> output =
           max_index((int8_t*)output_layer.get_data(batch_idx),
                                output_layer.width, 
                                output_layer.height,
                                output_layer.channel);
  cout << "output_layer.width:" <<  output_layer.width << endl;
  cout << "output_layer.height:" <<  output_layer.height << endl;
  cout << "output_layer.channel:" <<  output_layer.channel << endl;
  cout << "output_size():" << output.size() << endl;
  // cout << output.data() << endl;
  cv::Mat segMat = cv::Mat(output_layer.height, output_layer.width, CV_8UC1,
                 output.data()).clone();
  return vitis::ai::SegmentationResult{(int)input_tensors.width,
                            (int)input_tensors.height, segMat};
}

std::vector<vitis::ai::SegmentationResult> segmentation_post_process_8UC1(
    const vitis::ai::library::InputTensor& input_tensors,
    const vitis::ai::library::OutputTensor& output_tensors) {
  auto batch_size = input_tensors.batch;
  cout << "batch_size: " << batch_size << endl;
  auto ret = std::vector<vitis::ai::SegmentationResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.emplace_back(
        segmentation_post_process_8UC1(input_tensors, output_tensors, i));
  }
  return ret;
}

vitis::ai::SegmentationResult SegmentationImp::new_run_8UC1(const cv::Mat& input_image) 
{
  cout << "********" << endl;
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
  } else {
    image = input_image;
  }
  
  if(configurable_dpu_task_->getConfig().order_type() == 1) {
    configurable_dpu_task_->setInputImageBGR(image);
  } else if (configurable_dpu_task_->getConfig().order_type() == 2) {
    configurable_dpu_task_->setInputImageRGB(image);
  } else {
    LOG(FATAL) << "unknown image order type";
  }
 
  configurable_dpu_task_->run(0);

  auto result = segmentation_post_process_8UC1(
      configurable_dpu_task_->getInputTensor()[0][0],
      configurable_dpu_task_->getOutputTensor()[0][0]);
  cout << configurable_dpu_task_->getInputTensor()[0][0] << endl;
  cout << configurable_dpu_task_->getOutputTensor()[0][0] << endl;

  return result[0];
}


int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << "\n model name: " << argv[1] << "\n image_list_file " << argv[2] << "\n result_path: " << argv[3] << std::endl;
    return -1;
  }
  auto det = Segmentation::create(argv[1]);  // Init

  string g_output_dir_1 = argv[3];
  vector<string> names;
  LoadImageNames(argv[2], names);
  for (auto name : names) {
    cv::Mat img_resize;
    cv::Mat image = cv::imread(name);

    //cnannel 1
    auto result_channel_1 = det->new_run_8UC1(image);
    cout << "result.rows:" << result_channel_1.segmentation.rows << endl;
    cout << "result.cols:" << result_channel_1.segmentation.cols << endl;
    for (auto y = 0; y < result_channel_1.segmentation.rows; y++) 
    {
      for (auto x = 0; x < result_channel_1.segmentation.cols; x++) 
      {
        result_channel_1.segmentation.at<uchar>(y, x) *= 10;
      }
   }

    auto namesp = split(name, "/");
    // cv::Mat img;
    // cv::resize(result.segmentation, img, cv::Size(2048, 1024), 0, 0,cv::INTER_NEAREST);
    cv::imwrite(g_output_dir_1 + "/" + namesp[namesp.size() - 1], result_channel_1.segmentation);

  }

  return 0;
}


