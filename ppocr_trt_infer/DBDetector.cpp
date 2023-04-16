#include "DBDetector.h"

#include <fstream>
#include <glog/logging.h>
#include <sys/stat.h>

#include "logging.h"
#include "opencv2/imgproc.hpp"

#define INPUT_NAME "x"
#define OUTPUT_NAME "sigmoid_0.tmp_0"

using namespace nvinfer1;

static bool if_file_exists(const char* file_name)
{
	struct stat my_stat{};
	return (stat(file_name, &my_stat) == 0);
}

db_detector::db_detector()
{
}

void db_detector::load_onnx(const std::string& model_name)
{
	Logger logger;
	//根据tensorrt pipeline 构建网络
	IBuilder* builder = createInferBuilder(logger);
	builder->setMaxBatchSize(1);
	constexpr auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	INetworkDefinition* network = builder->createNetworkV2(explicit_batch);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	parser->parseFromFile(model_name.c_str(), static_cast<int>(ILogger::Severity::kWARNING));

	IBuilderConfig* config = builder->createBuilderConfig();
	config->setMaxWorkspaceSize(1ULL << 30);
	cuda_engine_ = builder->buildEngineWithConfig(*network, *config);

	std::string trt_name = model_name;
	const size_t sep_pos = trt_name.find_last_of(".");
	trt_name = trt_name.substr(0, sep_pos) + ".trt";
	const IHostMemory* gie_model_stream = cuda_engine_->serialize();
	std::string serialize_str;
	std::ofstream serialize_output_stream;
	serialize_str.resize(gie_model_stream->size());
	memcpy((void*)serialize_str.data(), gie_model_stream->data(), gie_model_stream->size());
	serialize_output_stream.open(trt_name.c_str());
	serialize_output_stream << serialize_str;
	serialize_output_stream.close();
	cuda_context_ = cuda_engine_->createExecutionContext();
	parser->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();
}

void db_detector::load_trt(const std::string& model_name)
{
	Logger logger;
	IRuntime* runtime = createInferRuntime(logger);
	std::ifstream fin(model_name);
	std::string cached_engine;
	while (fin.peek() != EOF)
	{
		std::stringstream buffer;
		buffer << fin.rdbuf();
		cached_engine.append(buffer.str());
	}
	fin.close();
	cuda_engine_ = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	cuda_context_ = cuda_engine_->createExecutionContext();
	runtime->destroy();
}

bool db_detector::init(const std::string& model_name, const float threshold)
{
	m_threshold_ = threshold;
	std::string trt_name = model_name;
	const size_t sep_pos = trt_name.find_last_of(".");
	trt_name = trt_name.substr(0, sep_pos) + ".trt";
	if (if_file_exists(trt_name.c_str()))
	{
		load_trt(trt_name);
	}
	else
	{
		load_onnx(model_name);
	}

	// 分配输入输出的空间,DEVICE侧和HOST侧
	i_input_index_ = cuda_engine_->getBindingIndex(INPUT_NAME);
	i_output_index_ = cuda_engine_->getBindingIndex(OUTPUT_NAME);

	Dims dims_i = cuda_engine_->getBindingDimensions(i_input_index_);
	LOG(INFO) << "input dims " << dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2] << " " << dims_i.d[3];
	int size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];

	input_size_ = cv::Size(dims_i.d[3], dims_i.d[2]);

	cudaMalloc(&array_dev_memory_[i_input_index_], size * sizeof(float));
	array_host_memory_[i_input_index_] = malloc(size * sizeof(float));
	//方便NHWC到NCHW的预处理
	input_wrappers_.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1,
	                             (unsigned char*)array_host_memory_[i_input_index_] + 0 * sizeof(float) * dims_i.d[2] *
	                             dims_i.d[3]);
	input_wrappers_.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1,
	                             (unsigned char*)array_host_memory_[i_input_index_] + 1 * sizeof(float) * dims_i.d[2] *
	                             dims_i.d[3]);
	input_wrappers_.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1,
	                             (unsigned char*)array_host_memory_[i_input_index_] + 2 * sizeof(float) * dims_i.d[2] *
	                             dims_i.d[3]);
	array_size_[i_input_index_] = size * sizeof(float);
	dims_i = cuda_engine_->getBindingDimensions(i_output_index_);
	LOG(INFO) << "output dims " << dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2] << " " << dims_i.d[3];
	size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];
	output_dim2_ = dims_i.d[2];
	output_dim3_ = dims_i.d[3];
	cudaMalloc(&array_dev_memory_[i_output_index_], size * sizeof(float));
	array_host_memory_[i_output_index_] = malloc(size * sizeof(float));
	array_size_[i_output_index_] = size * sizeof(float);
	cudaStreamCreate(&cuda_stream_);
	uninit_ = false;

	return true;
}

bool db_detector::un_init()
{
	if (uninit_ == true)
	{
		return false;
	}
	for (auto& p : array_dev_memory_)
	{
		cudaFree(p);
		p = nullptr;
	}
	for (auto& p : array_host_memory_)
	{
		free(p);
		p = nullptr;
	}
	cudaStreamDestroy(cuda_stream_);
	cuda_context_->destroy();
	cuda_engine_->destroy();
	uninit_ = true;

	return true;
}

db_detector::~db_detector()
{
	un_init();
}

bool db_detector::process_image(const cv::Mat& img, std::vector<OCRPredictResult>& ocr_results, const float threshold)
{
	m_threshold_ = threshold;
	ocr_results.clear();

	float ratio_h{};
	float ratio_w{};

	cv::Mat src_img;
	cv::Mat resize_img;
	img.copyTo(src_img);

	resize_op_.Run(img, resize_img, limit_type_,
	               limit_side_len_, ratio_h, ratio_w,
	               use_tensorrt_);

	cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

	auto resized = cv::Mat(cv::Size(input_size_.width, input_size_.height), CV_8UC3, cv::Scalar(124, 116, 104));
	resize_img.copyTo(resized(cv::Rect{0, 0, resize_img.cols, resize_img.rows}));

	const auto scaled_size = cv::Size(resize_img.cols, resize_img.rows);

	normalize_op_.Run(&resized, mean_, scale_, is_scale_);

	cv::split(resized, input_wrappers_);

	auto ret = cudaMemcpyAsync(array_dev_memory_[i_input_index_], array_host_memory_[i_input_index_],
	                           array_size_[i_input_index_], cudaMemcpyHostToDevice, cuda_stream_);
	auto ret1 = cuda_context_->enqueueV2(array_dev_memory_, cuda_stream_, nullptr);
	ret = cudaMemcpyAsync(array_host_memory_[i_output_index_], array_dev_memory_[i_output_index_],
	                      array_size_[i_output_index_], cudaMemcpyDeviceToHost, cuda_stream_);
	ret = cudaStreamSynchronize(cuda_stream_);

	decode_outputs(ocr_results, static_cast<float*>(array_host_memory_[i_output_index_]), src_img, ratio_h, ratio_w,
	               scaled_size);

	return true;
}

void db_detector::decode_outputs(std::vector<OCRPredictResult>& ocr_results, float* prob, const cv::Mat& src_img,
                                 const float ratio_h, const float ratio_w, const cv::Size& scaled_size)
{
	int n = output_dim2_ * output_dim3_;
	std::vector<float> pred(n, 0.0f);
	std::vector<unsigned char> cbuf(n, ' ');

	for (int i = 0; i < n; i++)
	{
		pred[i] = prob[i];
		cbuf[i] = static_cast<unsigned char>(prob[i] * 255);
	}

	cv::Mat tmp_cbuf_map(output_dim2_, output_dim3_, CV_8UC1, cbuf.data());
	cv::Mat tmp_pred_map(output_dim2_, output_dim3_, CV_32F, pred.data());

	cv::Mat cbuf_map = tmp_cbuf_map(cv::Range(0, scaled_size.height), cv::Range(0, scaled_size.width));
	cv::Mat pred_map = tmp_pred_map(cv::Range(0, scaled_size.height), cv::Range(0, scaled_size.width)).clone();

	const double threshold = det_db_thresh_ * 255;
	constexpr double max_value = 255;
	cv::Mat bit_map;
	cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
	if (use_dilation_)
	{
		cv::Mat dila_ele =
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
		cv::dilate(bit_map, bit_map, dila_ele);
	}

	auto boxes = post_processor_.BoxesFromBitmap(
		pred_map, bit_map, det_db_box_thresh_, det_db_unclip_ratio_,
		det_db_score_mode_);

	boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, src_img);

	for (int i = 0; i < boxes.size(); i++)
	{
		OCRPredictResult res;
		res.box = boxes[i];
		//std::stringstream ss;
		//ss << "BBOX: [";
		//for (auto &box : boxes[i])
		//{
		//    ss << "[";
		//    for (auto &p : box)
		//    {
		//        ss << p << " ";
		//    }
		//    ss << "] ";
		//}
		//ss << "]";
		//LOG(INFO) << ss.str();
		ocr_results.push_back(res);
	}

	// sort boex from top to bottom, from left to right
	Utility::sorted_boxes(ocr_results);
}
