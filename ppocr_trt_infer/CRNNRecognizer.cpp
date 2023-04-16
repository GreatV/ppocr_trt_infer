#include "sys/stat.h"
#include <fstream>

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <NvInferRuntime.h>

#include "CRNNRecognizer.h"

#include "logging.h"

#define INPUT_NAME "x"
#define OUTPUT_NAME "softmax_5.tmp_0"


static bool if_file_exists(const char* file_name)
{
	struct stat my_stat{};
	return (stat(file_name, &my_stat) == 0);
}

crnn_recognizer::crnn_recognizer(): cuda_engine_(nullptr), cuda_context_(nullptr), cuda_stream_(nullptr),
                                    i_input_index_(0), i_output_index_(0)
{
}

void crnn_recognizer::load_onnx(const std::string& model_name)
{
	Logger logger;
	//根据tensorrt pipeline 构建网络
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	builder->setMaxBatchSize(1);
	constexpr auto explicit_batch = 1U << static_cast<uint32_t>(
		nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	parser->parseFromFile(model_name.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	config->setMaxWorkspaceSize(1ULL << 30);
	cuda_engine_ = builder->buildEngineWithConfig(*network, *config);

	std::string trt_name = model_name;
	const size_t sep_pos = trt_name.find_last_of(".");
	trt_name = trt_name.substr(0, sep_pos) + ".trt";
	const nvinfer1::IHostMemory* gie_model_stream = cuda_engine_->serialize();
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

void crnn_recognizer::load_trt(const std::string& model_name)
{
	Logger logger;
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
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

bool crnn_recognizer::init(const std::string& model_name, const std::string& label_path, const float threshold)
{
	threshold_ = threshold;
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

	label_list_ = Utility::ReadDict(label_path);
	label_list_.insert(label_list_.begin(),
	                   "#"); // blank char for ctc
	label_list_.emplace_back(" ");

	// 分配输入输出的空间,DEVICE侧和HOST侧
	i_input_index_ = cuda_engine_->getBindingIndex(INPUT_NAME);
	i_output_index_ = cuda_engine_->getBindingIndex(OUTPUT_NAME);

	nvinfer1::Dims dims_i = cuda_engine_->getBindingDimensions(i_input_index_);
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
	LOG(INFO) << "output dims " << dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2];

	predict_shape_[0] = dims_i.d[0];
	predict_shape_[1] = dims_i.d[1];
	predict_shape_[2] = dims_i.d[2];

	size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2];

	cudaMalloc(&array_dev_memory_[i_output_index_], size * sizeof(float));
	array_host_memory_[i_output_index_] = malloc(size * sizeof(float));
	array_size_[i_output_index_] = size * sizeof(float);
	cudaStreamCreate(&cuda_stream_);
	uninit_ = false;

	return true;
}

bool crnn_recognizer::un_init()
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
}

crnn_recognizer::~crnn_recognizer()
{
	un_init();
}

bool crnn_recognizer::process_image(const std::vector<cv::Mat>& img_list, std::vector<OCRPredictResult>& ocr_results)
{
	int img_num = static_cast<int>(img_list.size());

	std::vector<std::string> rec_texts(img_list.size(), "");
	std::vector<float> rec_text_scores(img_list.size(), 0);

	std::vector<float> width_list;
	width_list.reserve(img_num);
	for (int i = 0; i < img_num; i++)
	{
		width_list.push_back(static_cast<float>(img_list[i].cols) / static_cast<float>(img_list[i].rows));
	}
	std::vector<int> indices = Utility::argsort(width_list);

	for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no++)
	{
		int img_h = rec_image_shape_[1];
		int img_w = rec_image_shape_[2];
		float max_wh_ratio = static_cast<float>(img_w) / static_cast<float>(img_h);
		int h = img_list[indices[beg_img_no]].rows;
		int w = img_list[indices[beg_img_no]].cols;
		float wh_ratio = static_cast<float>(w) / static_cast<float>(h);
		max_wh_ratio = std::max(max_wh_ratio, wh_ratio);

		cv::Mat src_img;
		img_list[indices[beg_img_no]].copyTo(src_img);
		cv::Mat resize_img;
		resize_op_.Run(src_img, resize_img, max_wh_ratio, use_tensorrt_, rec_image_shape_);

		cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
		auto step = static_cast<int>(std::ceil(
			static_cast<float>(resize_img.cols) / static_cast<float>(input_size_.width)));

		std::string str_res{};
		float score{};
		int count = 0;
		int pos = 0;
		// TODO: fix ctc decode error
		for (int i = 0; i < step; ++i)
		{
			auto offset_width = i * input_size_.width;
			// auto offset_w = (int)std::ceil(pos * (float) m_InputSize.width / (float) predict_shape_[1]);
			auto width = std::min(offset_width + input_size_.width, resize_img.cols);
			// LOG(INFO) << "image number: " << beg_img_no << ", step: " << step << ", count: " << count << ", offset_w: " << offset_w << ", w: " << w;

			auto resized = cv::Mat(cv::Size(input_size_.width, input_size_.height), CV_8UC3, cv::Scalar(127, 127, 127));
			cv::Mat tmp_img = resize_img(
				cv::Range(0, resize_img.rows),
				cv::Range(offset_width, width)
			);
			tmp_img.copyTo(resized(cv::Rect{0, 0, width - offset_width, resize_img.rows}));

			normalize_op_.Run(&resized, mean_, scale_, is_scale_);

			cv::split(resized, input_wrappers_);

			auto ret = cudaMemcpyAsync(array_dev_memory_[i_input_index_], array_host_memory_[i_input_index_],
			                           array_size_[i_input_index_], cudaMemcpyHostToDevice, cuda_stream_);
			auto ret1 = cuda_context_->enqueueV2(array_dev_memory_, cuda_stream_, nullptr);
			ret = cudaMemcpyAsync(array_host_memory_[i_output_index_], array_dev_memory_[i_output_index_],
			                      array_size_[i_output_index_], cudaMemcpyDeviceToHost, cuda_stream_);
			ret = cudaStreamSynchronize(cuda_stream_);

			std::string tmp_str{};
			float tmp_score{};
			int tmp_count = 0;
			int tmp_pos = -1;
			decode_outputs(tmp_str, tmp_score, tmp_count, tmp_pos,
			               static_cast<float*>(array_host_memory_[i_output_index_]));

			str_res += tmp_str;
			count += tmp_count;
			score += tmp_score;
			pos += (1 + tmp_pos);
		}

		if (count != 0)
		{
			score /= static_cast<float>(count);
		}

		rec_texts[indices[beg_img_no]] = str_res;
		rec_text_scores[indices[beg_img_no]] = score;
	}

	// output rec results
	for (size_t i = 0; i < rec_texts.size(); i++)
	{
		ocr_results[i].text = rec_texts[i];
		LOG(INFO) << rec_texts[i];
		ocr_results[i].score = rec_text_scores[i];
	}

	return true;
}

void crnn_recognizer::decode_outputs(std::string& str_res, float& score, int& count, int& pos, const float* prob)
{
	// ctc decode
	int last_index = 0;
	// int count = 0;

	// int pos = 0;
	for (int n = 0; n < predict_shape_[1]; n++)
	{
		// get idx
		int argmax_idx = static_cast<int>(Utility::argmax(
			&prob[(n + 0) * predict_shape_[2]],
			&prob[(n + 1) * predict_shape_[2]]));
		// get score
		const float max_value = *std::max_element(
			&prob[(n + 0) * predict_shape_[2]],
			&prob[(n + 1) * predict_shape_[2]]);

		if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index)))
		{
			score += max_value;
			count += 1;
			str_res += label_list_[argmax_idx];
			pos = n;
		}
		last_index = argmax_idx;
	}
	// LOG(INFO) << "pos: " << pos;
	// score /= count;
	// if (std::isnan(score))
	// {
	//     return;
	// }
}
