#ifndef OCR_REC_INFER_H
#define OCR_REC_INFER_H

#include "NvInfer.h"
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <opencv2/core.hpp>

#include "OCRPreProcess.h"
#include "OCRUtility.h"


class crnn_recognizer
{
public:
	crnn_recognizer();
	~crnn_recognizer();
	bool init(const std::string& model_name, const std::string& label_path, float threshold);
	bool un_init();
	bool process_image(const std::vector<cv::Mat>& img_list, std::vector<OCRPredictResult>& ocr_results);

private:
	void load_onnx(const std::string& model_name);
	void load_trt(const std::string& model_name);
	void decode_outputs(std::string& str_res, float& score, int& count, int& pos, const float* prob);

private:
	nvinfer1::ICudaEngine* cuda_engine_;
	nvinfer1::IExecutionContext* cuda_context_;
	cudaStream_t cuda_stream_;
	void* array_dev_memory_[2]{0};
	void* array_host_memory_[2]{0};
	int array_size_[2]{0};
	int i_input_index_;
	int i_output_index_;
	std::vector<cv::Mat> input_wrappers_{};
	cv::Size input_size_{};

	int predict_shape_[3]{};

	bool uninit_ = false;
	float threshold_ = 0.25f;

	std::vector<std::string> label_list_{};

	std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
	std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

	bool is_scale_ = true;
	bool use_tensorrt_ = false;
	std::string precision_ = "fp32";
	// int rec_batch_num_ = 1;
	int rec_img_h_ = 48;
	int rec_img_w_ = 320;
	std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};
	// pre-process
	CrnnResizeImg resize_op_;
	Normalize normalize_op_;
	PermuteBatch permute_op_;
};

#endif
