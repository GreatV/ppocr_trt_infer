#ifndef OCR_DET_INFER_H
#define OCR_DET_INFER_H

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/core.hpp>

#include "OCRPostProcess.h"
#include "OCRPreProcess.h"


class db_detector
{
public:
	db_detector();
	~db_detector();
	bool init(const std::string& model_name, float threshold);
	bool un_init();
	bool process_image(const cv::Mat& img, std::vector<OCRPredictResult>& ocr_results, float threshold);

private:
	void load_onnx(const std::string& model_name);
	void load_trt(const std::string& model_name);
	void decode_outputs(std::vector<OCRPredictResult>& ocr_results, float* prob, const cv::Mat& src_img, float ratio_h,
	                    float ratio_w, const cv::Size& scaled_size);

	nvinfer1::ICudaEngine* cuda_engine_;
	nvinfer1::IExecutionContext* cuda_context_;
	cudaStream_t cuda_stream_;
	void* array_dev_memory_[2]{0};
	void* array_host_memory_[2]{0};
	int array_size_[2]{0};
	int i_input_index_;
	int i_output_index_;
	int output_dim2_{};
	int output_dim3_{};
	std::vector<cv::Mat> input_wrappers_{};
	cv::Size input_size_{};

	bool uninit_ = false;
	float m_threshold_;

	std::string limit_type_ = "max";
	int limit_side_len_ = 960;

	double det_db_thresh_ = 0.3;
	double det_db_box_thresh_ = 0.5;
	double det_db_unclip_ratio_ = 2.0;
	std::string det_db_score_mode_ = "slow";
	bool use_dilation_ = false;

	bool use_tensorrt_ = false;
	std::string precision_ = "fp32";

	// BGR
	// std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
	// std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
	// RGB
	std::vector<float> mean_ = {0.406f, 0.456f, 0.485f};
	std::vector<float> scale_ = {1 / 0.225f, 1 / 0.224f, 1 / 0.229f};
	bool is_scale_ = true;

	// pre-process
	ResizeImgType0 resize_op_;
	Normalize normalize_op_;
	Permute permute_op_;

	// post-process
	DBPostProcessor post_processor_;
};

#endif
