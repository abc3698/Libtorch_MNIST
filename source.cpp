#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include "opencv2/opencv.hpp"

int main() {	
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("mnist_cnn.pt");

	assert(module != nullptr);
	std::cout << "ok\n";

	module->to(at::kCUDA);

	cv::Mat image = cv::imread("img_1.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat image2 = cv::imread("img_2.jpg", cv::IMREAD_GRAYSCALE);
	at::Tensor tensor_image =
		torch::from_blob(image.data, { 1, 1, 28, 28 }, at::kByte);
	tensor_image = tensor_image.to(at::kFloat);

	at::Tensor tensor_image2 =
		torch::from_blob(image2.data, { 1, 1, 28, 28 }, at::kByte);
	tensor_image2 = tensor_image2.to(at::kFloat);

	auto result = torch::cat({ tensor_image, tensor_image2 });
	result /= 255.;

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(result.to(at::kCUDA));

	at::Tensor output = module->forward(inputs).toTensor();

	std::cout << output.sizes() << std::endl;

	for (int s = 0; s < output.sizes()[0]; ++s)
	{
		std::cout << at::argmax(output[s]) << std::endl;
	}
}