#include <Eigen/Core>
#include <Eigen/Geometry>
#include <openvino/openvino.hpp>

int main() {
  auto compiled_model = ov::Core().compile_model(
      "/home/dknt/Project/dl_deploy/ovinf/model/policy.onnx", "CPU");
  ov::InferRequest infer_request = compiled_model.create_infer_request();
  auto input_info = compiled_model.input();

  size_t sz = input_info.get_shape()[0];
  Eigen::VectorXf input_vector(sz);
  input_vector.setZero();

  ov::Tensor input_tensor =
      ov::Tensor(input_info.get_element_type(), input_info.get_shape(),
                 input_vector.data());
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();
  auto output_info = compiled_model.output();
  ov::Tensor output_tensor = infer_request.get_output_tensor();
  Eigen::Map<Eigen::VectorXf> output_vector(
      static_cast<float*>(output_tensor.data()), output_tensor.get_shape()[0]);

  std::cout << "Output: " << output_vector.transpose() << std::endl;

  return 0;
}
