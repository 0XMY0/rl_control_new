#ifndef OVINF_LSTM_CHZ_H
#define OVINF_LSTM_CHZ_H
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <numeric>

#include <openvino/openvino.hpp>

// Helper function to convert ov::Shape to a string representation
namespace ov
{ // You can put it in the ov namespace or your own
	std::string shape_to_string(const ov::Shape &shape)
	{
		std::ostringstream oss;
		oss << "[";
		for (size_t i = 0; i < shape.size(); ++i)
		{
			oss << shape[i];
			if (i < shape.size() - 1)
			{
				oss << ", ";
			}
		}
		oss << "]";
		return oss.str();
	}
} // namespace ov

namespace chz
{
	class LSTMCHZPolicy
	{
	public:
		LSTMCHZPolicy()
		{
			// --- Configuration (Adjust these to your model's specifics) ---
			const std::string model_path = "/home/ubuntu/chz/model/lstm/policy_lstm_1.onnx"; // Path to your .xml
			const std::string device_name = "CPU";

			// These names MUST match the names in your exported ONNX/IR model
			std::string obs_input_name_ = "observation";
			std::string h_state_input_name_ = "hidden_state";
			std::string c_state_input_name_ = "cell_state";

			std::string action_output_name_ = "action";
			std::string h_state_output_name_ = "new_hidden_state";
			std::string c_state_output_name_ = "new_cell_state";

			std::cout << "Initializing LSTMCHZPolicy..." << std::endl;
			std::cout << "Loading OpenVINO Runtime" << std::endl;

			std::cout << "Reading model: " << model_path << std::endl;
			std::shared_ptr<ov::Model> model = core_.read_model(model_path);

			// --- Get input and output properties and validate ---
			// Observation Input
			obs_input_port_ = model->input(obs_input_name_);
			const ov::Shape &obs_shape = obs_input_port_.get_shape();
			input_dim = obs_shape[1];
			if (obs_shape.size() != 2 && obs_shape[0] != 1)
			{
				throw std::runtime_error("Observation input shape mismatch. Expected [1, input_dim]. Got " +
										 ov::shape_to_string(obs_shape));
			}
			std::cout << "Observation input (" << obs_input_name_ << ") shape: " << ov::shape_to_string(obs_shape)
					  << ", input_dim=" << input_dim << std::endl;

			// Hidden State Input
			h_state_input_port_ = model->input(h_state_input_name_);
			const ov::Shape &h_in_shape = h_state_input_port_.get_shape();
			lstm_hidden_size_ = h_in_shape.back();
			std::cout << "H-state input (" << h_state_input_name_ << ") shape: " << ov::shape_to_string(h_in_shape)
					  << ", lstm_hidden_size=" << lstm_hidden_size_ << std::endl;

			// Cell State Input
			c_state_input_port_ = model->input(c_state_input_name_);
			const ov::Shape &c_in_shape = c_state_input_port_.get_shape();
			if (c_in_shape.back() != lstm_hidden_size_)
			{ // Should match h_state
				throw std::runtime_error("C-state input size mismatch with H-state.");
			}
			std::cout << "C-state input (" << c_state_input_name_ << ") shape: " << ov::shape_to_string(c_in_shape) << std::endl;

			// Action Output
			action_output_port_ = model->output(action_output_name_);
			const ov::Shape &action_shape = action_output_port_.get_shape();
			output_dim = action_shape[1]; // Assuming shape [1, output_dim]
			if (action_shape.size() != 2 || action_shape[0] != 1)
			{
				throw std::runtime_error("Action output shape mismatch. Expected [1, 1, output_dim]. Got " +
										 ov::shape_to_string(action_shape));
			}
			std::cout << "Action output (" << action_output_name_ << ") shape: " << ov::shape_to_string(action_shape)
					  << ", output_dim=" << output_dim << std::endl;

			// Hidden State Output
			h_state_output_port_ = model->output(h_state_output_name_);
			const ov::Shape &h_out_shape = h_state_output_port_.get_shape();
			if (h_out_shape.back() != lstm_hidden_size_)
			{
				throw std::runtime_error("H-state output size mismatch with H-state input size.");
			}
			std::cout << "H-state output (" << h_state_output_name_ << ") shape: " << ov::shape_to_string(h_out_shape) << std::endl;

			// Cell State Output
			c_state_output_port_ = model->output(c_state_output_name_);
			const ov::Shape &c_out_shape = c_state_output_port_.get_shape();
			if (c_out_shape.back() != lstm_hidden_size_)
			{
				throw std::runtime_error("C-state output size mismatch with C-state input size.");
			}
			std::cout << "C-state output (" << c_state_output_name_ << ") shape: " << ov::shape_to_string(c_out_shape) << std::endl;

			// Compile model
			std::cout << "Compiling model for device: " << device_name << std::endl;
			compiled_model_ = core_.compile_model(model, device_name);
			infer_request_ = compiled_model_.create_infer_request();

			// Initialize Eigen members
			observation_.resize(input_dim);
			action_.resize(output_dim);
			// For a single layer, unidirectional LSTM, state vector size is lstm_hidden_size_
			// If your model uses num_layers or bidirectional, the state tensor shape is
			// [num_layers * num_directions, batch_size, hidden_size].
			// Here, we assume the Eigen vector will map to the flattened version for batch_size=1.
			h_state_.resize(lstm_hidden_size_);
			c_state_.resize(lstm_hidden_size_);

			Reset(); // Initialize states to zero

			std::cout << "LSTMCHZPolicy initialized. nx=" << input_dim << ", ny=" << output_dim
					  << ", lstm_hidden_size=" << lstm_hidden_size_ << std::endl;
		}

		// Reset LSTM hidden states (e.g., at the start of an episode)
		void Reset()
		{
			if (h_state_.size() > 0)
				h_state_.setZero();
			if (c_state_.size() > 0)
				c_state_.setZero();
			std::cout << "LSTM states reset." << std::endl;
		}

		// Set observation. The input Eigen::VectorXf 'obs' should have size 'input_dim'.
		void SetObservation(const Eigen::VectorXf &obs)
		{
			if (obs.size() != input_dim)
			{
				throw std::runtime_error("Provided observation Eigen vector size (" + std::to_string(obs.size()) +
										 ") does not match expected nx (" + std::to_string(input_dim) + ").");
			}
			observation_ = obs;
		}

		// Performs inference using current 'observation_' and internal h/c states.
		// Updates internal h/c states and 'action_'.
		void Infer()
		{
			// 1. Get Tensors for inputs
			ov::Tensor obs_tensor = infer_request_.get_tensor(obs_input_port_);
			ov::Tensor h_in_tensor = infer_request_.get_tensor(h_state_input_port_);
			ov::Tensor c_in_tensor = infer_request_.get_tensor(c_state_input_port_);

			// 2. Fill input Tensors
			// Observation: Expected shape [1, 1, input_dim]
			float *obs_data = obs_tensor.data<float>();
			std::copy_n(observation_.data(), input_dim, obs_data);

			// Hidden state: Expected shape for ONNX LSTM op is often [num_dir, batch, hidden_size]
			// If using Keras default single layer, tf2onnx might make this [1, 1, hidden_size]
			// Our Eigen h_state_ has size `lstm_hidden_size_`
			float *h_in_data = h_in_tensor.data<float>();
			std::copy_n(h_state_.data(), h_state_.size(), h_in_data);

			float *c_in_data = c_in_tensor.data<float>();
			std::copy_n(c_state_.data(), c_state_.size(), c_in_data);

			// 3. Perform inference
			infer_request_.infer();

			// 4. Get Tensors for outputs
			ov::Tensor action_tensor = infer_request_.get_tensor(action_output_port_);
			ov::Tensor h_out_tensor = infer_request_.get_tensor(h_state_output_port_);
			ov::Tensor c_out_tensor = infer_request_.get_tensor(c_state_output_port_);

			// 5. Process outputs
			// Action: Expected shape [1, 1, output_dim]
			const float *action_data = action_tensor.data<const float>();
			std::copy_n(action_data, output_dim, action_.data());

			// Update internal states
			const float *h_out_data = h_out_tensor.data<const float>();
			std::copy_n(h_out_data, h_state_.size(), h_state_.data());

			const float *c_out_data = c_out_tensor.data<const float>();
			std::copy_n(c_out_data, c_state_.size(), c_state_.data());
		}

		// Getter for the action produced by Infer()
		Eigen::VectorXf GetAction() const
		{
			return action_;
		}

		// Combined SetObservation, Infer, GetAction
		Eigen::VectorXf Infer(const Eigen::VectorXf &current_obs)
		{
			SetObservation(current_obs);
			Infer();
			return GetAction();
		}

	private:
		// OpenVINO members
		ov::Core core_;
		ov::CompiledModel compiled_model_;
		ov::InferRequest infer_request_;

		// Port objects for direct tensor access (safer than by index)
		ov::Output<const ov::Node> obs_input_port_;
		ov::Output<const ov::Node> h_state_input_port_;
		ov::Output<const ov::Node> c_state_input_port_;
		ov::Output<const ov::Node> action_output_port_;
		ov::Output<const ov::Node> h_state_output_port_;
		ov::Output<const ov::Node> c_state_output_port_;

		// Network data & state members
		Eigen::VectorXf observation_; // Size input_dim
		Eigen::VectorXf action_;	  // Size output_dim
		Eigen::VectorXf h_state_;	  // Size lstm_hidden_size_ (for one layer, one direction, batch=1)
		Eigen::VectorXf c_state_;	  // Size lstm_hidden_size_

		// Model dimensions
		int input_dim = 0;		   // Number of input features
		int output_dim = 0;		   // Number of output features
		int lstm_hidden_size_ = 0; // LSTM internal units
	};

} // namespace chz
#endif