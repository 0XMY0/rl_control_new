#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include "ovinf/ovinf_factory.hpp"

int main() {
  YAML::Node config =
      YAML::LoadFile("/home/wsy/Library/ovinf/config/humanoid.yaml");

  auto policy = ovinf::PolicyFactory::CreatePolicy(config["inference"]);

  // warm-up（一般15帧）
  policy->WarmUp(
      {
          .command = Eigen::Vector3f{0.0, 0.0, 0.0},
          .ang_vel = Eigen::Vector3f{0.0, 0.0, 0.0},
          .proj_gravity = Eigen::Vector3f{0.0, 0.0, 0.0},
          .joint_pos = Eigen::VectorXf::Zero(12),
          .joint_vel = Eigen::VectorXf::Zero(12),
      },
      20);

  // 输入多帧观测
  const int num_frames = 30;
  for (int i = 0; i < num_frames; ++i) {
    bool status = policy->InferUnsync({
      .command = Eigen::Vector3f{0.0, 0.0, 0.0},
      .ang_vel = Eigen::Vector3f{0.0, 0.0, 0.0},
      .proj_gravity = Eigen::Vector3f{0.0, 0.0, 0.0},
      .joint_pos = Eigen::VectorXf::Zero(12),
      .joint_vel = Eigen::VectorXf::Zero(12),
    });

    if (!status) {
      std::cerr << "[Frame " << i << "] Inference call failed!" << std::endl;
    }

    // 获取并打印结果（可选：只每N帧打印一次）
    auto res = policy->GetResult();
    if (res.has_value()) {
      std::cout << "[Frame " << i << "] Result: " << res.value().transpose() << std::endl;
    } else {
      std::cout << "[Frame " << i << "] Result not ready." << std::endl;
    }
  }

  return 0;
}
