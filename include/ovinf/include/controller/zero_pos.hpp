#ifndef ZERO_POS_HPP
#define ZERO_POS_HPP

#include <chrono>

#include "controller/controller_base.hpp"

namespace ovinf {
class ZeroPosController : public ControllerBase<float> {
 public:
  using Ptr = std::shared_ptr<ZeroPosController>;

  ZeroPosController() = delete;
  ~ZeroPosController() = default;

  ZeroPosController(RobotBase<float>::RobotPtr robot, YAML::Node const& config)
      : ControllerBase<float>(robot, config) {
    p_gains_ = VectorT::Zero(robot_->joint_size_);
    d_gains_ = VectorT::Zero(robot_->joint_size_);
    zero_pos_ = VectorT::Zero(robot_->joint_size_);
    start_pos_ = VectorT::Zero(robot_->joint_size_);
    duration_ = config["duration"].as<float>();

    for (auto const& pair : robot_->joint_names_) {
      p_gains_(pair.second) = config["p_gains"][pair.first].as<float>();
      d_gains_(pair.second) = config["d_gains"][pair.first].as<float>();
    }
  }

  virtual void WarmUp() final {}

  virtual void Init() final {
    start_pos_ = this->robot_->observer_->JointActualPosition();
    start_time_ = std::chrono::high_resolution_clock::now();
    ready_ = true;
  }

  virtual void Step(bool set_target = true) final {
    if (!ready_) {
      std::cerr << "ZeroPosController not ready" << std::endl;
      return;
    }

    auto current_time = std::chrono::high_resolution_clock::now();
    float ratio = std::min(
        float(1.0),
        std::chrono::duration<float>(current_time - start_time_).count() /
            duration_);

    VectorT temp_target = start_pos_ + ratio * (zero_pos_ - start_pos_);

    if (set_target) {
      robot_->executor_->JointTargetPosition() = temp_target;
      ComputeJointPd();
    }
  }

  virtual void Stop() final { ready_ = false; }

 private:
  VectorT zero_pos_;

  std::chrono::high_resolution_clock::time_point start_time_;
  VectorT start_pos_;
  float duration_ = 0.0f;
};

}  // namespace ovinf

#endif  // !ZERO_POS_HPP
