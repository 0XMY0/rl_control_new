#ifndef POLICY_CONTROLLER_HPP
#define POLICY_CONTROLLER_HPP

#include <chrono>
#include "controller/controller_base.hpp"
#include "ovinf/ovinf_factory.hpp"

namespace ovinf {

class PolicyController : public ControllerBase<float> {
 public:
  using Ptr = std::shared_ptr<PolicyController>;

  PolicyController() = delete;
  ~PolicyController() = default;

  PolicyController(RobotBase<float>::RobotPtr robot, YAML::Node const& config)
      : ControllerBase<float>(robot, config),
        decimation_(config["decimation"].as<int>()) {
    command_ = VectorT::Zero(3);
    counter_ = 0;
    inference_net_ = ovinf::PolicyFactory::CreatePolicy(config["inference"]);
  }

  virtual void WarmUp() final {
    inference_net_->WarmUp(
        {.command = command_,
         .ang_vel = robot_->observer_->AngularVelocity(),
         .proj_gravity = robot_->observer_->ProjGravity(),
         .joint_pos = robot_->observer_->JointActualPosition().segment(0, 12),
         .joint_vel = robot_->observer_->JointActualVelocity().segment(0, 12)});
  }

  virtual void Init() final {
    counter_ = 0;
    ready_ = true;
  }

  virtual void Step(bool set_target = true) final {
    if (!ready_) {
      std::cerr << "PolicyController not ready" << std::endl;
      return;
    }

    if (counter_++ % decimation_ == 0) {
      inference_net_->InferUnsync(
          {.command = command_,
           .ang_vel = robot_->observer_->AngularVelocity(),
           .proj_gravity = robot_->observer_->ProjGravity(),
           .joint_pos = robot_->observer_->JointActualPosition().segment(0, 12),
           .joint_vel = robot_->observer_->JointActualVelocity().segment(0, 12)});
    }

    if (set_target) {
      auto target_pos = inference_net_->GetResult();
      if (target_pos.has_value()) {
        for (size_t i = 0; i < 12; i++) {
          robot_->executor_->JointTargetPosition()[i] = target_pos.value()[i];
        }
        for (size_t i = 12; i < robot_->joint_size_; i++) {
          robot_->executor_->JointTargetPosition()[i] = 0.0;  // 注意是 i，不是固定 12
        }
      }
    }
  }

  virtual void Stop() final { ready_ = false; }

  VectorT& GetCommand() { return command_; }

 private:
  const int decimation_;
  VectorT command_;
  size_t counter_ = 0;
  ovinf::BasePolicy<float>::BasePolicyPtr inference_net_;
};

}  // namespace ovinf

#endif  // !POLICY_CONTROLLER_HPP
