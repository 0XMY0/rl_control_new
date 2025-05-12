#ifndef CONTROLLER_BASE_HPP
#define CONTROLLER_BASE_HPP

#include "robot/base/robot_base.hpp"

namespace ovinf {

// TODO

/**
 * @brief Controller base.
 *        This is just an example controller, no controller in this library is
 *        implemented in this.
 *        WarmUp() -> Step() -> Stop()
 */
template <typename T>
class ControllerBase {
 public:
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  ControllerBase() = delete;
  ControllerBase(RobotBase<T>::RobotPtr robot, YAML::Node const& config)
      : robot_(robot) {}

  /**
   * @brief Controller WarmUp
   *
   */
  virtual void WarmUp() = 0;

  virtual void Init() = 0;

  /**
   * @brief Step function. User should update observation and send torque
   *        command mannually.
   *
   */
  virtual void Step(bool set_target = true) = 0;

  virtual void Stop() = 0;

 protected:
  void ComputeJointPd() {
    for (size_t i = 0; i < robot_->joint_size_; ++i) {
      T position_error = robot_->executor_->JointTargetPosition()[i] -
                         robot_->observer_->JointActualPosition()[i];

      robot_->executor_->JointTargetTorque()[i] =
          p_gains_(i) * position_error;  // P
      robot_->executor_->JointTargetTorque()[i] +=
          -d_gains_(i) * robot_->observer_->JointActualVelocity()[i];  // D
    }
  }

 protected:
  RobotBase<T>::RobotPtr robot_;
  VectorT p_gains_;
  VectorT d_gains_;

  bool ready_ = false;
};

}  // namespace ovinf
#endif  // !CONTROLLER_BASE_HPP
