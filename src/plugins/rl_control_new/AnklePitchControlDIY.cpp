#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <bodyctrl_msgs/MotorStatusMsg.h>
#include <bodyctrl_msgs/CmdMotorCtrl.h>
#include <bodyctrl_msgs/MotorName.h>
#include <Eigen/Dense>
#include "parallel_ankle.hpp"  // 记得包含你自己的 ParallelAnkle 类头文件

namespace rl_control_new
{

class AnklePitchTestDIY : public nodelet::Nodelet
{
public:
  AnklePitchTestDIY() {}

private:
  virtual void onInit()
  {
    ros::NodeHandle& nh = getPrivateNodeHandle();

    pub_motor_cmd_ = nh.advertise<bodyctrl_msgs::CmdMotorCtrl>("/BodyControl/motor_ctrl", 10);
    sub_motor_state_ = nh.subscribe("/BodyControl/motor_state", 100, &AnklePitchTestDIY::motorStateCallback, this);
    sub_joy_ = nh.subscribe("/sbus_data", 10, &AnklePitchTestDIY::joyCallback, this);

    timer_ = nh.createTimer(ros::Duration(0.002), &AnklePitchTestDIY::timerCallback, this); // 500Hz

    ankle_target_pitch_ = 0.0;
    joy_received_ = false;
    motor_pos_.setZero();
    motor_vel_.setZero();

    // 初始化左踝关节

  
    ROS_INFO("[AnklePitchTestDIY] Initialized.");
  }

  void joyCallback(const sensor_msgs::Joy::ConstPtr& msg)
  {
    joy_received_ = true;
    last_joy_time_ = ros::Time::now();

    if (msg->buttons.size() < 2)
      return;

    if (msg->buttons[0]) {
      ankle_target_pitch_ += 0.01;  // 慢慢增加
    }
    if (msg->buttons[1]) {
      ankle_target_pitch_ -= 0.01;  // 慢慢减少
    }

    ankle_target_pitch_ = std::min(0.5, std::max(-0.5, ankle_target_pitch_));
  }

  void motorStateCallback(const bodyctrl_msgs::MotorStatusMsg::ConstPtr& msg)
  {
    for (const auto& m : msg->status) {
      if (m.name == bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_5) {
        motor_pos_(0) = m.pos;
        motor_vel_(0) = m.speed;
      } else if (m.name == bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_6) {
        motor_pos_(1) = m.pos;
        motor_vel_(1) = m.speed;
      }
    }
  }

  void timerCallback(const ros::TimerEvent&)
  {
    if (joy_received_ && (ros::Time::now() - last_joy_time_).toSec() > 5)
    {
      if (std::abs(ankle_target_pitch_) > 0.01) {
        ankle_target_pitch_ *= 0.98;
      } else {
        ankle_target_pitch_ = 0.0;
      }
    }

    // === 并联踝计算 ===
    // 当前pitch
    auto left_res = left_ankle.ForwardKinematics(motor_pos_(0), motor_pos_(1));
    // 当前pitch速度
    float current_pitch_vel = left_ankle.VelocityMapping(motor_vel_(0), motor_vel_(1));

    // 目标两个电机位置
    auto target_motor_angles = left_ankle.InverseKinematics(ankle_target_pitch_);

    // PD力矩
    float kp = 50.0f;
    float kd = 2.0f;
    float position_error = ankle_target_pitch_ - current_pitch;
    float velocity_error = 0.0f - current_pitch_vel;
    float ankle_torque = kp * position_error + kd * velocity_error;

    // 踝力矩到电机力矩映射
    auto motor_torque = left_ankle_.TorqueMapping(ankle_torque);

    // === 发布指令 ===
    bodyctrl_msgs::CmdMotorCtrl cmd_msg;
    cmd_msg.header.stamp = ros::Time::now();

    bodyctrl_msgs::MotorCtrl motor5, motor6;
    motor5.name = bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_5;
    motor6.name = bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_6;

    motor5.pos = target_motor_angles.first;
    motor5.tor = motor_torque.first;
    motor5.kp = 50.0;
    motor5.kd = 2.0;

    motor6.pos = target_motor_angles.second;
    motor6.tor = motor_torque.second;
    motor6.kp = 50.0;
    motor6.kd = 2.0;

    cmd_msg.cmds.push_back(motor5);
    cmd_msg.cmds.push_back(motor6);

    pub_motor_cmd_.publish(cmd_msg);
  }

  ros::Publisher pub_motor_cmd_;
  ros::Subscriber sub_motor_state_;
  ros::Subscriber sub_joy_;
  ros::Timer timer_;

  double ankle_target_pitch_;
  bool joy_received_;
  ros::Time last_joy_time_;

  Eigen::Vector2d motor_pos_;
  Eigen::Vector2d motor_vel_;

  ParallelAnkle<float> left_ankle({
    .l_bar1 = 0.06, //
    .l_rod1 = 0.215,  // 0.235
    .r_a2 = {0, 0.044, 0.215},
    .r_b2_0 = {-0.056, 0.044, 0.237},
    .r_c2_0 = {-0.056, 0.044, -0.0022},
    .l_bar2 = 0.04,
    .l_rod2 = 0.14,
    .r_a1 = {0, -0.043, 0.141},
    .r_b1_0 = {-0.056, -0.043, 0.163},
    .r_c1_0 = {-0.056, -0.043, -0.0023},}
   1e-6);
};

} // namespace rl_control_new

PLUGINLIB_EXPORT_CLASS(rl_control_new::AnklePitchTestDIY, nodelet::Nodelet)
