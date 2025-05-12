/**
 * @file AnkleTestNodelet.cpp
 * @brief 测试并联蹊关节，全身关节回零模拟：A键达到-30，B键回到0，C键全身回零，D键关掉PD
 * @author Siyuan Wang
 * @date 2025-04-30
 */
/*
ToDo:
*/
#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <bodyctrl_msgs/MotorStatusMsg.h>
#include <bodyctrl_msgs/MotorName.h>
#include <bodyctrl_msgs/CmdMotorCtrl.h>
#include <sensor_msgs/Joy.h>
#include <Eigen/Dense>
#include "parallel_ankle.hpp"
#include "Joystick.h"
namespace rl_control_new {

class AnkleTestNodelet : public nodelet::Nodelet {
public:
AnkleTestNodelet() : left_params_(), right_params_(), left_ankle_(left_params_, 1e-6f), right_ankle_(right_params_, 1e-6f) {}
~AnkleTestNodelet() noexcept override = default;
private:
    enum class TestState { IDLE = 0, MOVE_TO_TARGET, MOVE_TO_ZERO, FULL_RESET, STOP };

    struct xbox_map_t {
        float a = 0, b = 0, c = 0, d = 0;
        float e = 0, f = 0, g = 0, h = 0;
        float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    };

    void onInit() override {
        auto& nh = getPrivateNodeHandle();

        InitParams();

        pub_motor_cmd_ = nh.advertise<bodyctrl_msgs::CmdMotorCtrl>("/BodyControl/motor_ctrl", 10);
        sub_motor_state_ = nh.subscribe("/BodyControl/motor_state", 1000, &AnkleTestNodelet::OnMotorState, this);
        sub_joy_ = nh.subscribe("/sbus_data", 1000, &AnkleTestNodelet::OnJoyReceived, this);

        NODELET_INFO("AnkleTestNodelet initialized.");
    }

    void InitParams() {
        pi_ = M_PI;
        motor_num_ = 20;
        motor_dir_ = Eigen::VectorXd::Ones(motor_num_);
        motor_dir_ << 1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
                      1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
                      1.0, -1.0, -1.0, -1.0,
                     -1.0, -1.0, -1.0, 1.0;
        zero_cnt_ = Eigen::VectorXd::Zero(motor_num_);
        zero_offset_ = Eigen::VectorXd::Ones(motor_num_);
        zero_offset_ << 0.0, 0.0, -pi_/3.0, 2.0*pi_/3.0, -1.047, -1.047,
                       0.0, 0.0, -pi_/3.0, 2.0*pi_/3.0, -1.047, -1.047,
                       0.0, 0.2618, 0.0, 0.0,
                       0.0, -0.2618, 0.0, 0.0;
           init_pos_ = Eigen::VectorXd::Zero(motor_num_);
           init_pos_.segment(0, 12) << 0.0, 0.0, -0.3, 0.6, -0.30, 0.0,
                                                  0.0, 0.0, -0.3, 0.6, -0.30, 0.0;
        // ankle parameters
        this->left_params_.l_bar1 = 0.06;
        this->left_params_.l_rod1 = 0.215;
        this->left_params_.r_a1 = {0.0, 0.044, 0.215};
        this->left_params_.r_b1_0 = {-0.056, 0.044, 0.237};
        this->left_params_.r_c1_0 = {-0.056, 0.044, 0.022};
        this->left_params_.l_bar2 = 0.06;
        this->left_params_.l_rod2 = 0.14;
        this->left_params_.r_a2 = {0.0, -0.043, 0.141};
        this->left_params_.r_b2_0 = {-0.056, -0.043, 0.163};
        this->left_params_.r_c2_0 = {-0.056, -0.043, 0.023};

        this->right_params_.l_bar1 = 0.06;
        this->right_params_.l_rod1 = 0.14;
        this->right_params_.r_a1 = {0.0, 0.043, 0.141};
        this->right_params_.r_b1_0 = {-0.056, 0.043, 0.163};
        this->right_params_.r_c1_0 = {-0.056, 0.043, 0.023};
        this->right_params_.l_bar2 = 0.06;
        this->right_params_.l_rod2 = 0.215;
        this->right_params_.r_a2 = {0.0, -0.044, 0.215};
        this->right_params_.r_b2_0 = {-0.056, -0.044, 0.237};
        this->right_params_.r_c2_0 = {-0.056, -0.044, 0.022};
    
    
    

        this->right_ankle_ = ParallelAnkle<float>(this->right_params_, 1e-6f);
      

        this->left_ankle_ = ParallelAnkle<float>(left_params_, 1e-6f);


        
   motor_name.insert({0, bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_1});
   motor_name.insert({1, bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_2});
   motor_name.insert({2, bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_3});
   motor_name.insert({3, bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_4});
   motor_name.insert({4, bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_5});
   motor_name.insert({5, bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_6});
   motor_name.insert({6, bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_1});
   motor_name.insert({7, bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_2});
   motor_name.insert({8, bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_3});
   motor_name.insert({9, bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_4});
   motor_name.insert({10, bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_5});
   motor_name.insert({11, bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_6});
   motor_name.insert({12, bodyctrl_msgs::MotorName::MOTOR_ARM_LEFT_1});
   motor_name.insert({13, bodyctrl_msgs::MotorName::MOTOR_ARM_LEFT_2});
   motor_name.insert({14, bodyctrl_msgs::MotorName::MOTOR_ARM_LEFT_3});
   motor_name.insert({15, bodyctrl_msgs::MotorName::MOTOR_ARM_LEFT_4});
   motor_name.insert({16, bodyctrl_msgs::MotorName::MOTOR_ARM_RIGHT_1});
   motor_name.insert({17, bodyctrl_msgs::MotorName::MOTOR_ARM_RIGHT_2});
   motor_name.insert({18, bodyctrl_msgs::MotorName::MOTOR_ARM_RIGHT_3});
   motor_name.insert({19, bodyctrl_msgs::MotorName::MOTOR_ARM_RIGHT_4});

   for (int i = 0; i < motor_num_; i++) {
     motor_id.insert({motor_name[i], i});
   }

    }

    void OnMotorState(const bodyctrl_msgs::MotorStatusMsg::ConstPtr& msg) {
       float q_left_pitch = 0.0f, q_left_roll = 0.0f;
       float q_right_pitch = 0.0f, q_right_roll = 0.0f;
       for (const auto& motor : msg->status) {
           latest_motor_pos_[motor.name] = motor.pos;
           if (motor.name == bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_5)
               q_left_pitch = motor.pos - 1.047 + 0.374;
           else if (motor.name == bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_6)
               q_left_roll = -motor.pos - 1.047 + 0.374;
           else if (motor.name == bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_5)
               q_right_roll = -motor.pos - 1.047 + 0.374;
           else if (motor.name == bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_6)
               q_right_pitch = motor.pos - 1.047 + 0.374;
       }

       auto fk_left = left_ankle_.ForwardKinematics(q_left_pitch, q_left_roll);
       auto fk_right = right_ankle_.ForwardKinematics(q_right_pitch, q_right_roll);
       ROS_INFO_STREAM_THROTTLE(1.5, "[Left Ankle FK] pitch = " << fk_left(0) * 180.0 / M_PI << ", roll = " << fk_left(1) * 180.0 / M_PI);
       ROS_INFO_STREAM_THROTTLE(1.5, "[Right Ankle FK] pitch = " << fk_right(0) * 180.0 / M_PI << ", roll = " << fk_right(1) * 180.0 / M_PI);

        if (test_state_ == TestState::MOVE_TO_TARGET) MoveToPitch(-30.0f);
        else if (test_state_ == TestState::MOVE_TO_ZERO) MoveToPitch(0.0f);
        else if (test_state_ == TestState::FULL_RESET) FullReset();
        else if (test_state_ == TestState::STOP) StopAll();


    }

    void OnJoyReceived(const sensor_msgs::Joy::ConstPtr& msg) {
        xbox_map_.a = msg->axes[8];
        xbox_map_.b = msg->axes[9];
        xbox_map_.c = msg->axes[10];
        xbox_map_.d = msg->axes[11];

        if (xbox_map_.a > 0.5f) {
            test_state_ = TestState::MOVE_TO_TARGET;
            ROS_INFO("[FSM] A -> -30 deg");
        }
        if (xbox_map_.b > 0.5f) {
            test_state_ = TestState::MOVE_TO_ZERO;
            ROS_INFO("[FSM] B -> 0 deg");
        }
        if (xbox_map_.c > 0.5f) {
            test_state_ = TestState::FULL_RESET;
            initialized_reset_ = false;
            ROS_INFO("[FSM] C -> FULL RESET");
        }
        if (xbox_map_.d > 0.5f) {
            test_state_ = TestState::STOP;
            ROS_INFO("[FSM] D -> STOP");
        }
    }

    void MoveToPitch(float pitch_deg) {
        float pitch = pitch_deg * M_PI / 180.0f;
        auto mot = left_ankle_.InverseKinematics(pitch, 0.0f);
        auto mot2 = right_ankle_.InverseKinematics(pitch, 0.0f);
        bodyctrl_msgs::CmdMotorCtrl msg;
        msg.header.stamp = ros::Time::now();

        bodyctrl_msgs::MotorCtrl cmd1, cmd2;
        cmd1.name = bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_5;
        cmd1.kp = 10; cmd1.kd = 1; cmd1.spd = 0; cmd1.tor = 0;
        cmd1.pos = mot(0, 0) + 1.047 - 0.374;

        cmd2.name = bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_6;
        cmd2.kp = 10; cmd2.kd = 1; cmd2.spd = 0; cmd2.tor = 0;
        cmd2.pos = -(mot(1, 0) + 1.047 - 0.374);

        bodyctrl_msgs::MotorCtrl cmd3, cmd4;
        cmd3.name = bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_5;
        cmd3.kp = 10; cmd3.kd = 1; cmd3.spd = 0; cmd3.tor = 0;
        cmd3.pos = -(mot2(1, 0) + 1.047 - 0.374);

        cmd4.name = bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_6;
        cmd4.kp = 10; cmd4.kd = 1; cmd4.spd = 0; cmd4.tor = 0;
        cmd4.pos = (mot2(0, 0) + 1.047 - 0.374);

        msg.cmds.push_back(cmd1);
        msg.cmds.push_back(cmd2);
        msg.cmds.push_back(cmd3);
        msg.cmds.push_back(cmd4); 
        pub_motor_cmd_.publish(msg);
    }

    void FullReset() {
       static int reset_step = 0;
       const int total_steps = 10;
       static Eigen::VectorXd start_pos(12);
       if (!initialized_reset_) {
           for (int i = 0; i < 12; i++)
               start_pos(i) = latest_motor_pos_[motor_name[i]];
           reset_step = 0;
           initialized_reset_ = true;
       }

       if (reset_step > total_steps) {
           test_state_ = TestState::IDLE;
           initialized_reset_ = false;
           return;
       }

       Eigen::VectorXd target_pos(12);
       auto mot_l = left_ankle_.InverseKinematics(init_pos_(4), init_pos_(5));
       auto mot_r = right_ankle_.InverseKinematics(init_pos_(10), init_pos_(11));

       for (int i = 0; i < 12; i++) {
           if (i == 4)
               target_pos(i) = mot_l(0) + 1.047 - 0.374;
           else if (i == 5)
               target_pos(i) = -(mot_l(1) + 1.047 - 0.374);
           else if (i == 10)
               target_pos(i) = -(mot_r(1) + 1.047 - 0.374);
           else if (i == 11)
               target_pos(i) = mot_r(0) + 1.047 - 0.374;
           else
               target_pos(i) = (init_pos_(i) - zero_offset_(i)) * motor_dir_(i);
       }

       float alpha = static_cast<float>(reset_step) / total_steps;
       Eigen::VectorXd interp = (1 - alpha) * start_pos + alpha * target_pos;

       bodyctrl_msgs::CmdMotorCtrl msg;
       for (int i = 0; i < 12; i++) {
           bodyctrl_msgs::MotorCtrl cmd;
           cmd.name = motor_name[i];
           if (i==4 ||i==5 ||i==10||i==11){
               cmd.kp = 20; cmd.kd = 1; cmd.spd = 0; cmd.tor = 0;
           }

           else{
           cmd.kp = 200; cmd.kd = 1.5; cmd.spd = 0; cmd.tor = 0;}
           cmd.pos = interp(i);
           msg.cmds.push_back(cmd);
       }

       msg.header.stamp = ros::Time::now();
       pub_motor_cmd_.publish(msg);
       reset_step++;
   }
    void StopAll() {
        bodyctrl_msgs::CmdMotorCtrl msg;
        for (int i = 0; i < 12; i++) {
            bodyctrl_msgs::MotorCtrl cmd;
            cmd.name = motor_name[i];
            cmd.kp = 0.0; cmd.kd = 0.0; cmd.pos = 0.0; cmd.spd = 0.0; cmd.tor = 0.0;
            msg.cmds.push_back(cmd);
        }
        msg.header.stamp = ros::Time::now();
        pub_motor_cmd_.publish(msg);
    }

    // ROS
    ros::Publisher pub_motor_cmd_;
    ros::Subscriber sub_motor_state_;
    ros::Subscriber sub_joy_;

    // State
    TestState test_state_ = TestState::IDLE;
    xbox_map_t xbox_map_;
    float pi_;
    int motor_num_;
    Eigen::VectorXd motor_dir_, zero_cnt_, zero_offset_, init_pos_;
    std::map<int, int> motor_id, motor_name;
    std::unordered_map<int, float> latest_motor_pos_;
    ParallelAnkle<float>::AnkleParameters left_params_;
    ParallelAnkle<float>::AnkleParameters right_params_;
    ParallelAnkle<float> left_ankle_;
    ParallelAnkle<float> right_ankle_;
    bool initialized_reset_ = false;
    
};

} // namespace rl_control_new

PLUGINLIB_EXPORT_CLASS(rl_control_new::AnkleTestNodelet, nodelet::Nodelet)
