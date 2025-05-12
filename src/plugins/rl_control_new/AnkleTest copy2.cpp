/**
 * @file AnkleTestNodelet.cpp
 * @brief 测试并联踝关节，全身关节回零模拟：A键达到-30，B键回到0，C键全身回零，D键关掉PD
 * @author Siyuan Wang
 * @date 2025-04-30
 */
/*
ToDo:
*/
#include "util/LockFreeQueue.h"
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float64.h>
#include <stdio.h>
#include <bodyctrl_msgs/Imu.h>
#include <bodyctrl_msgs/CmdSetMotorSpeed.h>
#include <thread>
#include <cmath>
#include <iostream>
#include <time.h>
#include <fstream>
// #include <fast_ros/fast_ros.h>
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
#include "ovinf/ovinf_factory.hpp"
#include "ovinf/ovinf_humanoid.h"  // 或其他定义 PolicyInput 的头文件
#include <geometry_msgs/Twist.h>
namespace rl_control_new {

class AnkleTestNodelet : public nodelet::Nodelet {
public:
AnkleTestNodelet() : left_params(), right_params(), left_ankle(left_params, 1e-6f), right_ankle(right_params, 1e-6f) {}
~AnkleTestNodelet() noexcept override = default;
private:
    enum class TestState {
        IDLE,
        MOVE_TO_TARGET,
        INFER, 
        FULL_RESET,
        STOP
    };

    struct xbox_map_t {
        float a = 0, b = 0, c = 0, d = 0;
        float e = 0, f = 0, g = 0, h = 0;
        float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    };

    void onInit() override {
        auto& nh = getPrivateNodeHandle();

        InitParams();
       
        pub_motor_cmd = nh.advertise<bodyctrl_msgs::CmdMotorCtrl>("/BodyControl/motor_ctrl", 1000);
        sub_motor_state = nh.subscribe("/BodyControl/motor_state", 1000, &AnkleTestNodelet::OnMotorState, this);
        sub_joy = nh.subscribe("/sbus_data", 1000, &AnkleTestNodelet::OnJoyReceived, this);
        subImuXsens = nh.subscribe("/BodyControl/imu", 1000, &AnkleTestNodelet::OnXsensImuStatusMsg, this);
        subCmdVel = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 1000, &AnkleTestNodelet::OnCmdVelMsg, this);

        YAML::Node config = YAML::LoadFile("/home/wsy/Library/ovinf/config/humanoid.yaml");
        policy = std::make_shared<ovinf::HumanoidPolicy>(config["inference"]);


        // infer_timer = nh.createTimer(
        //     ros::Duration(0.01),  // 10ms
        //     &AnkleTestNodelet::InferTimerCallback, 
        //     this, 
        //     false,  // 不自动启动
        //     false   // 需要后续调用 start()
        // );
        std::thread([this]() {
            ros::Rate rate(100);  // 100 Hz
            while (ros::ok()) {
                RunFSM();  // 调用你的状态机主逻辑
                rate.sleep();
            }
        }).detach();


        NODELET_INFO("AnkleTestNodelet initialized.");
    }

    void InitParams() {
        pi = M_PI;
        motor_num = 20;
        motor_dir = Eigen::VectorXd::Ones(motor_num);
        motor_dir << 1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
                      1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
                      1.0, -1.0, -1.0, -1.0,
                     -1.0, -1.0, -1.0, 1.0;
        zero_cnt = Eigen::VectorXd::Zero(motor_num);
        zero_offset = Eigen::VectorXd::Ones(motor_num);
        zero_offset << 0.0, 0.0, -pi/3.0, 2.0*pi/3.0, -0.673, -0.673,
                       0.0, 0.0, -pi/3.0, 2.0*pi/3.0, -0.673, -0.673,
                       0.0, 0.2618, 0.0, 0.0,
                       0.0, -0.2618, 0.0, 0.0;  //-1.047
        init_pos = Eigen::VectorXd::Zero(motor_num);
        init_pos.segment(0, 12) << 0.0, 0.0, -0.3, 0.6, -0.30, 0.0,
                                                  0.0, 0.0, -0.3, 0.6, -0.30, 0.0;
        // ankle parameters
        this->left_params.l_bar1 = 0.06;
        this->left_params.l_rod1 = 0.215;
        this->left_params.r_a1 = {0.0, 0.044, 0.215};
        this->left_params.r_b1_0 = {-0.056, 0.044, 0.237};
        this->left_params.r_c1_0 = {-0.056, 0.044, 0.022};
        this->left_params.l_bar2 = 0.06;
        this->left_params.l_rod2 = 0.14;
        this->left_params.r_a2 = {0.0, -0.043, 0.141};
        this->left_params.r_b2_0 = {-0.056, -0.043, 0.163};
        this->left_params.r_c2_0 = {-0.056, -0.043, 0.023};

        this->right_params.l_bar1 = 0.06;
        this->right_params.l_rod1 = 0.14;
        this->right_params.r_a1 = {0.0, 0.043, 0.141};
        this->right_params.r_b1_0 = {-0.056, 0.043, 0.163};
        this->right_params.r_c1_0 = {-0.056, 0.043, 0.023};
        this->right_params.l_bar2 = 0.06;
        this->right_params.l_rod2 = 0.215;
        this->right_params.r_a2 = {0.0, -0.044, 0.215};
        this->right_params.r_b2_0 = {-0.056, -0.044, 0.237};
        this->right_params.r_c2_0 = {-0.056, -0.044, 0.022};

        this->right_ankle = ParallelAnkle<float>(this->right_params, 1e-6f);
        
        this->left_ankle = ParallelAnkle<float>(left_params, 1e-6f);
        // proj_gravity = Eigen::Vector3f::Zero();



        YAML::Node config =
        YAML::LoadFile("/home/wsy/Library/ovinf/config/humanoid.yaml");
        auto policy = ovinf::PolicyFactory::CreatePolicy(config["inference"]);

        kp = Eigen::VectorXd::Ones(motor_num) * 50.0;
        kd = Eigen::VectorXd::Ones(motor_num) * 5.0;
        // 在 AnkleTestNodelet 类中添加

        // 记得修改
        kp << 200.0, 200.0, 200.0, 200.0, 15.0, 15.0,
            200.0, 200.0, 200.0, 200.0, 15.0, 15.0;
   
        kd << 2.0, 2.0, 2.0, 2.0, 1.25, 1.25,
            2.0, 2.0, 2.0, 2.0, 1.25, 1.25;

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

        for (int i = 0; i < motor_num; i++) {
            motor_id.insert({motor_name[i], i});
        }


        xsense_data = Eigen::VectorXd::Zero(13);
        data = Eigen::VectorXd::Zero(350);
        Q_a = Eigen::VectorXd::Zero(motor_num);
        Qdot_a = Eigen::VectorXd::Zero(motor_num);
        Tor_a = Eigen::VectorXd::Zero(motor_num);
        Q_d = Eigen::VectorXd::Zero(motor_num);
        Qdot_d = Eigen::VectorXd::Zero(motor_num);
        Tor_d = Eigen::VectorXd::Zero(motor_num);
        q_a = Eigen::VectorXd::Zero(motor_num);
        qdot_a = Eigen::VectorXd::Zero(motor_num);
        tor_a = Eigen::VectorXd::Zero(motor_num);
        q_d = Eigen::VectorXd::Zero(motor_num);
        qdot_d = Eigen::VectorXd::Zero(motor_num);
        tor_d = Eigen::VectorXd::Zero(motor_num);
        euler_rpy =Eigen::VectorXf::Zero(3);
        ct_scale = Eigen::VectorXd::Ones(motor_num);

        ct_scale << 2.5, 2.1, 2.5, 2.5, 1.4, 1.4, 
        2.5, 2.1, 2.5, 2.5, 1.4, 1.4, 
        1.4, 1.4, 1.4, 1.4, 
        1.4, 1.4, 1.4, 1.4;

    }


// 获取电机状态 + 数据处理
    void OnMotorState(const bodyctrl_msgs::MotorStatusMsg::ConstPtr& msg) {
        float q_left_pitch = 0.0f, q_left_roll = 0.0f;
        float q_right_pitch = 0.0f, q_right_roll = 0.0f;
        int id = 0;
        for (const auto& motor : msg->status) {
             id = motor_id[motor.name];
             Q_a(id) = motor.pos;
             Qdot_a(id) = motor.speed;
             Tor_a(id) = motor.current*ct_scale(id);
        }
 
        for (int i = 0; i < motor_num; i++) {
             q_a(i) = Q_a(i) * motor_dir(i) + zero_offset(i);
             qdot_a(i) = Qdot_a(i) * motor_dir(i);
             tor_a(i) = Tor_a(i) * motor_dir(i);
        }

        auto fk_left = left_ankle.ForwardKinematics(q_a(4), q_a(5));
        auto fk_right = right_ankle.ForwardKinematics(q_a(11), q_a(10)); //顺序很重要
        q_a(4) = fk_left(0);
        q_a(5) = fk_left(1);
        q_a(10) = fk_right(0);
        q_a(11) = fk_right(1);

        ROS_INFO_STREAM_THROTTLE(1.5, "[Left Ankle FK] pitch = " << fk_left(0)  * 180.0 / M_PI << ", roll = " << fk_left(1)  * 180.0 / M_PI);
        ROS_INFO_STREAM_THROTTLE(1.5, "[Right Ankle FK] pitch = " << fk_right(0)  * 180.0 / M_PI << ", roll = " << fk_right(1)  * 180.0 / M_PI);
     }


     void RunFSM() {
        // joystick 命令
        input.command = last_cmd_vel.cast<float>();
    
        // 关节状态（由 OnMotorState 设置后用于推理）
        input.joint_pos = q_a.cast<float>();
        input.joint_vel = qdot_a.cast<float>();
    
        // 处理最新 IMU 数据
        while (!queueImuXsens.empty()) {
            auto imu_msg = queueImuXsens.pop();
            xsense_data(0) = imu_msg->euler.yaw;
            xsense_data(1) = imu_msg->euler.pitch;
            xsense_data(2) = imu_msg->euler.roll;
            xsense_data(3) = imu_msg->angular_velocity.x;
            xsense_data(4) = imu_msg->angular_velocity.y;
            xsense_data(5) = imu_msg->angular_velocity.z;
            xsense_data(6) = imu_msg->linear_acceleration.x;
            xsense_data(7) = imu_msg->linear_acceleration.y;
            xsense_data(8) = imu_msg->linear_acceleration.z;
            last_imu_ang_vel << imu_msg->angular_velocity.x,
                                 imu_msg->angular_velocity.y,
                                 imu_msg->angular_velocity.z;
        }
    
        input.ang_vel = last_imu_ang_vel.cast<float>();
    
        // 重力投影计算
        euler_rpy << xsense_data(0), xsense_data(1), xsense_data(2);
        Eigen::Matrix3f Rwb(
            Eigen::AngleAxisf(euler_rpy[0], Eigen::Vector3f::UnitZ()) *
            Eigen::AngleAxisf(euler_rpy[1], Eigen::Vector3f::UnitY()) *
            Eigen::AngleAxisf(euler_rpy[2], Eigen::Vector3f::UnitX()));
        input.proj_gravity = Rwb.transpose() * Eigen::Vector3f{0.0, 0.0, -1.0};
    
        // 状态机行为控制
        switch (state) {
            case TestState::INFER: {
                static bool entered = false;
                if (!entered) {
                    entered = true;
                    warmup_counter = 0;
                    is_first_infer = true;
                    NODELET_INFO("[FSM] Entering INFER state, infer loop running");
                }
                StepInference();
            
                break;
            }
            case TestState::FULL_RESET: {
                NODELET_INFO_ONCE("[FSM] Entering FULL_RESET state");
                FullReset();
                break;
            }
            case TestState::STOP: {
                NODELET_INFO_ONCE("[FSM] Entering STOP state");
                StopAll();
                break;
            }
            default:
                break;
        }
    }
    
    void StepInference() {
        if (is_first_infer && warmup_counter < warmup_iters) {
            policy->WarmUp(input, 1);
            warmup_counter++;
            return;
        }
    
        is_first_infer = false;
    
        policy->InferUnsync(input);
        auto result = policy->GetResult();
        if (!result.has_value()) return;
    
        Eigen::VectorXf q_d = Eigen::VectorXf::Zero(motor_num);
        q_d.segment(0, 12) = result.value();
    
        Eigen::VectorXf Q_d = Eigen::VectorXf::Zero(motor_num);
        Q_d.segment(0, 12) = q_d.segment(0, 12);
    
        // 脚踝 IK 解算
        auto mot_l = left_ankle.InverseKinematics(q_d(4), q_d(5));                
        auto mot_r = right_ankle.InverseKinematics(q_d(10), q_d(11));
        Q_d(4)  = mot_l(0);
        Q_d(5)  = mot_l(1);
        Q_d(10) = mot_r(1);
        Q_d(11) = mot_r(0);
    
        bodyctrl_msgs::CmdMotorCtrl msg_out;
        msg_out.header.stamp = ros::Time::now();
        for (int i = 0; i < 12; i++) {
            bodyctrl_msgs::MotorCtrl cmd;
            cmd.name = motor_name[i];
            cmd.kp = kp(i);
            cmd.kd = kd(i);
            cmd.pos = (Q_d(i) - zero_offset(i)) * motor_dir(i);
            cmd.spd = 0;
            cmd.tor = 0;
            msg_out.cmds.push_back(cmd);
        }
    
        pub_motor_cmd.publish(msg_out);
    }
    
     void OnMotorStatusMsg(const bodyctrl_msgs::MotorStatusMsg::ConstPtr &msg) {
        auto wrapper = msg;
        queueMotorState.push(wrapper);
      }
    
    //   void OnImuStatusMsg(const bodyctrl_msgs::Imu::ConstPtr &msg) {
    //     auto wrapper = msg;
    //     queueImuRm.push(wrapper);
    //   }
    
      void OnXsensImuStatusMsg(const bodyctrl_msgs::Imu::ConstPtr& msg)
      {
        auto wrapper = msg;
        queueImuXsens.push(wrapper);
      }
    
      void xbox_mapread(const sensor_msgs::Joy::ConstPtr &msg)
      {
        auto wrapper = msg;
        queueJoyCmd.push(wrapper);
      }
    
      void OnCmdVelMsg(const geometry_msgs::Twist::ConstPtr& msg){
        auto wrapper = msg;
        queueCmdVel.push(wrapper);
      }
    void OnJoyReceived(const sensor_msgs::Joy::ConstPtr& msg) {
        xbox_map.a = msg->axes[8];
        xbox_map.b = msg->axes[9];
        xbox_map.c = msg->axes[10];
        xbox_map.d = msg->axes[11];


        xbox_map.x1 = msg->axes[3];
        xbox_map.x2 = msg->axes[0];
        xbox_map.y1 = msg->axes[2];

        y_speed_command = xbox_map.x1 * -0.25; 
        if (xbox_map.y1 > 0){
            x_speed_command = xbox_map.y1 * 1.5;
        } 
        else{
            x_speed_command = xbox_map.y1 * 0.25;
        }
        
        yaw_speed_command = xbox_map.x2 * -0.8;
        last_cmd_vel << x_speed_command, y_speed_command, yaw_speed_command;


        if (xbox_map.a > 0.5f) {
            state = TestState::MOVE_TO_TARGET;
            ROS_INFO("[FSM] A -> -30 deg");
        }
        if (xbox_map.b > 0.5f) {
            state = TestState::INFER;
            ROS_INFO("[FSM] Inference");
        }
        if (xbox_map.c > 0.5f) {
            state = TestState::FULL_RESET;
            initialized_reset = false;
            ROS_INFO("[FSM] C -> FULL RESET");
        }
        if (xbox_map.d > 0.5f) {
            state = TestState::STOP;
            ROS_INFO("[FSM] D -> STOP");
        }
    }

    void MoveToPitch(float pitch_deg) {
        float pitch = pitch_deg * M_PI / 180.0f;
        auto mot = left_ankle.InverseKinematics(pitch, 0.0f);
        auto mot2 = right_ankle.InverseKinematics(pitch, 0.0f);
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
        pub_motor_cmd.publish(msg);
    }

    void FullReset() {
       static int reset_step = 0;
       const int total_steps = 10;
       static Eigen::VectorXd start_pos(12);
       if (!initialized_reset) {
           for (int i = 0; i < 12; i++)
               start_pos(i) = Q_a(i);
           reset_step = 0;
           initialized_reset = true;
       }

       if (reset_step > total_steps) {
           state = TestState::IDLE;
           initialized_reset = false;
           return;
       }

       Eigen::VectorXd target_pos(12);
       auto mot_l = left_ankle.InverseKinematics(init_pos(4), init_pos(5));
       auto mot_r = right_ankle.InverseKinematics(init_pos(10), init_pos(11));

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
               target_pos(i) = (init_pos(i) - zero_offset(i)) * motor_dir(i);
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
            cmd.kp = 200; cmd.kd = 1.5; cmd.spd = 0; cmd.tor = 0;
           }

           cmd.pos = interp(i);
           msg.cmds.push_back(cmd);
       }

       msg.header.stamp = ros::Time::now();
       pub_motor_cmd.publish(msg);
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
        pub_motor_cmd.publish(msg);
    }

    // ROS
    ros::Publisher pub_motor_cmd;
    ros::Subscriber sub_motor_state;
    ros::Subscriber sub_joy;
    ros::NodeHandle nh;
    ros::Timer infer_timer;
    // State
    TestState state = TestState::IDLE;
    xbox_map_t xbox_map;
    float pi;
    int motor_num;
    bool initialized_reset = false;
    float x_speed_command = 0.;  
    float y_speed_command = 0.;  
    float yaw_speed_command = 0.;
    bool is_first_infer ;
    size_t warmup_counter;
    const size_t warmup_iters = 20;
    std::map<int, int> motor_id, motor_name;
    // std::unordered_map<int, float> latest_motor_pos_;
    ParallelAnkle<float>::AnkleParameters left_params;
    ParallelAnkle<float>::AnkleParameters right_params;
    ParallelAnkle<float> left_ankle;
    ParallelAnkle<float> right_ankle;

    // fast_ros::Subscriber subImu, subImuXsens;
    Eigen::VectorXd motor_dir;
    Eigen::VectorXd  init_pos;
    ros::Subscriber subImu, subImuXsens;
    ros::Subscriber subCmdVel;
    Eigen::VectorXd q_a;
    Eigen::VectorXd qdot_a;
    Eigen::VectorXd tor_a;
    Eigen::VectorXd q_d;
    Eigen::VectorXd qdot_d;
    Eigen::VectorXd tor_d;
    Eigen::VectorXd Q_a;
    Eigen::VectorXd Qdot_a;
    Eigen::VectorXd Tor_a;
    Eigen::VectorXd Q_d;
    Eigen::VectorXd Qdot_d;
    Eigen::VectorXd Tor_d;
    Eigen::VectorXd ct_scale;
    Eigen::VectorXd data;
    Eigen::VectorXd zero_offset;
    Eigen::VectorXd zero_cnt;
    Eigen::VectorXd xsense_data;
    Eigen::VectorXd kp;
    Eigen::VectorXd kd;
    Eigen::Vector3f euler_rpy;
    Eigen::Vector3f last_cmd_vel;
    Eigen::Vector3f last_imu_ang_vel;
    std::shared_ptr<ovinf::HumanoidPolicy> policy;
    ovinf::ProprioceptiveObservation<float> input;
    LockFreeQueue<bodyctrl_msgs::MotorStatusMsg::ConstPtr> queueMotorState;
    LockFreeQueue<bodyctrl_msgs::Imu::ConstPtr> queueImuRm;
    LockFreeQueue<bodyctrl_msgs::Imu::ConstPtr> queueImuXsens;
    LockFreeQueue<sensor_msgs::Joy::ConstPtr> queueJoyCmd;
    LockFreeQueue<geometry_msgs::Twist::ConstPtr> queueCmdVel;


};

} // namespace rl_control_new

PLUGINLIB_EXPORT_CLASS(rl_control_new::AnkleTestNodelet, nodelet::Nodelet)
