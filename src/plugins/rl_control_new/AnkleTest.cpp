/**
 * @file AnkleTestNodelet.cpp
 * @brief 测试并联蹊关节，全身关节回零模拟：A键达到-30，B键回到0，C键全身回零，D键关掉PD
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
#include <geometry_msgs/Twist.h>
#include "ovinf/openvino_lstm_chz.h"
#include "utils/csv_logger.hpp"

namespace rl_control_new
{
	constexpr double pi_ = M_PI;
	class AnkleTestNodelet : public nodelet::Nodelet
	{
	public:
		AnkleTestNodelet() : left_params_(), right_params_(), left_ankle_(left_params_, 1e-6f), right_ankle_(right_params_, 1e-6f) {}
		~AnkleTestNodelet() noexcept override = default;

	private:
		enum class TestState
		{
			IDLE,
			MOVE_TO_TARGET,
			INFER,
			FULL_RESET,
			STOP
		};

		struct xbox_map_t
		{
			float a = 0, b = 0, c = 0, d = 0;
			float e = 0, f = 0, g = 0, h = 0;
			float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
		};

		class RobotStateSensed
		{
		public:
			Eigen::VectorXd q;
			Eigen::VectorXd qdot;
			Eigen::VectorXd tor;
			Eigen::Vector3d imu_ang;
			Eigen::Vector3d imu_avel;
			Eigen::Vector3d imu_acc;
			Eigen::Vector3d proj_gravity;

			RobotStateSensed() : q(Eigen::VectorXd::Zero(20)),
								 qdot(Eigen::VectorXd::Zero(20)),
								 tor(Eigen::VectorXd::Zero(20)),
								 imu_ang(Eigen::Vector3d::Zero()),
								 imu_avel(Eigen::Vector3d::Zero()),
								 imu_acc(Eigen::Vector3d::Zero()),
								 proj_gravity((Eigen::Vector3d() << 0.0, 0.0, -1.0).finished()) {}
		} robot_state_sensed_;

		class RobotStateCmd
		{
		public:
			RobotStateCmd() : q(Eigen::VectorXd::Zero(12)) {}
			Eigen::VectorXd q;
		} robot_state_cmd_;

		void GetSens(const bodyctrl_msgs::MotorStatusMsg::ConstPtr &msg)
		{
			// joints
			for (const auto &motor : msg->status)
			{
				int id = motor_id[motor.name];
				if(id < 12)
				{
					raw_sensed_leg_pos(id) = motor.pos; // chz test 
				}
				robot_state_sensed_.q(id) = motor.pos * motor_dir_(id) + zero_offset_(id);
				robot_state_sensed_.qdot(id) = motor.speed * motor_dir_(id);
				robot_state_sensed_.tor(id) = motor.current;
			}

			// for parellel_ankle
			auto fk_left = left_ankle_.ForwardKinematics(robot_state_sensed_.q(4), robot_state_sensed_.q(5));
			auto fk_right = right_ankle_.ForwardKinematics(robot_state_sensed_.q(11), robot_state_sensed_.q(10)); // 顺序很重要
			robot_state_sensed_.q(4) = fk_left(0);
			robot_state_sensed_.q(5) = fk_left(1);
			robot_state_sensed_.q(10) = fk_right(0);
			robot_state_sensed_.q(11) = fk_right(1);

			// imu
			if (!queueImuXsens.empty())
			{
				while (1)
				{
					auto imu_msg = queueImuXsens.pop();
					if (queueImuXsens.empty())
					{
						robot_state_sensed_.imu_ang << imu_msg->euler.roll, imu_msg->euler.pitch, imu_msg->euler.yaw;
						robot_state_sensed_.imu_avel << imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z;
						robot_state_sensed_.imu_acc << imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z;
						break;
					}
				}
				Eigen::Matrix3d Rwb(
					Eigen::AngleAxisd(robot_state_sensed_.imu_ang(2), Eigen::Vector3d::UnitZ()) *
					Eigen::AngleAxisd(robot_state_sensed_.imu_ang(1), Eigen::Vector3d::UnitY()) *
					Eigen::AngleAxisd(robot_state_sensed_.imu_ang(0), Eigen::Vector3d::UnitX()));
				robot_state_sensed_.proj_gravity =
					(Rwb.transpose() * Eigen::Vector3d{0.0, 0.0, -1.0});
			}
		}

		void SendCmd(const Eigen::VectorXd &kp, const Eigen::VectorXd &kd)
		{
			// for parallel_ankle
			auto q_d = robot_state_cmd_.q;
			auto mot_l = left_ankle_.InverseKinematics(q_d(4), q_d(5));
			auto mot_r = right_ankle_.InverseKinematics(q_d(10), q_d(11));
			q_d(4) = mot_l(0);
			q_d(5) = mot_l(1);
			q_d(10) = mot_r(1);
			q_d(11) = mot_r(0);

			bodyctrl_msgs::CmdMotorCtrl msg_out;
			msg_out.header.stamp = ros::Time::now();
			for (int i = 0; i < 12; i++)
			{
				bodyctrl_msgs::MotorCtrl cmd;
				cmd.name = motor_name[i];
				cmd.kp = kp(i);
				cmd.kd = kd(i);
				raw_desired_leg_pos(i) = (q_d(i) - zero_offset_(i)) * motor_dir_(i); // chz test
				cmd.pos = (q_d(i) - zero_offset_(i)) * motor_dir_(i);
				cmd.spd = 0;
				cmd.tor = 0;
				msg_out.cmds.push_back(cmd);
			}
			pub_motor_cmd_.publish(msg_out);
		}

		void onInit() override
		{
			// notelet handle
			auto &nh = getPrivateNodeHandle();

			// robot motors
			InitParams();
			raw_sensed_leg_pos = Eigen::VectorXd::Zero(12);
			raw_desired_leg_pos = Eigen::VectorXd::Zero(12);

			// ros
			pub_motor_cmd_ = nh.advertise<bodyctrl_msgs::CmdMotorCtrl>("/BodyControl/motor_ctrl", 1000);
			sub_motor_state_ = nh.subscribe("/BodyControl/motor_state", 1000, &AnkleTestNodelet::OnMotorState, this);
			sub_joy_ = nh.subscribe("/sbus_data", 1000, &AnkleTestNodelet::OnJoyReceived, this);
			subImuXsens = nh.subscribe("/BodyControl/imu", 1000, &AnkleTestNodelet::OnXsensImuStatusMsg, this);
			subCmdVel = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 1000, &AnkleTestNodelet::OnCmdVelMsg, this);

			// network
			policy_.Reset();

			default_joint_angles = (Eigen::VectorXd(12) << 0.0, 0.0, -0.2, 0.4, -0.2, 0.0,
									0.0, 0.0, -0.2, 0.4, -0.2, 0.0)
									   .finished();
			// network infer timer
			infer_timer_ = nh.createTimer(
				ros::Duration(0.02), // 20ms
				&AnkleTestNodelet::InferTimerCallback,
				this,
				false, // 不自动启动
				false  // 需要后续调用 start()
			);

			// state
			phase = 0.0;
			observation = Eigen::VectorXf::Zero(49);
			action = Eigen::VectorXf::Zero(13);
			commands = Eigen::VectorXd::Zero(4);
			commands(3) = 1.25;

			// logger
			CreateLog();

			NODELET_INFO("AnkleTestNodelet initialized.");
		}

		void InitParams()
		{
			// motor
			motor_num_ = 20;
			motor_dir_ = (Eigen::VectorXd(motor_num_) << 1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
						  1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
						  1.0, -1.0, -1.0, -1.0,
						  -1.0, -1.0, -1.0, 1.0)
							 .finished();
			zero_offset_ = (Eigen::VectorXd(motor_num_) << 0.0, 0.0, -pi_ / 3.0, 2.0 * pi_ / 3.0, -0.673, -0.673,
							0.0, 0.0, -pi_ / 3.0, 2.0 * pi_ / 3.0, -0.673, -0.673,
							0.0, 0.2618, 0.0, 0.0,
							0.0, -0.2618, 0.0, 0.0)
							   .finished();

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
			for (int i = 0; i < motor_num_; i++)
			{
				motor_id.insert({motor_name[i], i});
			}

			// parellel ankle
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
		}

		void OnMotorState(const bodyctrl_msgs::MotorStatusMsg::ConstPtr &msg)
		{
			// --- START TIMING OnMotorState ---
			static auto last_onmotorstate_time_ = std::chrono::steady_clock::now();
			auto current_time = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::milli> period = current_time - last_onmotorstate_time_;
			onmotorstate_period_ms = period.count();
			last_onmotorstate_time_ = current_time;
			// --- END TIMING OnMotorState ---

			GetSens(msg);
			
			// safety check
			static bool ifsafe = true;
			if(!CheckSens())
			{
				ifsafe = false;
				NODELET_ERROR("[FSM] Safety check failed, stop all motors");
				state_ = TestState::STOP;
			}

			switch (state_)
			{
			case TestState::INFER:
				static bool entered = false;
				if (!entered)
				{
					entered = true;
					infer_timer_.start(); // 启动定时推理
					NODELET_INFO("[FSM] Entering INFER state, timer started");
				}
				break;
			case TestState::FULL_RESET:
				NODELET_INFO_ONCE("[FSM] Entering FULL_RESET state");
				FullReset();
				break;
			case TestState::STOP:
				NODELET_INFO_ONCE("[FSM] Entering STOP state");
				StopAll();
				break;
			default:
				break;
			}
			WriteLog();
		}

		void PreProcess()
		{
			static constexpr struct
			{
				float lin_vel = 2.0f;
				float ang_vel = 0.25f;
				float dof_pos = 1.0f;
				float dof_vel = 0.05f;
			} obs_scales;
			static constexpr float clip_observations = 100.0f;
			static const Eigen::Vector4f commands_scale = (Eigen::Vector4f(4) << obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel, 1.0f).finished();

			// bound the commands
			Eigen::Vector4d commands_bounded = this->commands;
			commands_bounded(0) = std::clamp(commands_bounded(0), -0.5, 1.0);
			commands_bounded(1) = std::clamp(commands_bounded(1), -0.3, 0.3);
			commands_bounded(2) = std::clamp(commands_bounded(2), -0.75, 0.75);
			commands_bounded(3) = std::clamp(commands_bounded(3), 0.9, 1.6);

			observation.segment<3>(0) = robot_state_sensed_.imu_avel.cast<float>() * obs_scales.ang_vel;
			observation.segment<3>(3) = robot_state_sensed_.proj_gravity.cast<float>();
			observation.segment<4>(6) = commands_bounded.cast<float>().cwiseProduct(commands_scale);
			observation.segment<12>(10) = (robot_state_sensed_.q.segment<12>(0) - default_joint_angles).cast<float>() * obs_scales.dof_pos;
			observation.segment<12>(22) = robot_state_sensed_.qdot.segment<12>(0).cast<float>() * obs_scales.dof_vel;
			observation.segment<13>(34) = action.cast<float>();
			observation(47) = sin(2.0 * pi_ * this->phase);
			observation(48) = cos(2.0 * pi_ * this->phase);
			observation = observation.cwiseMin(clip_observations).cwiseMax(-clip_observations);
		}

		void PostProcess()
		{
			static constexpr double action_scale = 0.25;
			static constexpr double clip_actions = 100.0;
			static const Eigen::Vector2d dphase_bound = (Eigen::Vector2d(2) << 0.6, 2.0).finished();
			
			robot_state_cmd_.q = (action.segment<12>(0).cast<double>()).cwiseMin(clip_actions).cwiseMax(-clip_actions) * action_scale + default_joint_angles;

			// test: limit on desired joint angles
			double pos_diff_min[12] = {-0.08, -0.06, -0.11, -0.34, -0.15, -0.15,
									   -0.34, -0.06, -0.11, -0.34, -0.15, -0.15};

			double pos_diff_max[12] = {0.34, 0.06, 0.23, 0.08, 0.15, 0.15,
									   0.08, 0.06, 0.23, 0.08, 0.15, 0.15};

			for (int i = 0; i < 12; i++)
			{
				robot_state_cmd_.q(i) = std::clamp(robot_state_cmd_.q(i), robot_state_sensed_.q(i) + pos_diff_min[i], robot_state_sensed_.q(i) + pos_diff_max[i]);
			}

			double dphase = std::clamp(1.0 * action(12), dphase_bound(0), dphase_bound(1));
			this->phase += dphase * 0.02; // 20ms
			if (this->phase > 1.0)
				this->phase -= 1.0;
		}

		void InferTimerCallback(const ros::TimerEvent &)
		{
			PreProcess();
			policy_.SetObservation(observation);

			auto start_time = std::chrono::steady_clock::now();
			policy_.Infer();
			auto end_time = std::chrono::steady_clock::now();
			this->inference_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
			
			action = policy_.GetAction();

			// print the first 5 inference input/output
			// static int inference_count_ = 0;
			// inference_count_++;
			// if (inference_count_ < 5)
			// {
			// 	std::stringstream ss;
			// 	ss << "[Inference] Input: " << observation.transpose();
			// 	NODELET_INFO_STREAM(ss.str());
			// 	ss.str("");
			// 	ss << "[Inference] Output: " << action.transpose();
			// 	NODELET_INFO_STREAM(ss.str());
			// 	ss.str("");
			// }

			PostProcess();

			const Eigen::VectorXd kp = (Eigen::VectorXd(12) << 150.0, 150.0, 150.0, 200.0, 40.0, 40.0,
										   150.0, 150.0, 150.0, 200.0, 40.0, 40.0)
										   .finished();
			const Eigen::VectorXd kd = (Eigen::VectorXd(12) << 2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
										   2.0, 2.0, 2.0, 4.0, 2.0, 2.0)
										   .finished();
			if(state_ == TestState::INFER)
			{
				SendCmd(kp, kd);
			}
		}

		void OnXsensImuStatusMsg(const bodyctrl_msgs::Imu::ConstPtr &msg)
		{
			auto wrapper = msg;
			queueImuXsens.push(wrapper);
		}

		void OnCmdVelMsg(const geometry_msgs::Twist::ConstPtr &msg)
		{
			auto wrapper = msg;
			queueCmdVel.push(wrapper);
		}

		void OnJoyReceived(const sensor_msgs::Joy::ConstPtr &msg)
		{
			static bool if_first = true;
			static auto last_time = std::chrono::steady_clock::now();
			auto current_time = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::milli> period = current_time - last_time;
			
			xbox_map_.a = msg->axes[8];
			xbox_map_.b = msg->axes[9];
			xbox_map_.c = msg->axes[10];
			xbox_map_.d = msg->axes[11];
			xbox_map_.e = msg->axes[4];
			xbox_map_.f = msg->axes[7];
			xbox_map_.g = msg->axes[5];
			xbox_map_.h = msg->axes[6];
			xbox_map_.x1 = msg->axes[3];
			xbox_map_.x2 = msg->axes[0];
			xbox_map_.y1 = msg->axes[2];
			xbox_map_.y2 = msg->axes[1];
			
			if (xbox_map_.d > 0.5f)
			{
				state_ = TestState::STOP;
				ROS_INFO("[FSM] D -> STOP");
			}

			// Get speed
			this->commands(1) = xbox_map_.x1 * -0.5;
			if (xbox_map_.y1 > 0)
			{
				this->commands(0) = xbox_map_.y1 * 1.0;
			}
			else
			{
				this->commands(0) = xbox_map_.y1 * 0.5;
			}
			this->commands(2) = xbox_map_.x2 * -0.75;
			this->commands(3) = xbox_map_.y2 * 0.35 + 1.25;

			// do not respond to butttons within 500ms
			if(xbox_map_.a > 0.5f || xbox_map_.b > 0.5f || xbox_map_.c > 0.5f || xbox_map_.d > 0.5f)
			{
				last_time = current_time;
			}
			if(period.count() < 500.0)
			{
				return;
			}

			// if (xbox_map_.a > 0.5f)
			// {
			// 	state_ = TestState::MOVE_TO_TARGET;
			// 	ROS_INFO("[FSM] A -> -30 deg");
			// }
			if (xbox_map_.b > 0.5f)
			{
				state_ = TestState::INFER;
				ROS_INFO("[FSM] Inference");
			}
			if (xbox_map_.c > 0.5f)
			{
				state_ = TestState::FULL_RESET;
				initialized_reset_ = false;
				ROS_INFO("[FSM] C -> FULL RESET");
			}

			if(if_first)
			{
				if_first = false;
			}
		}

		void MoveToPitch(float pitch_deg)
		{
			float pitch = pitch_deg * M_PI / 180.0f;
			auto mot = left_ankle_.InverseKinematics(pitch, 0.0f);
			auto mot2 = right_ankle_.InverseKinematics(pitch, 0.0f);
			bodyctrl_msgs::CmdMotorCtrl msg;
			msg.header.stamp = ros::Time::now();

			bodyctrl_msgs::MotorCtrl cmd1, cmd2;
			cmd1.name = bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_5;
			cmd1.kp = 10;
			cmd1.kd = 1;
			cmd1.spd = 0;
			cmd1.tor = 0;
			cmd1.pos = mot(0, 0) + 1.047 - 0.374;

			cmd2.name = bodyctrl_msgs::MotorName::MOTOR_LEG_LEFT_6;
			cmd2.kp = 10;
			cmd2.kd = 1;
			cmd2.spd = 0;
			cmd2.tor = 0;
			cmd2.pos = -(mot(1, 0) + 1.047 - 0.374);

			bodyctrl_msgs::MotorCtrl cmd3, cmd4;
			cmd3.name = bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_5;
			cmd3.kp = 10;
			cmd3.kd = 1;
			cmd3.spd = 0;
			cmd3.tor = 0;
			cmd3.pos = -(mot2(1, 0) + 1.047 - 0.374);

			cmd4.name = bodyctrl_msgs::MotorName::MOTOR_LEG_RIGHT_6;
			cmd4.kp = 10;
			cmd4.kd = 1;
			cmd4.spd = 0;
			cmd4.tor = 0;
			cmd4.pos = (mot2(0, 0) + 1.047 - 0.374);

			msg.cmds.push_back(cmd1);
			msg.cmds.push_back(cmd2);
			msg.cmds.push_back(cmd3);
			msg.cmds.push_back(cmd4);
			// pub_motor_cmd_.publish(msg);
		}

		void FullReset()
		{
			static int reset_step = 0;
			constexpr int total_steps = 3000;
			static Eigen::VectorXd start_pos(12);
			if (!initialized_reset_)
			{
				start_pos = robot_state_sensed_.q.head(12);
				reset_step = 0;
				initialized_reset_ = true;
			}

			if (reset_step > total_steps)
			{
				state_ = TestState::IDLE;
				initialized_reset_ = false;
				ROS_INFO("Full reset completed.");
				return;
			}

			static const Eigen::VectorXd target_pos = (Eigen::VectorXd(12) << 0.0, 0.0, -0.2, 0.4, -0.2, 0.0,
													   0.0, 0.0, -0.2, 0.4, -0.2, 0.0)
														  .finished();

			double alpha = -0.5 * cos(M_PI * reset_step / total_steps) + 0.5;
			robot_state_cmd_.q = (1.0 - alpha) * start_pos + alpha * target_pos;

			Eigen::VectorXd kp = (Eigen::VectorXd(12) << 200.0, 200.0, 200.0, 200.0, 20.0, 20.0,
								  200.0, 200.0, 200.0, 200.0, 20.0, 20.0)
									 .finished();
			Eigen::VectorXd kd = (Eigen::VectorXd(12) << 1.5, 1.5, 1.5, 1.5, 1.0, 1.0,
								  1.5, 1.5, 1.5, 1.5, 1.0, 1.0)
									 .finished();
			SendCmd(kp, kd);
			reset_step++;
		}

		bool CheckSens()
		{
			double max_speed[12] = {5.5, 3.5, 7.0, 8.0, 11.0, 11.0,
									 5.5, 3.5, 7.0, 8.0, 11.0, 11.0};
			bool ifsafenow = true;
			for (int i = 0; i < 12; i++)
			{
				if (robot_state_sensed_.qdot(i) > max_speed[i] || robot_state_sensed_.qdot(i) < -max_speed[i])
				{
					ifsafenow = false;
					NODELET_ERROR("Safety check failed, joint %d exceeds speed limit: %f", i, robot_state_sensed_.qdot(i));
				}
			}
			return ifsafenow;
		}

		void StopAll()
		{
			robot_state_cmd_.q.setZero();
			SendCmd(Eigen::VectorXd::Zero(12), Eigen::VectorXd::Zero(12));
		}

		void CreateLog()
		{
			auto now = std::chrono::system_clock::now();
			std::time_t now_time = std::chrono::system_clock::to_time_t(now);
			std::tm *now_tm = std::localtime(&now_time);
			std::stringstream ss;
			ss << std::put_time(now_tm, "%Y-%m-%d-%H-%M-%S");
			std::string current_time = ss.str();

			std::string log_dir = "/home/ubuntu/chz/logs";
			std::filesystem::path config_file_path(log_dir);
			if (config_file_path.is_relative())
			{
				config_file_path = canonical(config_file_path);
			}

			if (!exists(config_file_path))
			{
				create_directories(config_file_path);
			}

			std::string logger_file =
				config_file_path.string() + "/" + current_time + "_humanoid.csv";

			// Get headers
			std::vector<std::string> headers;

			headers.push_back("imu_avel_x");
			headers.push_back("imu_avel_y");
			headers.push_back("imu_avel_z");
			headers.push_back("imu_grav_x");
			headers.push_back("imu_grav_y");
			headers.push_back("imu_grav_z");
			headers.push_back("command_vel_x");
			headers.push_back("command_vel_y");
			headers.push_back("command_vel_w");
			headers.push_back("command_dphase");
			for (size_t i = 0; i < 12; ++i)
			{
				headers.push_back("actual_pos" + std::to_string(i));
			}
			for (size_t i = 0; i < 12; ++i)
			{
				headers.push_back("actual_vel" + std::to_string(i));
			}
			for (size_t i = 0; i < 12; ++i)
			{
				headers.push_back("desired_pos" + std::to_string(i));
			}
			headers.push_back("phase");
			headers.push_back("onmotorstate_period_ms");
			headers.push_back("inference_time_ms");
			// for (size_t i = 0; i < 12; ++i)
			// {
			// 	headers.push_back("raw_actual_pos" + std::to_string(i));
			// }
			// for (size_t i = 0; i < 12; ++i)
			// {
			// 	headers.push_back("raw_desired_pos" + std::to_string(i));
			// }
			for(size_t i = 0; i < 49; ++i)
			{
				headers.push_back("observation" + std::to_string(i));
			}
			for(size_t i = 0; i < 13; ++i)
			{
				headers.push_back("action" + std::to_string(i));
			}

			csv_logger_ = std::make_shared<ovinf::CsvLogger>(logger_file, headers);
		}

		void WriteLog()
		{
			std::vector<ovinf::CsvLogger::Number> datas;
			// for(size_t i = 0; i < 49; ++i)
			// {
			// 	datas.push_back(observation(i));
			// }
			for(size_t i = 0; i < 3; ++i)
			{
				datas.push_back(robot_state_sensed_.imu_avel(i));
			}
			for(size_t i = 0; i < 3; ++i)
			{
				datas.push_back(robot_state_sensed_.proj_gravity(i));
			}
			for(size_t i = 0; i < 4; ++i)
			{
				datas.push_back(this->commands(i));
			}
			for(size_t i = 0; i < 12; ++i)
			{
				datas.push_back(robot_state_sensed_.q(i));
			}
			for(size_t i = 0; i < 12; ++i)
			{
				datas.push_back(robot_state_sensed_.qdot(i));
			}
			for(size_t i = 0; i < 12; ++i)
			{
				datas.push_back(robot_state_cmd_.q(i));
			}
			datas.push_back(this->phase);
			datas.push_back(this->onmotorstate_period_ms);
			datas.push_back(this->inference_time_ms);
			// for(size_t i = 0; i < 12; ++i)
			// {
			// 	datas.push_back(raw_sensed_leg_pos(i));
			// }
			// for(size_t i = 0; i < 12; ++i)
			// {
			// 	datas.push_back(raw_desired_leg_pos(i));
			// }
			for(size_t i = 0; i < 49; ++i)
			{
				datas.push_back(observation(i));
			}
			for(size_t i = 0; i < 13; ++i)
			{
				datas.push_back(action(i));
			}
			csv_logger_->Write(datas);
		}

		// ROS
		ros::Publisher pub_motor_cmd_;
		ros::Subscriber sub_motor_state_, sub_joy_, subImuXsens, subCmdVel;
		ros::NodeHandle nh;
		ros::Timer infer_timer_;
		LockFreeQueue<bodyctrl_msgs::MotorStatusMsg::ConstPtr> queueMotorState;
		LockFreeQueue<bodyctrl_msgs::Imu::ConstPtr> queueImuRm;
		LockFreeQueue<bodyctrl_msgs::Imu::ConstPtr> queueImuXsens;
		LockFreeQueue<sensor_msgs::Joy::ConstPtr> queueJoyCmd;
		LockFreeQueue<geometry_msgs::Twist::ConstPtr> queueCmdVel;
		// State
		TestState state_ = TestState::IDLE;
		// JoyCon
		xbox_map_t xbox_map_;
		// Motor
		int motor_num_;
		Eigen::VectorXd motor_dir_, zero_offset_;
		std::map<int, int> motor_id, motor_name;
		Eigen::VectorXd raw_sensed_leg_pos, raw_desired_leg_pos; // for testing
		// Parallel Ankle
		ParallelAnkle<float>::AnkleParameters left_params_;
		ParallelAnkle<float>::AnkleParameters right_params_;
		ParallelAnkle<float> left_ankle_;
		ParallelAnkle<float> right_ankle_;
		bool initialized_reset_ = false;
		// Network
		Eigen::VectorXd default_joint_angles;
		Eigen::VectorXf observation, action;
		Eigen::VectorXd commands;
		chz::LSTMCHZPolicy policy_; // chz: a simple LSTM policy api
		double phase;				// chz: phase obs
		// logger items
		double onmotorstate_period_ms = 0.0;
		double inference_time_ms = 0.0;
		ovinf::CsvLogger::Ptr csv_logger_;
	};

} // namespace rl_control_new

PLUGINLIB_EXPORT_CLASS(rl_control_new::AnkleTestNodelet, nodelet::Nodelet)
