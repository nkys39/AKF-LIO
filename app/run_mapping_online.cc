#include <gflags/gflags.h>
#include <unistd.h>
#include <csignal>
#include <vector>
#include <string>

#include "laser_mapping.h"

/// run the lidar mapping in online mode

DEFINE_string(traj_log_file, (std::string(std::string(ROOT_DIR) + "Log/" + "traj.txt")), "path to traj log file");

std::shared_ptr<akf_lio::LaserMapping> laser_mapping;

void SigHandle(int sig)
{
    akf_lio::options::FLAG_EXIT = true;
    RCLCPP_WARN(rclcpp::get_logger("akf_lio"), "catch sig %d", sig);
}

int main(int argc, char **argv)
{
    // Initialize ROS2 first
    rclcpp::init(argc, argv);

    // Remove ROS2-specific arguments from argc/argv for gflags
    std::vector<std::string> non_ros_args = rclcpp::remove_ros_arguments(argc, argv);

    // Convert to argc/argv format for gflags
    std::vector<char*> filtered_argv;
    for (auto& arg : non_ros_args) {
        filtered_argv.push_back(arg.data());
    }
    int filtered_argc = static_cast<int>(filtered_argv.size());
    char** filtered_argv_ptr = filtered_argv.data();

    // Now parse remaining arguments with gflags
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(filtered_argv_ptr[0]);
    google::ParseCommandLineFlags(&filtered_argc, &filtered_argv_ptr, false);

    auto node = std::make_shared<rclcpp::Node>("laserMapping");

    laser_mapping = std::make_shared<akf_lio::LaserMapping>();
    laser_mapping->InitROS(node);

    signal(SIGINT, SigHandle);
    rclcpp::Rate rate(5000);

    // online, almost same with offline, just receive the messages from ros
    while (rclcpp::ok())
    {
        if (akf_lio::options::FLAG_EXIT)
        {
            break;
        }
        rclcpp::spin_some(node);
        laser_mapping->Run();
        rate.sleep();
    }

    akf_lio::Timer::PrintAll();
    LOG(INFO) << "save trajectory to: " << FLAGS_traj_log_file;
    laser_mapping->Savetrajectory(FLAGS_traj_log_file);

    rclcpp::shutdown();
    return 0;
}
