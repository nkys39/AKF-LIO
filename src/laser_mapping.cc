#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <execution>
#include <fstream>
#include <chrono>

#include "laser_mapping.h"
#include "utils.h"

#include <tbb/parallel_for.h>

#include <opencv2/opencv.hpp>

namespace akf_lio
{

    bool LaserMapping::InitROS(rclcpp::Node::SharedPtr node)
    {
        node_ = node;
        LoadParams();
        SubAndPubToROS();

        // localmap init (after LoadParams)
        ivox_ = std::make_shared<IVoxType>(ivox_options_);

        // esekf init
        std::vector<double> epsi(23, 0.001);
        kf_.init_dyn_share(
            get_f, df_dx, df_dw,
            [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
            { ObsModel(s, ekfom_data); },
            options::NUM_MAX_ITERATIONS, epsi.data());

        return true;
    }

    bool LaserMapping::LoadParams()
    {
        // get params from param server
        int lidar_type, ivox_nearby_type;
        double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
        common::V3D lidar_T_wrt_IMU;
        common::M3D lidar_R_wrt_IMU;

        // Declare and get parameters
        p_imu_->alpha = node_->declare_parameter<double>("akf_update_alpha", 1.0);

        preprocess_->Blind() = node_->declare_parameter<double>("preprocess.blind", 0.01);
        preprocess_->TimeScale() = static_cast<float>(node_->declare_parameter<double>("preprocess.time_scale", 1e-3));
        lidar_type = node_->declare_parameter<int>("preprocess.lidar_type", 1);
        preprocess_->NumScans() = node_->declare_parameter<int>("preprocess.scan_line", 16);

        options::LIDAR_COV = static_cast<float>(node_->declare_parameter<double>("mapping.lidar_cov", 0.001));
        gyr_cov = node_->declare_parameter<double>("mapping.gyr_cov", 0.1);
        acc_cov = node_->declare_parameter<double>("mapping.acc_cov", 0.1);
        b_gyr_cov = node_->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        b_acc_cov = node_->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        extrinT_ = node_->declare_parameter<std::vector<double>>("mapping.extrinsic_T", std::vector<double>());
        extrinR_ = node_->declare_parameter<std::vector<double>>("mapping.extrinsic_R", std::vector<double>());

        path_pub_en_ = node_->declare_parameter<bool>("publish.path_publish_en", true);
        scan_reg_pub_en_ = node_->declare_parameter<bool>("publish.scan_reg_pub_en", false);
        dense_pub_en_ = node_->declare_parameter<bool>("publish.dense_publish_en", false);
        map_pub_en_ = node_->declare_parameter<bool>("publish.map_publish_en", false);
        gaussian_publish_en_ = node_->declare_parameter<bool>("publish.gaussian_publish_en", false);
        gaussian_pub_dis_ = node_->declare_parameter<double>("publish.gaussian_pub_dis", 10.0);
        gaussian_pub_min_cnt_ = node_->declare_parameter<double>("publish.gaussian_pub_min_cnt", 5.0);

        adap_voxel_size_en_ = node_->declare_parameter<bool>("adap_voxel_size_en", true);
        target_point_size_ = node_->declare_parameter<int>("target_point_size", 2000);
        preprocess_->PointFilterNum() = node_->declare_parameter<int>("point_filter_num", 2);
        options::NUM_MAX_ITERATIONS = node_->declare_parameter<int>("max_iteration", 4);
        init_uncertainty_ = node_->declare_parameter<double>("init_uncertainty", 0.01);
        t_ratio_b_ = node_->declare_parameter<double>("t_ratio_b", 0.0);
        ivox_options_.resolution_ = node_->declare_parameter<double>("ivox_grid_resolution", 0.5);
        ivox_nearby_type = node_->declare_parameter<int>("ivox_nearby_type", 26);
        options::T_MAL = static_cast<float>(node_->declare_parameter<double>("t_mal", 11.28));
        t_stop_pseudo_merge_ = node_->declare_parameter<double>("t_stop_pseudo_merge", 11.28);
        options::TIME_TO_DELETE_LOCAL_MAP = node_->declare_parameter<double>("time_to_delete_local_map", 1000.0);

        runtime_pos_log_ = node_->declare_parameter<bool>("runtime_pos_log_enable", true);

        LOG(INFO) << "lidar_type " << lidar_type;
        if (lidar_type == 1)
        {
            preprocess_->SetLidarType(LidarType::AVIA);
            LOG(INFO) << "Using AVIA Lidar";
        }
        else if (lidar_type == 2)
        {
            preprocess_->SetLidarType(LidarType::VELO32);
            LOG(INFO) << "Using Velodyne 32 Lidar";
        }
        else if (lidar_type == 3)
        {
            preprocess_->SetLidarType(LidarType::OUST64);
            LOG(INFO) << "Using OUST 64 Lidar";
        }
        else
        {
            LOG(WARNING) << "unknown lidar_type";
            return false;
        }

        if (ivox_nearby_type == 0)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
        }
        else if (ivox_nearby_type == 6)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
        }
        else if (ivox_nearby_type == 18)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        }
        else if (ivox_nearby_type == 26)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
        }
        else
        {
            LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        }

        path_.header.stamp = node_->get_clock()->now();
        path_.header.frame_id = "camera_init";
        gt_path_.header.stamp = node_->get_clock()->now();
        gt_path_.header.frame_id = "camera_init";

        fout_pre.open((std::string(std::string(ROOT_DIR) + "Log/" + "mat_pre.txt")), std::ios::out);
        fout_out.open((std::string(std::string(ROOT_DIR) + "Log/" + "mat_out.txt")), std::ios::out);
        if (fout_pre && fout_out)
            std::cout << "~~~~" << ROOT_DIR << " file opened" << std::endl;
        else
            std::cout << "~~~~" << ROOT_DIR << " doesn't exist" << std::endl;

        lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
        lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

        p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
        p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
        p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        return true;
    }

    void LaserMapping::SubAndPubToROS()
    {
        // ROS subscribe initialization
        std::string lidar_topic, imu_topic;
        lidar_topic = node_->declare_parameter<std::string>("common.lid_topic", "/livox/lidar");
        imu_topic = node_->declare_parameter<std::string>("common.imu_topic", "/livox/imu");

        // QoS settings for sensor data
        rclcpp::QoS qos_sensor(rclcpp::KeepLast(200000));
        qos_sensor.best_effort();

        if (preprocess_->GetLidarType() == LidarType::AVIA)
        {
            sub_livox_pcl_ = node_->create_subscription<livox_interfaces::msg::CustomMsg>(
                lidar_topic, qos_sensor,
                [this](const livox_interfaces::msg::CustomMsg::SharedPtr msg)
                { LivoxPCLCallBack(msg); });
        }
        else
        {
            sub_pcl_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
                lidar_topic, qos_sensor,
                [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg)
                { StandardPCLCallBack(msg); });
        }

        sub_imu_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, qos_sensor,
            [this](const sensor_msgs::msg::Imu::SharedPtr msg)
            { IMUCallBack(msg); });

        // ROS publisher init
        path_.header.stamp = node_->get_clock()->now();
        path_.header.frame_id = "camera_init";

        rclcpp::QoS qos_pub(rclcpp::KeepLast(100000));
        pub_laser_cloud_world_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_dense_world", qos_pub);
        pub_laser_cloud_reg_world_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_reg_world", qos_pub);
        pub_odom_aft_mapped_ = node_->create_publisher<nav_msgs::msg::Odometry>("/Odometry", qos_pub);
        pub_path_ = node_->create_publisher<nav_msgs::msg::Path>("/path", qos_pub);
        gt_pub_path_ = node_->create_publisher<nav_msgs::msg::Path>("/gt_path", qos_pub);
        point_cov_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/scan_gaussian_reg", rclcpp::QoS(10000));
        pub_map_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", qos_pub);
        map_cov_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/map_gaussian", rclcpp::QoS(10000));

        // TF broadcaster
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
    }

    LaserMapping::LaserMapping()
    {
        preprocess_.reset(new PointCloudPreprocess());
        p_imu_.reset(new ImuProcess());
    }

    PointType LaserMapping::Merge2(const PointType &pt1, const PointType &pt2)
    {
        PointType pt3 = pt2;
        common::V3D p1(pt1.x, pt1.y, pt1.z), p2(pt2.x, pt2.y, pt2.z), p3;
        double ratio_1 = (double)pt1.pt_num / (double)(pt1.pt_num + pt2.pt_num);
        double ratio_2 = (double)pt2.pt_num / (double)(pt1.pt_num + pt2.pt_num);

        p3 = p1 * ratio_1 + p2 * ratio_2;
        pt3.x = p3(0);
        pt3.y = p3(1);
        pt3.z = p3(2);
        pt3.intensity = (pt1.intensity * ratio_1 + pt2.intensity * ratio_2);
        pt3.cov =
            ratio_1 * (pt1.cov + p1 * p1.transpose()) + ratio_2 * (pt2.cov + p2 * p2.transpose()) - p3 * p3.transpose();

        pt3.pt_num = pt1.pt_num + pt2.pt_num;
        pt3.time = pt1.time > pt2.time ? pt1.time : pt2.time;

        if (pt3.pt_num > options::MAX_FEA_NUM)
            pt3.pt_num = options::MAX_FEA_NUM;

        int use_sum = pt1.use_num + pt2.use_num;
        if (use_sum != 0)
        {
            double ratio_3 = (double)pt1.use_num / (double)(use_sum);
            double ratio_4 = (double)pt2.use_num / (double)(use_sum);
            common::V3D p4 = ratio_3 * p1 + ratio_4 * p2;
            pt3.uncertainty = (ratio_3 * pt1.uncertainty + ratio_4 * pt2.uncertainty);
            pt3.use_num = use_sum;
            if (pt3.use_num > options::MAX_FEA_NUM)
                pt3.use_num = options::MAX_FEA_NUM;
        }

        return pt3;
    }

    void LaserMapping::Run()
    {
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        
        if (!SyncPackages())
        {
            return;
        }

        if (first_lidar_time_ < 1e-6)
            first_lidar_time_ = measures_.lidar_bag_time_;
        if (runtime_pos_log_)
            LOG(INFO) << "measures_.lidar_bag_time_ - first_lidar_time_ " << measures_.lidar_bag_time_ - first_lidar_time_;
        common::V3D euler_cur, ext_euler, print_status;
        ext_euler.setZero();
        print_status.setZero();

        /// IMU process, kf prediction, undistortion
        p_imu_->Process(measures_, kf_, scan_undistort_);
        pred_cov_ = kf_.get_P().block<6, 6>(0, 0);
        if (scan_undistort_->empty() || (scan_undistort_ == nullptr))
        {
            PublishOdometry(pub_odom_aft_mapped_);
            if (path_pub_en_)
                PublishPath(pub_path_);
            LOG(WARNING) << "No point, skip this scan!";
            return;
        }

        flg_EKF_inited_ = (measures_.lidar_bag_time_ - first_lidar_time_) >= options::INIT_TIME;

        /// downsample
        Timer::Evaluate([&, this]()
                        { VoxelGridDownsample(scan_undistort_, scan_down_reg_); },
                        "Downsample PointCloud");

        if (flg_first_scan_)
        {
            PointVector points_to_add;

            state_point_ = kf_.get_x();
            state_cov_.setZero();
            int cur_pts = cur_all_points.size();
            points_to_add.reserve(cur_pts);
            PointType point_world;
            for (int i = 0; i < cur_pts; i++)
            {
                PointBodyToWorld(&(cur_all_points.at(i)), &point_world);
                points_to_add.emplace_back(point_world);
            }
            PublishOdometry(pub_odom_aft_mapped_);
            if (path_pub_en_)
                PublishPath(pub_path_);
            if (dense_pub_en_)
                PublishFrameWorld();

            ivox_->AddPoints(points_to_add);
            first_lidar_time_ = measures_.lidar_bag_time_;
            flg_first_scan_ = false;
            return;
        }

        size_t cur_pts = scan_down_reg_.size();
        if (cur_pts < 5)
        {
            PublishOdometry(pub_odom_aft_mapped_);
            if (path_pub_en_)
                PublishPath(pub_path_);
            LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_reg_.size();
            return;
        }
        scan_down_world_.resize(cur_pts);
        nearest_points_.clear();
        nearest_points_.resize(cur_pts);
        residuals_.resize(cur_pts);
        residuals_.assign(cur_pts, 0);
        point_selected_surf_.resize(cur_pts);
        plane_coef_.clear();
        plane_coef_.resize(cur_pts);

        state_point_ = kf_.get_x();
        euler_cur = SO3ToEuler(state_point_.rot);

        print_status(0) = avg_voxel_size_;
        print_status(1) = double(cur_pts) / double(target_point_size_);

        ext_euler(0) = kf_.G_cur(3);
        ext_euler(1) = kf_.G_cur(4);
        ext_euler(2) = kf_.G_cur(5);

        fout_pre << std::setw(20) << measures_.lidar_bag_time_ - first_lidar_time_ << " " << euler_cur.transpose() << " "
                 << state_point_.pos.transpose() << " " << p_imu_->est_noise_.block<3, 3>(0, 0).diagonal().transpose()
                 << " " << p_imu_->est_noise_.block<3, 3>(3, 0).diagonal().transpose() << " " << ext_euler.transpose()
                 << " " << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
                 << state_point_.ba.transpose() << " " << state_point_.grav << " " << print_status.transpose() << std::endl;

        // ICP and iterated Kalman filter update
        Timer::Evaluate(
            [&, this]()
            {
                // iterated state estimation
                double solve_H_time = 0;
                // update the observation model, will call nn and point-to-plane residual computation
                kf_.update_iterated_dyn_share_akf(solve_H_time);
                // save the state
                state_point_ = kf_.get_x();
                state_cov_ = kf_.get_P().block<6, 6>(0, 0);
                euler_cur_ = SO3ToEuler(state_point_.rot);
                pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;

                euler_cur = SO3ToEuler(state_point_.rot);
                ext_euler(0) = kf_.G_cur(0);
                ext_euler(1) = kf_.G_cur(1);
                ext_euler(2) = kf_.G_cur(2);
            },
            "IEKF Solve and Update");

        Timer::Evaluate([&, this]()
                        { MapIncremental(); }, "    Incremental Mapping");

        auto frame_end_time = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::duration<double>>(frame_end_time - frame_start_time).count() * 1000;
        LOG(INFO) << "[Frame Time] Frame " << frame_num_ << " processing time: " << frame_time << " ms";

        LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " downsamp " << cur_pts
                  << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

        PublishOdometry(pub_odom_aft_mapped_);

        if (path_pub_en_)
            PublishPath(pub_path_);

        if (dense_pub_en_)
            PublishFrameWorld();

        if (scan_reg_pub_en_)
            PublishFrameRegWorld(pub_laser_cloud_reg_world_);

        if (map_pub_en_ && frame_num_ % 10 == 0)
        {
            PublishMap();
        }

        print_status(2) = double(effect_feat_num_) / double(cur_pts);

        fout_out << std::setw(20) << measures_.lidar_bag_time_ - first_lidar_time_ << " " << euler_cur.transpose() << " "
                 << state_point_.pos.transpose() << " " << p_imu_->est_noise_.block<3, 3>(6, 0).diagonal().transpose()
                 << " " << p_imu_->est_noise_.block<3, 3>(9, 0).diagonal().transpose() << " " << ext_euler.transpose()
                 << " " << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
                 << state_point_.ba.transpose() << " " << state_point_.grav << " " << print_status.transpose() << std::endl;
           
        frame_num_++;
    }

    void LaserMapping::VoxelGridDownsample(PointCloudType::Ptr &cloud_in, PointVector &cloud_down_reg)
    {
        size_t cloud_in_size = cloud_in->size();

        common::V3D loc_xyz;
        int64_t ijk0, ijk1, ijk2, idx;

        cloud_down_reg.clear();
        cloud_down_reg.reserve(cloud_in_size);

        cur_all_points.clear();
        cur_all_points.resize(cloud_in_size);

        std::vector<size_t> index0(cloud_in_size);

        common::V3D p3d;

        for (size_t i = 0; i < cloud_in_size; ++i)
        {
            index0[i] = i;
        }
        std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                      {
        PointType &tp = cur_all_points[i];
        common::V3D pt3d;
        common::M3D body_cov;
        tp.x = cloud_in->points[i].x;
        tp.y = cloud_in->points[i].y;
        tp.z = cloud_in->points[i].z;

        tp.intensity = cloud_in->points[i].intensity;
        tp.pt_num = 1;
        tp.use_num = 1;
        tp.time = measures_.lidar_bag_time_ - first_lidar_time_ + cloud_in->points[i].curvature / 1000;
        pt3d << cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z;
        tp.cov = options::LIDAR_COV * Eigen::Matrix3d::Identity();
        tp.uncertainty = init_uncertainty_; });

        if (adap_voxel_size_en_ && target_point_size_ >= cloud_in_size)
        {
            scan_voxel_size_ = min_voxel_size_;
        }
        else
        {
            if (adap_voxel_size_en_)
            {
                int max_sample_cnt = 10;

                if (!voxel_size_sliding_.empty())
                {
                    scan_voxel_size_ = 0;
                    for (auto &it : voxel_size_sliding_)
                    {
                        scan_voxel_size_ += it;
                    }
                    scan_voxel_size_ /= voxel_size_sliding_.size();
                    scan_voxel_size_ = round(scan_voxel_size_ / 0.001f) * 0.001f;
                }

                for (int iter = 0; iter < max_sample_cnt; iter++)
                {
                    cur_iter = iter;
                    std::unordered_map<size_t, size_t> sample_volume_hash;
                    sample_volume_hash.clear();
                    sample_volume_hash.reserve(cloud_in_size / 2);

                    std::vector<size_t> hash_values(cloud_in_size);
                    std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                                  {
                    p3d << cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z;
                    for (int j = 0; j < 3; j++) {
                        p3d[j] = p3d[j] / scan_voxel_size_;
                    }
                    ijk0 = int64_t(floor(p3d[0]));
                    ijk1 = int64_t(floor(p3d[1]));
                    ijk2 = int64_t(floor(p3d[2]));
                    hash_values[i] = ComputeVoxelHash(ijk0, ijk1, ijk2); });

                    for (size_t i = 0; i < cloud_in_size; ++i)
                    {
                        sample_volume_hash[hash_values[i]]++;
                    }

                    double sample_target_ratio = double(sample_volume_hash.size()) / double(target_point_size_);

                    if (fabs(sample_target_ratio - 1.0f) < 0.1f)
                    {
                        break;
                    }
                    else
                    {
                        double last_scan_voxel_size_ = scan_voxel_size_;
                        scan_voxel_size_ = (sample_target_ratio / 2.0f + 0.5f) * scan_voxel_size_;
                        scan_voxel_size_ = round(scan_voxel_size_ / 0.001f) * 0.001f;
                        if (fabs(last_scan_voxel_size_ - scan_voxel_size_) < 0.001f)
                            break;
                    }
                }
                avg_voxel_size_ = scan_voxel_size_;
                voxel_size_sliding_.push_back(scan_voxel_size_);
                if (voxel_size_sliding_.size() > 3)
                    voxel_size_sliding_.pop_front();
            }
        }

        if (scan_voxel_size_ < min_voxel_size_)
            scan_voxel_size_ = min_voxel_size_;

        std::unordered_map<size_t, PointVector> leaves_;
        leaves_.clear();
        leaves_.reserve(cloud_in_size / 2);

        std::vector<size_t> final_hash_values(cloud_in_size);
        std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                      {
        loc_xyz << cur_all_points[i].x, cur_all_points[i].y, cur_all_points[i].z;
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = loc_xyz[j] / scan_voxel_size_;
        }
        ijk0 = int64_t(floor(loc_xyz[0]));
        ijk1 = int64_t(floor(loc_xyz[1]));
        ijk2 = int64_t(floor(loc_xyz[2]));
        final_hash_values[i] = ComputeVoxelHash(ijk0, ijk1, ijk2); });

        for (size_t cp = 0; cp < cloud_in_size; cp++)
        {
            auto &voxel = leaves_[final_hash_values[cp]];
            if (voxel.empty())
            {
                voxel.reserve(4);
            }
            voxel.push_back(cur_all_points[cp]);
        }

        cloud_down_reg.reserve(leaves_.size());
        for (const auto &leaf : leaves_)
        {
            cloud_down_reg.push_back(leaf.second.front());
        }

        return;
    }

    bool LaserMapping::time_list(const PointType2 &x, const PointType2 &y) { return (x.curvature < y.curvature); };

    void LaserMapping::StandardPCLCallBack(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        mtx_buffer_.lock();
        Timer::Evaluate(
            [&, this]()
            {
                scan_count_++;
                double msg_time = static_cast<double>(msg->header.stamp.sec) +
                                  static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
                if (msg_time < last_timestamp_lidar_)
                {
                    LOG(ERROR) << "lidar loop back, clear buffer";
                    lidar_buffer_.clear();
                }

                PointCloudType::Ptr ptr(new PointCloudType());
                preprocess_->Process(msg, ptr);
                lidar_buffer_.push_back(ptr);
                time_buffer_.push_back(msg_time);
                last_timestamp_lidar_ = msg_time;
            },
            "Preprocess (Standard)");
        mtx_buffer_.unlock();
    }

    void LaserMapping::LivoxPCLCallBack(const livox_interfaces::msg::CustomMsg::SharedPtr msg)
    {
        mtx_buffer_.lock();
        Timer::Evaluate(
            [&, this]()
            {
                scan_count_++;
                double msg_time = static_cast<double>(msg->header.stamp.sec) +
                                  static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
                if (msg_time < last_timestamp_lidar_)
                {
                    LOG(WARNING) << "lidar loop back, clear buffer";
                    lidar_buffer_.clear();
                }

                last_timestamp_lidar_ = msg_time;

                PointCloudType::Ptr ptr(new PointCloudType());
                preprocess_->Process(msg, ptr);
                lidar_buffer_.emplace_back(ptr);
                time_buffer_.emplace_back(last_timestamp_lidar_);
            },
            "Preprocess (Livox)");

        mtx_buffer_.unlock();
    }

    void LaserMapping::IMUCallBack(const sensor_msgs::msg::Imu::SharedPtr msg_in)
    {
        publish_count_++;
        auto msg = std::make_shared<sensor_msgs::msg::Imu>(*msg_in);

        double timestamp = static_cast<double>(msg->header.stamp.sec) +
                           static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

        mtx_buffer_.lock();
        if (timestamp < last_timestamp_imu_)
        {
            LOG(WARNING) << "imu loop back, clear buffer";
            imu_buffer_.clear();
        }

        last_timestamp_imu_ = timestamp;
        imu_buffer_.emplace_back(msg);
        mtx_buffer_.unlock();
    }

    bool LaserMapping::SyncPackages()
    {
        if (lidar_buffer_.empty() || (imu_buffer_.empty()))
        {
            return false;
        }

        /*** push a lidar scan ***/
        if (!lidar_pushed_)
        {
            measures_.lidar_ = lidar_buffer_.front();
            measures_.lidar_bag_time_ = time_buffer_.front();

            if (measures_.lidar_->points.size() <= 1)
            {
                LOG(WARNING) << "Too few input point cloud!";
                lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
            }
            else if (measures_.lidar_->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_)
            {
                lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
            }
            else
            {
                scan_num_++;
                lidar_end_time_ = measures_.lidar_bag_time_ + measures_.lidar_->points.back().curvature / double(1000);
                lidar_mean_scantime_ +=
                    (measures_.lidar_->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
            }

            measures_.lidar_end_time_ = lidar_end_time_;
            lidar_pushed_ = true;
        }

        if (last_timestamp_imu_ < lidar_end_time_)
        {
            return false;
        }

        /*** push imu_ data, and pop from imu_ buffer ***/

        auto get_imu_time = [](const sensor_msgs::msg::Imu::SharedPtr &msg) {
            return static_cast<double>(msg->header.stamp.sec) +
                   static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
        };

        double imu_time = get_imu_time(imu_buffer_.front());
        measures_.imu_.clear();
        while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_))
        {
            imu_time = get_imu_time(imu_buffer_.front());
            if (imu_time > lidar_end_time_)
                break;
            measures_.imu_.push_back(imu_buffer_.front());
            imu_buffer_.pop_front();
        }

        lidar_buffer_.pop_front();
        time_buffer_.pop_front();
        lidar_pushed_ = false;
        return true;
    }

    void LaserMapping::MapIncremental()
    {
        PointVector points_to_add;
        PointVector points_to_update;

        state_point_ = kf_.get_x();
        int cur_pts = scan_down_reg_.size();
        points_to_add.clear();
        points_to_add.resize(cur_pts);
        std::vector<size_t> index(cur_pts);
        for (size_t i = 0; i < cur_pts; i++)
        {
            index[i] = i;
        }
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i)
                      {
        PointType &point_body = scan_down_reg_.at(i);
        PointType &point_world = points_to_add.at(i);
        PointBodyToWorld(&point_body, &point_world);
        point_world.uncertainty = init_uncertainty_; });

        points_to_update.clear();
        points_to_update.reserve(cur_pts * options::MIN_NUM_MATCH_POINTS);

        for (size_t i = 0; i < cur_pts; i++)
        {
            auto &points_near = nearest_points_[i];
            PointType &point_world = points_to_add.at(i);
            if (!points_near.empty() && point_selected_surf_[i])
            {
                int j = points_near.size();
                for (int k = 0; k < j; k++)
                {
                    auto &p_update = points_near.at(k);
                    p_update.uncertainty = point_world.uncertainty;
                    p_update.pt_num = 0;
                    p_update.use_num = 1;
                    points_to_update.emplace_back(p_update);
                }
            }
        }

        ivox_->UpdateUncertainty(points_to_update);

        ivox_->AddPoints(points_to_add);
        ivox_->ErasePoints(state_point_.pos, lidar_end_time_ - first_lidar_time_);
    }

    /**
     * Lidar point cloud registration
     * will be called by the eskf custom observation model
     * compute point-to-plane residual here
     * @param s kf state
     * @param ekfom_data H matrix
     */
    void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
    {
        int cnt_pts = scan_down_reg_.size();
        std::vector<size_t> index(cnt_pts);
        for (size_t i = 0; i < cnt_pts; i++)
        {
            index[i] = i;
        }
        state_point_ = kf_.get_x();
        state_cov_ = kf_.get_P().block<6, 6>(0, 0);
        Timer::Evaluate(
            [&, this]()
            {
                auto R_wl = (s.rot * s.offset_R_L_I);
                auto t_wl = (s.rot * s.offset_T_L_I + s.pos);

                {
                    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i)
                                  {
                    PointType &point_body = scan_down_reg_.at(i);
                    PointType &point_world = scan_down_world_.at(i);

                    /* transform to world frame */
                    common::V3D p_body(point_body.x, point_body.y, point_body.z);
                    PointBodyToWorld(&point_body, &point_world);
                    //                                        if(ekfom_data.converge)
                    {
                        auto &points_near = nearest_points_[i];
                        /** Find the closest surfaces in the map **/
                        ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS, 3);

                        auto &p_sum = plane_coef_[i];
                        if (!points_near.empty()) {
                            PointType p1;
                            common::V3D incident_normal;
                            common::M3D project_2D_cov;
                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_2D;

                            p_sum = points_near.at(0);

                            for (int j = 1; j < points_near.size(); j++) {
                                if (p_sum.pt_num < options::MIN_NUM_MATCH_POINTS) {
                                    p1 = Merge2(points_near.at(j), p_sum);
                                    p_sum = p1;
                                    continue;
                                } else {
                                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es1(p_sum.cov);
                                    Eigen::Vector3d normal_x = es1.eigenvectors().col(0);
                                    Eigen::Vector3d normal_y = es1.eigenvectors().col(1);
                                    Eigen::Vector3d normal_z = es1.eigenvectors().col(2);
                                    Eigen::Vector3d pq = Eigen::Vector3d(p_sum.x, p_sum.y, p_sum.z) -
                                                         Eigen::Vector3d(point_world.x, point_world.y, point_world.z);
                                    double p2q =
                                        pow(normal_y.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(1)) +
                                        pow(normal_z.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(2));

                                    if ((es1.eigenvalues().real()(1) < 0.04f) ||
                                        p2q > t_stop_pseudo_merge_ * t_stop_pseudo_merge_) {
                                        p1 = Merge2(points_near.at(j), p_sum);
                                        p_sum = p1;
                                        continue;
                                    } else {
                                        for (int k = j; k < points_near.size();) points_near.pop_back();
                                    }
                                }
                            }
                            if (p_sum.pt_num < options::MIN_NUM_MATCH_POINTS) {
                                points_near.clear();
                            } else {
                                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es1(p_sum.cov);
                                Eigen::Vector3d normal_x = es1.eigenvectors().col(0);
                                Eigen::Vector3d normal_y = es1.eigenvectors().col(1);
                                Eigen::Vector3d normal_z = es1.eigenvectors().col(2);
                                Eigen::Vector3d pq = Eigen::Vector3d(p_sum.x, p_sum.y, p_sum.z) -
                                                     Eigen::Vector3d(point_world.x, point_world.y, point_world.z);
                                double p2q = pow(normal_y.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(1)) +
                                             pow(normal_z.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(2));

                                if ((es1.eigenvalues().real()(1) >= 0.04f) &&
                                    p2q <= t_stop_pseudo_merge_ * t_stop_pseudo_merge_) {
                                    double p2pl = normal_x.transpose() * pq;

                                    if (points_near.empty() || fabs(p2pl) > 1.0f / 9.0f * sqrt(p_body.norm())) {
                                        points_near.clear();
                                    } else {
                                        point_body.uncertainty = p2pl * p2pl;
                                    }
                                } else
                                    points_near.clear();
                            }
                        }
                    } });
                }
            },
            "    ObsModel (Lidar Match)");
        //        LOG(INFO)<<  "ObsModel (Lidar Match)";

        effect_feat_num_ = 0;

        corr_pts_.resize(cnt_pts);
        corr_norm_.resize(cnt_pts);
        for (int i = 0; i < cnt_pts; i++)
        {
            if (!nearest_points_[i].empty())
            {
                point_selected_surf_[i] = true;
            }
            else
                point_selected_surf_[i] = false;

            if (point_selected_surf_[i])
            {
                point_selected_idx_[effect_feat_num_] = i;
                corr_norm_[effect_feat_num_] = plane_coef_[i];
                corr_pts_[effect_feat_num_] = scan_down_reg_.at(i);
                effect_feat_num_++;
            }
            else
                scan_down_reg_.at(i).uncertainty = (init_uncertainty_);
        }
        corr_pts_.resize(effect_feat_num_);
        corr_norm_.resize(effect_feat_num_);

        if (effect_feat_num_ < 1)
        {
            ekfom_data.valid = false;
            RCLCPP_WARN(node_->get_logger(), "No Effective Points!");
            return;
        }
        Timer::Evaluate(
            [&, this]()
            {
                ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12); // 23
                ekfom_data.h.resize(effect_feat_num_);
                ekfom_data.R_inv.resize(effect_feat_num_, 1);
                std::vector<size_t> index2(effect_feat_num_);
                for (size_t i = 0; i < effect_feat_num_; i++)
                {
                    index2[i] = i;
                }
                const common::M3D off_R = s.offset_R_L_I.toRotationMatrix();
                const common::V3D off_t = s.offset_T_L_I;
                const common::M3D Rt = s.rot.toRotationMatrix().transpose();

                std::for_each(std::execution::par_unseq, index2.begin(), index2.end(), [&](const size_t &i)
                              {
                                common::V3D point_this_be(corr_pts_[i].x, corr_pts_[i].y, corr_pts_[i].z);
                                common::M3D point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                                common::V3D point_this = off_R * point_this_be + off_t;
                                common::M3D point_crossmat = SKEW_SYM_MATRIX(point_this);
                                common::V3D point_world = s.rot.toRotationMatrix() * point_this + s.pos;

                                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es1(corr_norm_[i].cov);
                                common::V3D norm_p = es1.eigenvectors().col(0);
                                common::V3D map_q(corr_norm_[i].x, corr_norm_[i].y, corr_norm_[i].z);
                                ekfom_data.h_x.block<1, 3>(i, 0) << norm_p.transpose();
                                ekfom_data.h_x.block<1, 3>(i, 3) << -norm_p.transpose() * (s.rot.toRotationMatrix() * point_crossmat);
                                ekfom_data.h(i) = norm_p.transpose() * (map_q - point_world);

                                double thickness = (norm_p.transpose() * (corr_norm_[i].cov) * norm_p);

                                ekfom_data.R_inv(i) = 1.0f / (exp(t_ratio_b_*corr_norm_[i].uncertainty) * thickness);});       
            },"    ObsModel (IEKF Build Jacobian)");
    }

    /////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

    void LaserMapping::PublishPath(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pub_path)
    {
        SetPosestamp(msg_body_pose_);
        msg_body_pose_.header.stamp = rclcpp::Time(static_cast<int64_t>(lidar_end_time_ * 1e9));
        msg_body_pose_.header.frame_id = "camera_init";

        /*** if path is too large, the rviz will crash ***/
        path_.poses.push_back(msg_body_pose_);
        if (run_in_offline_ == false)
        {
            pub_path->publish(path_);
        }
    }

    void LaserMapping::PublishOdometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pub_odom_aft_mapped)
    {
        odom_aft_mapped_.header.frame_id = "camera_init";
        odom_aft_mapped_.child_frame_id = "body";
        odom_aft_mapped_.header.stamp = rclcpp::Time(static_cast<int64_t>(lidar_end_time_ * 1e9));
        SetPosestamp(odom_aft_mapped_.pose);

        auto P = kf_.get_P();
        for (int i = 0; i < 6; i++)
        {
            odom_aft_mapped_.pose.covariance[i * 6 + 0] = P(i, 0);
            odom_aft_mapped_.pose.covariance[i * 6 + 1] = P(i, 1);
            odom_aft_mapped_.pose.covariance[i * 6 + 2] = P(i, 2);
            odom_aft_mapped_.pose.covariance[i * 6 + 3] = P(i, 3);
            odom_aft_mapped_.pose.covariance[i * 6 + 4] = P(i, 4);
            odom_aft_mapped_.pose.covariance[i * 6 + 5] = P(i, 5);
        }
        pub_odom_aft_mapped->publish(odom_aft_mapped_);

        // TF broadcast
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = odom_aft_mapped_.header.stamp;
        transform.header.frame_id = "camera_init";
        transform.child_frame_id = "body";
        transform.transform.translation.x = odom_aft_mapped_.pose.pose.position.x;
        transform.transform.translation.y = odom_aft_mapped_.pose.pose.position.y;
        transform.transform.translation.z = odom_aft_mapped_.pose.pose.position.z;
        transform.transform.rotation = odom_aft_mapped_.pose.pose.orientation;
        tf_broadcaster_->sendTransform(transform);
    }

    void LaserMapping::PublishFrameWorld()
    {
        PointCloudType::Ptr laserCloudWorld;

        PointCloudType::Ptr laserCloudFullRes(scan_undistort_);
        int size = laserCloudFullRes->points.size();
        laserCloudWorld.reset(new PointCloudType(size, 1));
        for (int i = 0; i < size; i++)
        {
            common::V3D p_body(laserCloudFullRes->points.at(i).x, laserCloudFullRes->points.at(i).y,
                               laserCloudFullRes->points.at(i).z);
            common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                                 state_point_.pos);

            laserCloudWorld->points[i].x = p_global(0);
            laserCloudWorld->points[i].y = p_global(1);
            laserCloudWorld->points[i].z = p_global(2);
            laserCloudWorld->points[i].intensity = laserCloudFullRes->points.at(i).intensity;
            laserCloudWorld->points[i].curvature = laserCloudFullRes->points.at(i).curvature;
            laserCloudWorld->points[i].normal_x = laserCloudFullRes->points.at(i).normal_x;
            laserCloudWorld->points[i].normal_y = laserCloudFullRes->points.at(i).normal_y;
            laserCloudWorld->points[i].normal_z = laserCloudFullRes->points.at(i).normal_z;
        }
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = rclcpp::Time(static_cast<int64_t>(lidar_end_time_ * 1e9));
        laserCloudmsg.header.frame_id = "camera_init";
        pub_laser_cloud_world_->publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;
    }

    void LaserMapping::GetRainbowColor(float value, common::V3F &color)
    {
        // this is HSV color palette with hue values going only from 0.0 to 0.833333.

        value = std::min(value, 1.0f);
        value = std::max(value, 0.0f);

        float h = value * 5.0f + 1.0f;
        int i = floor(h);
        float f = h - i;
        if (!(i & 1))
            f = 1 - f; // if i is even
        float n = 1 - f;

        if (i <= 1)
            color[0] = n, color[1] = 0, color[2] = 1;
        else if (i == 2)
            color[0] = 0, color[1] = n, color[2] = 1;
        else if (i == 3)
            color[0] = 0, color[1] = 1, color[2] = n;
        else if (i == 4)
            color[0] = n, color[1] = 1, color[2] = 0;
        else if (i >= 5)
            color[0] = 1, color[1] = n, color[2] = 0;
    }

    void LaserMapping::PublishMap()
    {
        ivox_->GetMapPoints(map_points_);
        int size = map_points_.size();
        LOG(INFO) << "map size " << size << " voxel size " << ivox_->NumValidGrids() << " average point each voxel "
                  << float(size) / float(ivox_->NumValidGrids());
        if (size == 0)
            return;
        sensor_msgs::msg::PointCloud2 laserCloudMap;
        PointCloudType::Ptr map_pub(new PointCloudType());
        PointType2 body_p;
        map_pub->clear();
        double max_intensity = 0, min_z = 10000.0, max_z = -10000.0;
        min_z = state_point_.pos.z() + 3;
        max_z = state_point_.pos.z() + 30;
        {
            common::M3D world_cov;
            visualization_msgs::msg::Marker p_cov;
            p_cov.type = visualization_msgs::msg::Marker::SPHERE;
            p_cov.action = visualization_msgs::msg::Marker::ADD;
            p_cov.header.frame_id = "camera_init";
            p_cov.header.stamp = rclcpp::Time(static_cast<int64_t>(lidar_end_time_ * 1e9));
            p_cov.lifetime = rclcpp::Duration(10, 0); // 10 seconds
            pa_cov_.markers.clear();
            common::V3D state_pos(state_point_.pos);
            common::V3D map_p;
            double dis = 0;
            int pt_cnt = -1;
            for (auto it = map_points_.begin(); it != map_points_.end(); it++)
            {
                world_cov = it->cov;
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_cov3(world_cov);
                Eigen::Matrix3d sorted_evecs_cov3 = es_cov3.eigenvectors().real().transpose();
                sorted_evecs_cov3.row(2) = sorted_evecs_cov3.row(0).cross(sorted_evecs_cov3.row(1));
                common::V3D ev_cov3 = es_cov3.eigenvalues().real();

                body_p.x = it->x;
                body_p.y = it->y;
                body_p.z = it->z;
                body_p.curvature = it->pt_num;
                body_p.intensity = it->intensity;
                body_p.normal_x = (it->uncertainty);
                body_p.normal_y = it->use_num;
                body_p.normal_z = sqrt(fabs(ev_cov3(2)));
                map_pub->push_back(body_p);
                if (gaussian_publish_en_)
                {
                    if (it->intensity > max_intensity)
                        max_intensity = it->intensity;

                    if (it->pt_num <= gaussian_pub_min_cnt_)
                        continue;
                    map_p << it->x, it->y, it->z;
                    dis = (map_p.x() - state_pos.x()) * (map_p.x() - state_pos.x()) +
                          (map_p.y() - state_pos.y()) * (map_p.y() - state_pos.y()) +
                          (map_p.z() - state_pos.z()) * (map_p.z() - state_pos.z());

                    if (dis > gaussian_pub_dis_ * gaussian_pub_dis_)
                        continue; // pub local gmm only

                    p_cov.scale.x = 4 * sqrt(fabs(ev_cov3(0))); // scale is diameter 2 sigema
                    p_cov.scale.y = 4 * sqrt(fabs(ev_cov3(1)));
                    p_cov.scale.z = 4 * sqrt(fabs(ev_cov3(2)));
                    p_cov.pose.position.x = it->x;
                    p_cov.pose.position.y = it->y;
                    p_cov.pose.position.z = it->z;
                    p_cov.id = pa_cov_.markers.size();
                    float x = 0;

                    common::V3F rgb;
                    {
                        x = 1.0 - ((it->z - min_z) / (max_z - min_z));
                        GetRainbowColor(x, rgb);
                    }
                    p_cov.color.r = rgb[0];
                    p_cov.color.g = rgb[1];
                    p_cov.color.b = rgb[2];
                    p_cov.ns = "gaussian_map";

                    p_cov.color.a = 0.8;
                    Eigen::Matrix3d rotation3 = sorted_evecs_cov3.transpose();
                    Eigen::Quaterniond eq3(rotation3);
                    p_cov.pose.orientation.w = eq3.w();
                    p_cov.pose.orientation.x = eq3.x();
                    p_cov.pose.orientation.y = eq3.y();
                    p_cov.pose.orientation.z = eq3.z();
                    pa_cov_.markers.push_back(p_cov);
                }
            }
            map_cov_pub_->publish(pa_cov_);
        }

        pcl::toROSMsg(*map_pub, laserCloudMap);
        laserCloudMap.header.stamp = rclcpp::Time(static_cast<int64_t>(lidar_end_time_ * 1e9));
        laserCloudMap.header.frame_id = "camera_init";
        pub_map_->publish(laserCloudMap);
    }

    void LaserMapping::PublishFrameRegWorld(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub_laser_cloud_reg_world)
    {
        int size = scan_down_reg_.size();
        PointCloudType::Ptr laser_cloud(new PointCloudType(size, 1));

        for (int i = 0; i < size; i++)
        {
            PointBodyToWorldPub(&scan_down_reg_.at(i), &laser_cloud->points[i]);
            if (point_selected_surf_[i])
                laser_cloud->points[i].normal_x = plane_coef_[i].pt_num;
            else
                laser_cloud->points[i].normal_x = 0;
        }
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laser_cloud, laserCloudmsg);
        laserCloudmsg.header.stamp = rclcpp::Time(static_cast<int64_t>(lidar_end_time_ * 1e9));
        laserCloudmsg.header.frame_id = "camera_init";
        pub_laser_cloud_reg_world->publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;
        if (gaussian_publish_en_)
        {
            common::M3D world_cov;
            visualization_msgs::msg::MarkerArray pa_scan_cov;
            visualization_msgs::msg::Marker p_cov;

            p_cov.type = visualization_msgs::msg::Marker::SPHERE;
            p_cov.action = visualization_msgs::msg::Marker::ADD;
            p_cov.header.frame_id = "camera_init";
            p_cov.header.stamp = rclcpp::Time(static_cast<int64_t>(lidar_end_time_ * 1e9));
            p_cov.lifetime = rclcpp::Duration(0, 0);
            pa_scan_cov.markers.clear();

            size = scan_down_reg_.size();
            scan_down_reg_.resize(size);
            for (int i = 0; i < scan_down_reg_.size(); i++)
            {
                PointBodyToWorld(&scan_down_reg_[i], &scan_down_world_[i]);
                common::V3D p_body(scan_down_world_[i].x, scan_down_world_[i].y, scan_down_world_[i].z);
                world_cov = scan_down_world_.at(i).cov.block<3, 3>(0, 0);
                p_cov.pose.position.x = scan_down_world_[i].x;
                p_cov.pose.position.y = scan_down_world_[i].y;
                p_cov.pose.position.z = scan_down_world_[i].z;
                p_cov.id = i;
                p_cov.ns = "scan_cov";
                p_cov.color.r = 1;
                p_cov.color.g = 0;
                p_cov.color.b = 0;

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_cov3(world_cov);
                Eigen::Matrix3d sorted_evecs_cov3 = es_cov3.eigenvectors().real().transpose();
                sorted_evecs_cov3.row(2) = sorted_evecs_cov3.row(0).cross(sorted_evecs_cov3.row(1));
                common::V3D ev_cov3 = es_cov3.eigenvalues().real();
                p_cov.scale.x = 12 * sqrt(fabs(ev_cov3(0))); // scale is diameter
                p_cov.scale.y = 12 * sqrt(fabs(ev_cov3(1)));
                p_cov.scale.z = 12 * sqrt(fabs(ev_cov3(2)));
                p_cov.color.a = 1;
                Eigen::Matrix3d rotation3 = sorted_evecs_cov3.transpose();
                Eigen::Quaterniond eq3(rotation3);
                p_cov.pose.orientation.w = eq3.w();
                p_cov.pose.orientation.x = eq3.x();
                p_cov.pose.orientation.y = eq3.y();
                p_cov.pose.orientation.z = eq3.z();
                pa_scan_cov.markers.push_back(p_cov);
            }

            size = corr_norm_.size();
            for (int i = 0; i < size; i++)
            {
                world_cov = corr_norm_.at(i).cov.block<3, 3>(0, 0);
                p_cov.pose.position.x = corr_norm_.at(i).x;
                p_cov.pose.position.y = corr_norm_.at(i).y;
                p_cov.pose.position.z = corr_norm_.at(i).z;
                p_cov.id = i;

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_cov3(world_cov);
                Eigen::Matrix3d sorted_evecs_cov3 = es_cov3.eigenvectors().real().transpose();
                sorted_evecs_cov3.row(2) = sorted_evecs_cov3.row(0).cross(sorted_evecs_cov3.row(1));
                common::V3D ev_cov3 = es_cov3.eigenvalues().real();

                Eigen::Matrix3d rotation3 = sorted_evecs_cov3.transpose();
                Eigen::Quaterniond eq3(rotation3);
                p_cov.pose.orientation.w = eq3.w();
                p_cov.pose.orientation.x = eq3.x();
                p_cov.pose.orientation.y = eq3.y();
                p_cov.pose.orientation.z = eq3.z();

                p_cov.scale.x = 4 * sqrt(fabs(ev_cov3(0))); // scale is diameter
                p_cov.scale.y = 4 * sqrt(fabs(ev_cov3(1)));
                p_cov.scale.z = 4 * sqrt(fabs(ev_cov3(2)));
                p_cov.color.r = 0;
                p_cov.color.g = 0;
                p_cov.color.b = 1;
                p_cov.color.a = 0.3;
                p_cov.ns = "cor_cov";
                p_cov.type = visualization_msgs::msg::Marker::SPHERE;
                pa_scan_cov.markers.push_back(p_cov);

                rotation3.setZero();
                rotation3.row(0) = sorted_evecs_cov3.row(0);
                if (rotation3.row(0).sum() < 0)
                    rotation3 = -rotation3;
                Eigen::Quaterniond eq(rotation3.transpose());
                eq.normalize();
                p_cov.pose.orientation.w = eq.w();
                p_cov.pose.orientation.x = eq.x();
                p_cov.pose.orientation.y = eq.y();
                p_cov.pose.orientation.z = eq.z();

                p_cov.scale.x = 1;
                p_cov.scale.y = 0.2;
                p_cov.scale.z = 0.2;
                p_cov.color.r = 1;
                p_cov.color.g = 1;
                p_cov.color.b = 0;
                p_cov.color.a = 1;
                p_cov.ns = "cor_normal";
                p_cov.type = visualization_msgs::msg::Marker::ARROW;
                pa_scan_cov.markers.push_back(p_cov);
            }
            p_cov.type = visualization_msgs::msg::Marker::LINE_LIST;
            p_cov.ns = "scan_cor_line";
            p_cov.pose.position.x = 0;
            p_cov.pose.position.y = 0;
            p_cov.pose.position.z = 0;
            p_cov.pose.orientation.w = 1;
            p_cov.pose.orientation.x = 0;
            p_cov.pose.orientation.y = 0;
            p_cov.pose.orientation.z = 0;
            p_cov.color.r = 0;
            p_cov.color.g = 1;
            p_cov.color.b = 0;
            p_cov.color.a = 1;
            p_cov.scale.x = 0.03;

            for (int i = 0; i < size; i++)
            {
                geometry_msgs::msg::Point p;
                PointType eff_world;
                p_cov.id = i;
                p.x = corr_norm_[i].x;
                p.y = corr_norm_[i].y;
                p.z = corr_norm_[i].z;
                p_cov.points.push_back(p);
                PointBodyToWorld(&corr_pts_.at(i), &eff_world);
                p.x = eff_world.x;
                p.y = eff_world.y;
                p.z = eff_world.z;
                p_cov.points.push_back(p);
                pa_scan_cov.markers.push_back(p_cov);
                p_cov.points.clear();
            }
            point_cov_pub_->publish(pa_scan_cov);
        }
    }

    void LaserMapping::Savetrajectory(const std::string &traj_file)
    {
        std::ofstream ofs;
        ofs.open(traj_file, std::ios::out);
        if (!ofs.is_open())
        {
            LOG(ERROR) << "Failed to open traj_file: " << traj_file;
            return;
        }

        ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
        for (const auto &p : path_.poses)
        {
            double stamp_sec = static_cast<double>(p.header.stamp.sec) +
                               static_cast<double>(p.header.stamp.nanosec) * 1e-9;
            ofs << std::fixed << std::setprecision(6) << stamp_sec << " " << std::setprecision(15)
                << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
                << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
        }

        ofs.close();

        // ofs kitti
        std::string kitti_file = (std::string(std::string(ROOT_DIR) + "Log/" + "kitti.txt"));
        ofs.open(kitti_file, std::ios::out);

        if (!ofs.is_open())
        {
            LOG(ERROR) << "Failed to open kitti_file: " << kitti_file;
            return;
        }

        for (const auto &p : path_.poses)
        {
            SO3 r_matrix;
            r_matrix.x() = p.pose.orientation.x;
            r_matrix.y() = p.pose.orientation.y;
            r_matrix.z() = p.pose.orientation.z;
            r_matrix.w() = p.pose.orientation.w;
            Eigen::MatrixXd kitti_pose;
            kitti_pose.resize(3, 4);
            kitti_pose.block<3, 3>(0, 0) = r_matrix.toRotationMatrix();
            kitti_pose(0, 3) = p.pose.position.x;
            kitti_pose(1, 3) = p.pose.position.y;
            kitti_pose(2, 3) = p.pose.position.z;
            ofs << std::fixed << std::setprecision(18) << kitti_pose(0, 0) << " " << kitti_pose(0, 1) << " "
                << kitti_pose(0, 2) << " " << kitti_pose(0, 3) << " " << kitti_pose(1, 0) << " " << kitti_pose(1, 1) << " "
                << kitti_pose(1, 2) << " " << kitti_pose(1, 3) << " " << kitti_pose(2, 0) << " " << kitti_pose(2, 1) << " "
                << kitti_pose(2, 2) << " " << kitti_pose(2, 3) << std::endl;
        }
        ofs.close();
    }

    ///////////////////////////  private method /////////////////////////////////////////////////////////////////////
    template <typename T>
    void LaserMapping::SetPosestamp(T &out)
    {
        out.pose.position.x = state_point_.pos(0);
        out.pose.position.y = state_point_.pos(1);
        out.pose.position.z = state_point_.pos(2);
        out.pose.orientation.x = state_point_.rot.coeffs()[0];
        out.pose.orientation.y = state_point_.rot.coeffs()[1];
        out.pose.orientation.z = state_point_.rot.coeffs()[2];
        out.pose.orientation.w = state_point_.rot.coeffs()[3];
    }

    void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po)
    {
        common::V3D p_body(pi->x, pi->y, pi->z);
        common::V3D p_this(state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I);
        common::V3D p_global(state_point_.rot * p_this + state_point_.pos);
        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
        po->uncertainty = pi->uncertainty;
        po->pt_num = pi->pt_num;
        po->use_num = pi->use_num;
        po->time = pi->time;

        common::M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRIX(p_this);

        po->cov = state_point_.rot * state_point_.offset_R_L_I * pi->cov * state_point_.offset_R_L_I.conjugate() *
                  state_point_.rot.conjugate();
    }

    void LaserMapping::PointBodyToWorldPub(const PointType *pi, PointType2 *const po)
    {
        common::V3D p_body(pi->x, pi->y, pi->z);
        common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                             state_point_.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
        po->curvature = pi->pt_num;
    }

    void LaserMapping::PointBodyToWorld(const common::V3F &pi, PointType *const po)
    {
        common::V3D p_body(pi.x(), pi.y(), pi.z());
        common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                             state_point_.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = std::abs(po->z);
    }

    void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po)
    {
        common::V3D p_body_lidar(pi->x, pi->y, pi->z);
        common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

        po->x = p_body_imu(0);
        po->y = p_body_imu(1);
        po->z = p_body_imu(2);
        po->intensity = pi->intensity;
    }

} // namespace akf_lio