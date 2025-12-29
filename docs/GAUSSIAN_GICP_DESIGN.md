# Gaussian GICP 設計書

small_gicp に AKF-LIO の Gaussian Map 手法を統合するための設計書

## 1. 概要

### 1.1 目的

既存の VGICP ベースのローカライゼーションを、AKF-LIO の Gaussian Map 手法で置き換え、以下の利点を得る：

- マハラノビス距離による robust な対応点探索
- Pseudo-merge による高精度な平面推定
- 適応的な不確実性追跡と重み付け

### 1.2 システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                      オフライン処理                              │
├─────────────────────────────────────────────────────────────────┤
│  [既存点群地図 PCD] → [Gaussian Map 構築] → [Gaussian Map 保存]  │
│                              ↓                                   │
│                    ・ボクセル化                                   │
│                    ・共分散計算                                   │
│                    ・pseudo-merge                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      オンライン処理                              │
├─────────────────────────────────────────────────────────────────┤
│  [Gaussian Map ロード]                                           │
│         ↓                                                        │
│  [リアルタイム点群] → [初期姿勢] → [Gaussian マッチング] → [姿勢] │
│         ↑                              ↓                         │
│      [LIO]                    ・マハラノビス距離で対応探索        │
│                               ・Point-to-Gaussian 残差           │
│                               ・Gauss-Newton 最適化              │
│                               ・uncertainty 更新                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 VGICP vs Gaussian GICP 比較

| 項目 | VGICP (small_gicp) | Gaussian GICP (提案) |
|------|-------------------|---------------------|
| マップ表現 | ボクセル化点群 | Gaussian Map (平均 + 共分散) |
| 対応点探索 | ユークリッド距離 | マハラノビス距離 |
| コスト関数 | point-to-distribution | Gaussian-to-Gaussian |
| 最適化 | Gauss-Newton | Gauss-Newton |
| 共分散の使い方 | 局所平面から毎回計算 | マップに事前格納 + 適応更新 |
| 不確実性追跡 | なし | あり (uncertainty, use_num) |

---

## 2. データ構造

### 2.1 Gaussian Point

```cpp
namespace gaussian_gicp {

struct GaussianPoint {
    // 基本情報
    Eigen::Vector3d mean;           // 平均位置
    Eigen::Matrix3d cov;            // 3x3 共分散行列
    Eigen::Vector3d normal;         // 法線ベクトル（最小固有値の固有ベクトル）

    // 統計情報（AKF-LIO 由来）
    int pt_num;                     // 構成点数（マップ構築時）
    int use_num;                    // マッチング使用回数
    double uncertainty;             // 残差ベースの不確実性
    double planarity;               // 平面度 (λ1 / λ3)
    double last_update_time;        // 最終更新時刻

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace gaussian_gicp
```

### 2.2 Gaussian Voxel Map

```cpp
namespace gaussian_gicp {

class GaussianVoxelMap {
public:
    struct Config {
        double resolution = 0.5;              // ボクセル解像度
        int nearby_type = 26;                 // 近傍探索タイプ (0, 6, 18, 26)
        size_t capacity = 10000000;           // 最大ボクセル数
    };

    using KeyType = Eigen::Vector3i;
    using VoxelType = std::vector<GaussianPoint>;

    // 基本操作
    void addGaussian(const KeyType& key, const GaussianPoint& gaussian);
    bool getGaussiansInRadius(const Eigen::Vector3d& query,
                               double radius,
                               std::vector<GaussianPoint>& results) const;
    GaussianPoint& getGaussian(size_t idx);

    // 入出力
    void save(const std::string& filename) const;
    void load(const std::string& filename);

    // 統計
    size_t getNumGaussians() const;
    size_t getNumVoxels() const;

private:
    KeyType posToVoxel(const Eigen::Vector3d& pos) const;
    void generateNearbyKeys(const KeyType& key, std::vector<KeyType>& nearby) const;

    Config config_;
    std::unordered_map<KeyType, VoxelType, hash_vec<3>> voxels_;
    std::vector<KeyType> nearby_offsets_;
};

}  // namespace gaussian_gicp
```

---

## 3. オフライン処理：Gaussian Map 構築

### 3.1 Map Builder

```cpp
namespace gaussian_gicp {

class GaussianMapBuilder {
public:
    struct Config {
        double voxel_resolution = 0.5;        // ボクセル解像度
        int min_points_per_voxel = 5;         // 最小点数
        double mahalanobis_threshold = 7.82;  // マージ閾値 (χ² 3DoF 95%)
        int nearby_type = 26;                 // 近傍探索タイプ
        double init_uncertainty = 0.01;       // 初期不確実性
    };

    // 点群から Gaussian Map を構築
    GaussianVoxelMap build(const pcl::PointCloud<pcl::PointXYZ>& cloud);

    // 複数点群から増分的に構築
    void addPointCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                       const Eigen::Matrix4d& pose);
    GaussianVoxelMap getMap() const;

private:
    // 点群から Gaussian を計算
    GaussianPoint computeGaussian(const std::vector<Eigen::Vector3d>& points);

    // Pseudo-merge 処理
    void applyPseudoMerge();

    // 2つの Gaussian をマージ
    GaussianPoint mergeGaussians(const GaussianPoint& g1, const GaussianPoint& g2);

    // マハラノビス距離
    double mahalanobisDistance(const GaussianPoint& g1, const GaussianPoint& g2);

    Config config_;
    GaussianVoxelMap map_;
};

}  // namespace gaussian_gicp
```

### 3.2 Gaussian 計算アルゴリズム

```cpp
GaussianPoint GaussianMapBuilder::computeGaussian(
    const std::vector<Eigen::Vector3d>& points) {

    GaussianPoint g;

    // 1. 平均計算
    g.mean = Eigen::Vector3d::Zero();
    for (const auto& p : points) {
        g.mean += p;
    }
    g.mean /= points.size();

    // 2. 共分散計算
    g.cov = Eigen::Matrix3d::Zero();
    for (const auto& p : points) {
        Eigen::Vector3d d = p - g.mean;
        g.cov += d * d.transpose();
    }
    g.cov /= (points.size() - 1);

    // 3. 固有値分解で法線と平面度を計算
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(g.cov);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    Eigen::Matrix3d eigenvectors = solver.eigenvectors();

    // 最小固有値の固有ベクトル = 法線
    g.normal = eigenvectors.col(0);

    // 平面度 = λ1 / λ3 (小さいほど平面的)
    g.planarity = eigenvalues(0) / (eigenvalues(2) + 1e-6);

    // 4. 統計情報初期化
    g.pt_num = points.size();
    g.use_num = 0;
    g.uncertainty = config_.init_uncertainty;
    g.last_update_time = 0.0;

    return g;
}
```

### 3.3 Pseudo-merge アルゴリズム

```cpp
void GaussianMapBuilder::applyPseudoMerge() {
    for (auto& [key, gaussians] : map_.voxels_) {
        for (auto& g : gaussians) {
            // 近傍ボクセルを探索
            std::vector<KeyType> nearby_keys;
            map_.generateNearbyKeys(key, nearby_keys);

            for (const auto& neighbor_key : nearby_keys) {
                auto it = map_.voxels_.find(neighbor_key);
                if (it == map_.voxels_.end()) continue;

                for (auto& neighbor_g : it->second) {
                    if (neighbor_g.pt_num == 0) continue;  // 削除済み

                    // マハラノビス距離を計算
                    double mal_dist = mahalanobisDistance(g, neighbor_g);

                    if (mal_dist < config_.mahalanobis_threshold) {
                        // Gaussian をマージ
                        g = mergeGaussians(g, neighbor_g);
                        neighbor_g.pt_num = 0;  // 削除マーク
                    }
                }
            }
        }
    }

    // マークされた Gaussian を削除
    map_.removeMarkedGaussians();
}

GaussianPoint GaussianMapBuilder::mergeGaussians(
    const GaussianPoint& g1, const GaussianPoint& g2) {

    GaussianPoint merged;

    double w1 = static_cast<double>(g1.pt_num);
    double w2 = static_cast<double>(g2.pt_num);
    double w_sum = w1 + w2;

    // 加重平均で位置を更新
    merged.mean = (w1 * g1.mean + w2 * g2.mean) / w_sum;

    // 共分散の統合（AKF-LIO 式）
    merged.cov = (w1 / w_sum) * (g1.cov + g1.mean * g1.mean.transpose()) +
                 (w2 / w_sum) * (g2.cov + g2.mean * g2.mean.transpose()) -
                 merged.mean * merged.mean.transpose();

    // 統計情報の統合
    merged.pt_num = g1.pt_num + g2.pt_num;
    merged.use_num = g1.use_num + g2.use_num;

    // uncertainty の加重平均
    if (merged.use_num > 0) {
        double u_w1 = static_cast<double>(g1.use_num);
        double u_w2 = static_cast<double>(g2.use_num);
        merged.uncertainty = (u_w1 * g1.uncertainty + u_w2 * g2.uncertainty) /
                            (u_w1 + u_w2);
    } else {
        merged.uncertainty = (g1.uncertainty + g2.uncertainty) / 2.0;
    }

    // 法線と平面度を再計算
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(merged.cov);
    merged.normal = solver.eigenvectors().col(0);
    merged.planarity = solver.eigenvalues()(0) / (solver.eigenvalues()(2) + 1e-6);

    merged.last_update_time = std::max(g1.last_update_time, g2.last_update_time);

    return merged;
}

double GaussianMapBuilder::mahalanobisDistance(
    const GaussianPoint& g1, const GaussianPoint& g2) {

    Eigen::Vector3d d = g1.mean - g2.mean;
    Eigen::Matrix3d combined_cov = g1.cov + g2.cov;

    return d.transpose() * combined_cov.inverse() * d;
}
```

---

## 4. オンライン処理：Gaussian マッチング

### 4.1 Gaussian GICP クラス

```cpp
namespace gaussian_gicp {

class GaussianGICP {
public:
    struct Config {
        // 最適化パラメータ
        int max_iterations = 20;
        double convergence_threshold = 1e-6;

        // 対応探索パラメータ
        double max_correspondence_distance = 2.0;
        double mahalanobis_threshold = 7.82;

        // AKF-LIO 由来パラメータ
        double akf_update_alpha = 0.1;        // uncertainty 更新学習率
        double t_ratio_b = 1.0;               // 残差重み付け係数
        int min_observation_count = 5;        // 信頼できる最小観測回数
        double uncertainty_weight_scale = 1.0; // uncertainty の重み影響度

        // 並列化
        int num_threads = 4;
    };

    struct Result {
        Eigen::Matrix4d transformation;
        double fitness_score;
        int num_correspondences;
        bool converged;
        int iterations;
    };

    explicit GaussianGICP(GaussianVoxelMap* map);

    // メインのマッチング関数
    Result align(const pcl::PointCloud<pcl::PointXYZ>& source,
                 const Eigen::Matrix4d& initial_guess);

    // マッチング後のマップ更新（オプション）
    void updateMapAfterMatch(bool enable);

private:
    struct Correspondence {
        int source_idx;
        size_t target_gaussian_idx;
        GaussianPoint target_gaussian;
        double mahalanobis_distance;
    };

    // 対応点探索
    std::vector<Correspondence> findCorrespondences(
        const pcl::PointCloud<pcl::PointXYZ>& transformed_source);

    // 姿勢更新計算
    Eigen::Matrix<double, 6, 1> computeUpdate(
        const pcl::PointCloud<pcl::PointXYZ>& source,
        const std::vector<Correspondence>& correspondences,
        const Eigen::Matrix4d& current_T);

    // 適応的重み計算
    double computeAdaptiveWeight(const GaussianPoint& gaussian,
                                  double mahalanobis_dist);

    // マップ更新
    void updateGaussianUncertainty(const std::vector<Correspondence>& correspondences,
                                    const Eigen::Matrix4d& final_T,
                                    const pcl::PointCloud<pcl::PointXYZ>& source);

    // ユーティリティ
    static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);
    static Eigen::Matrix4d expSE3(const Eigen::Matrix<double, 6, 1>& xi);

    GaussianVoxelMap* map_;
    Config config_;
    bool update_map_after_match_ = false;
};

}  // namespace gaussian_gicp
```

### 4.2 対応点探索（マハラノビス距離）

```cpp
std::vector<GaussianGICP::Correspondence> GaussianGICP::findCorrespondences(
    const pcl::PointCloud<pcl::PointXYZ>& transformed_source) {

    std::vector<Correspondence> correspondences;
    correspondences.reserve(transformed_source.size());

    std::mutex mutex;

    #pragma omp parallel for num_threads(config_.num_threads)
    for (size_t i = 0; i < transformed_source.size(); ++i) {
        const auto& pt = transformed_source.points[i];
        Eigen::Vector3d query(pt.x, pt.y, pt.z);

        // Gaussian Map から近傍を探索
        std::vector<GaussianPoint> nearby_gaussians;
        std::vector<size_t> nearby_indices;
        map_->getGaussiansInRadius(query, config_.max_correspondence_distance,
                                    nearby_gaussians, nearby_indices);

        if (nearby_gaussians.empty()) continue;

        // マハラノビス距離で最近傍を選択
        double min_mal_dist = std::numeric_limits<double>::max();
        GaussianPoint best_match;
        size_t best_idx = 0;

        for (size_t j = 0; j < nearby_gaussians.size(); ++j) {
            const auto& g = nearby_gaussians[j];
            Eigen::Vector3d d = query - g.mean;

            // マハラノビス距離（共分散を考慮）
            double mal_dist = d.transpose() * g.cov.inverse() * d;

            if (mal_dist < min_mal_dist && mal_dist < config_.mahalanobis_threshold) {
                min_mal_dist = mal_dist;
                best_match = g;
                best_idx = nearby_indices[j];
            }
        }

        if (min_mal_dist < config_.mahalanobis_threshold) {
            std::lock_guard<std::mutex> lock(mutex);
            correspondences.push_back({
                static_cast<int>(i),
                best_idx,
                best_match,
                min_mal_dist
            });
        }
    }

    return correspondences;
}
```

### 4.3 Point-to-Gaussian 最適化

```cpp
Eigen::Matrix<double, 6, 1> GaussianGICP::computeUpdate(
    const pcl::PointCloud<pcl::PointXYZ>& source,
    const std::vector<Correspondence>& correspondences,
    const Eigen::Matrix4d& current_T) {

    // Gauss-Newton: H * delta = -b
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

    Eigen::Matrix3d R = current_T.block<3, 3>(0, 0);
    Eigen::Vector3d t = current_T.block<3, 1>(0, 3);

    for (const auto& corr : correspondences) {
        const auto& src_pt = source.points[corr.source_idx];
        Eigen::Vector3d p_src(src_pt.x, src_pt.y, src_pt.z);

        // 変換後の点
        Eigen::Vector3d p_transformed = R * p_src + t;

        // 残差: 変換点と Gaussian 平均の差
        Eigen::Vector3d residual = p_transformed - corr.target_gaussian.mean;

        // AKF-LIO 式の適応的重み
        double weight = computeAdaptiveWeight(
            corr.target_gaussian,
            corr.mahalanobis_distance);

        // 情報行列に uncertainty を反映
        Eigen::Matrix3d adaptive_info =
            corr.target_gaussian.cov.inverse() /
            (1.0 + config_.uncertainty_weight_scale * corr.target_gaussian.uncertainty);

        // ヤコビアン (回転と並進に対する微分)
        // J = [I | -[p_transformed]_x]  (3x6)
        Eigen::Matrix<double, 3, 6> J;
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J.block<3, 3>(0, 3) = -skewSymmetric(p_transformed);

        // H と b を累積
        H += weight * J.transpose() * adaptive_info * J;
        b += weight * J.transpose() * adaptive_info * residual;
    }

    // 正則化（数値安定性）
    H += 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();

    // 解く
    Eigen::Matrix<double, 6, 1> delta = H.ldlt().solve(-b);

    return delta;
}
```

### 4.4 適応的重み計算

```cpp
double GaussianGICP::computeAdaptiveWeight(
    const GaussianPoint& gaussian,
    double mahalanobis_dist) {

    // 1. 不確実性に基づく重み（uncertainty が高い点は重みを下げる）
    double uncertainty_weight = 1.0 / (1.0 + gaussian.uncertainty);

    // 2. 平面度に基づく重み（平面的な点を優先）
    double planarity_weight = 1.0 / (1.0 + gaussian.planarity);

    // 3. 観測回数に基づく信頼度
    double observation_weight = std::min(
        1.0,
        static_cast<double>(gaussian.pt_num) / config_.min_observation_count);

    // 4. マハラノビス距離に基づく重み（遠い対応は重みを下げる）
    double distance_weight = std::exp(-config_.t_ratio_b * mahalanobis_dist /
                                       config_.mahalanobis_threshold);

    return uncertainty_weight * planarity_weight * observation_weight * distance_weight;
}
```

### 4.5 マップ更新（オプション）

```cpp
void GaussianGICP::updateGaussianUncertainty(
    const std::vector<Correspondence>& correspondences,
    const Eigen::Matrix4d& final_T,
    const pcl::PointCloud<pcl::PointXYZ>& source) {

    if (!update_map_after_match_) return;

    Eigen::Matrix3d R = final_T.block<3, 3>(0, 0);
    Eigen::Vector3d t = final_T.block<3, 1>(0, 3);

    for (const auto& corr : correspondences) {
        // 最終残差を計算
        const auto& src_pt = source.points[corr.source_idx];
        Eigen::Vector3d p_src(src_pt.x, src_pt.y, src_pt.z);
        Eigen::Vector3d transformed_pt = R * p_src + t;

        double residual = (transformed_pt - corr.target_gaussian.mean).norm();

        // マップの Gaussian を更新
        GaussianPoint& map_g = map_->getGaussian(corr.target_gaussian_idx);

        // uncertainty の適応的更新（AKF-LIO 式）
        double alpha = config_.akf_update_alpha;
        map_g.uncertainty = (1.0 - alpha) * map_g.uncertainty +
                           alpha * residual * residual;

        map_g.use_num++;
    }
}
```

---

## 5. ファイル入出力

### 5.1 Gaussian Map ファイル形式

```cpp
// ヘッダー
struct GaussianMapHeader {
    uint32_t magic = 0x474D4150;  // "GMAP"
    uint32_t version = 1;
    double voxel_resolution;
    uint64_t num_gaussians;
    uint64_t num_voxels;
};

// 各 Gaussian (80 bytes)
// - mean: 3 x double = 24 bytes
// - cov: 9 x double = 72 bytes (row-major, symmetric なので 6 でも可)
// - normal: 3 x double = 24 bytes
// - pt_num: int = 4 bytes
// - use_num: int = 4 bytes
// - uncertainty: double = 8 bytes
// - planarity: double = 8 bytes
```

### 5.2 保存・読み込み実装

```cpp
void GaussianVoxelMap::save(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);

    GaussianMapHeader header;
    header.voxel_resolution = config_.resolution;
    header.num_gaussians = getNumGaussians();
    header.num_voxels = getNumVoxels();
    ofs.write(reinterpret_cast<char*>(&header), sizeof(header));

    for (const auto& [key, gaussians] : voxels_) {
        for (const auto& g : gaussians) {
            ofs.write(reinterpret_cast<const char*>(g.mean.data()),
                      3 * sizeof(double));
            ofs.write(reinterpret_cast<const char*>(g.cov.data()),
                      9 * sizeof(double));
            ofs.write(reinterpret_cast<const char*>(g.normal.data()),
                      3 * sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&g.pt_num), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&g.use_num), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&g.uncertainty),
                      sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&g.planarity),
                      sizeof(double));
        }
    }
}

void GaussianVoxelMap::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);

    GaussianMapHeader header;
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.magic != 0x474D4150) {
        throw std::runtime_error("Invalid Gaussian Map file");
    }

    config_.resolution = header.voxel_resolution;

    for (uint64_t i = 0; i < header.num_gaussians; ++i) {
        GaussianPoint g;
        ifs.read(reinterpret_cast<char*>(g.mean.data()), 3 * sizeof(double));
        ifs.read(reinterpret_cast<char*>(g.cov.data()), 9 * sizeof(double));
        ifs.read(reinterpret_cast<char*>(g.normal.data()), 3 * sizeof(double));
        ifs.read(reinterpret_cast<char*>(&g.pt_num), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&g.use_num), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&g.uncertainty), sizeof(double));
        ifs.read(reinterpret_cast<char*>(&g.planarity), sizeof(double));

        KeyType key = posToVoxel(g.mean);
        voxels_[key].push_back(g);
    }
}
```

---

## 6. ROS2 インターフェース

### 6.1 ローカライゼーションノード

```cpp
class GaussianLocalizationNode : public rclcpp::Node {
public:
    GaussianLocalizationNode() : Node("gaussian_localization") {
        // パラメータ宣言
        declare_parameter("map_file", "");
        declare_parameter("voxel_resolution", 0.5);
        declare_parameter("max_iterations", 20);
        declare_parameter("mahalanobis_threshold", 7.82);
        declare_parameter("akf_update_alpha", 0.1);
        declare_parameter("update_map", false);

        // Gaussian Map ロード
        std::string map_file = get_parameter("map_file").as_string();
        map_ = std::make_shared<GaussianVoxelMap>();
        map_->load(map_file);
        RCLCPP_INFO(get_logger(), "Loaded Gaussian Map with %zu gaussians",
                    map_->getNumGaussians());

        // マッチャー初期化
        GaussianGICP::Config config;
        config.max_iterations = get_parameter("max_iterations").as_int();
        config.mahalanobis_threshold = get_parameter("mahalanobis_threshold").as_double();
        config.akf_update_alpha = get_parameter("akf_update_alpha").as_double();

        matcher_ = std::make_shared<GaussianGICP>(map_.get());
        matcher_->setConfig(config);
        matcher_->updateMapAfterMatch(get_parameter("update_map").as_bool());

        // サブスクライバ
        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "cloud", rclcpp::SensorDataQoS(),
            std::bind(&GaussianLocalizationNode::cloudCallback, this,
                      std::placeholders::_1));

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10,
            std::bind(&GaussianLocalizationNode::odomCallback, this,
                      std::placeholders::_1));

        // パブリッシャ
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            "localized_pose", 10);

        pose_cov_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "localized_pose_cov", 10);
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void publishPose(const Eigen::Matrix4d& T, const rclcpp::Time& stamp);

    std::shared_ptr<GaussianVoxelMap> map_;
    std::shared_ptr<GaussianGICP> matcher_;

    // 最新オドメトリ
    std::mutex odom_mutex_;
    nav_msgs::msg::Odometry latest_odom_;
    bool odom_received_ = false;
};
```

### 6.2 マップ構築ノード

```cpp
class GaussianMapBuilderNode : public rclcpp::Node {
public:
    GaussianMapBuilderNode() : Node("gaussian_map_builder") {
        // パラメータ
        declare_parameter("output_file", "gaussian_map.gmap");
        declare_parameter("voxel_resolution", 0.5);
        declare_parameter("min_points_per_voxel", 5);
        declare_parameter("mahalanobis_threshold", 7.82);

        // Builder 設定
        GaussianMapBuilder::Config config;
        config.voxel_resolution = get_parameter("voxel_resolution").as_double();
        config.min_points_per_voxel = get_parameter("min_points_per_voxel").as_int();
        config.mahalanobis_threshold = get_parameter("mahalanobis_threshold").as_double();

        builder_ = std::make_shared<GaussianMapBuilder>(config);

        // サブスクライバ
        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "cloud", rclcpp::SensorDataQoS(),
            std::bind(&GaussianMapBuilderNode::cloudCallback, this,
                      std::placeholders::_1));

        // 保存サービス
        save_srv_ = create_service<std_srvs::srv::Trigger>(
            "save_map",
            std::bind(&GaussianMapBuilderNode::saveMapCallback, this,
                      std::placeholders::_1, std::placeholders::_2));
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void saveMapCallback(const std_srvs::srv::Trigger::Request::SharedPtr req,
                         std_srvs::srv::Trigger::Response::SharedPtr res);

    std::shared_ptr<GaussianMapBuilder> builder_;
};
```

---

## 7. パラメータ設定

### 7.1 YAML 設定ファイル

```yaml
gaussian_localization:
  ros__parameters:
    # マップ
    map_file: "/path/to/gaussian_map.gmap"

    # ボクセル設定
    voxel_resolution: 0.5
    nearby_type: 26  # 0, 6, 18, 26

    # マッチング
    max_iterations: 20
    convergence_threshold: 1.0e-6
    max_correspondence_distance: 2.0
    mahalanobis_threshold: 7.82  # χ² 3DoF 95%

    # AKF-LIO 由来
    akf_update_alpha: 0.1
    t_ratio_b: 1.0
    min_observation_count: 5
    uncertainty_weight_scale: 1.0
    init_uncertainty: 0.01

    # オプション
    update_map: false  # オンラインマップ更新
    num_threads: 4

gaussian_map_builder:
  ros__parameters:
    output_file: "gaussian_map.gmap"
    voxel_resolution: 0.5
    min_points_per_voxel: 5
    mahalanobis_threshold: 7.82
    nearby_type: 26
```

---

## 8. ディレクトリ構造

```
small_gicp/
├── include/small_gicp/
│   ├── gaussian/
│   │   ├── gaussian_point.hpp          # Gaussian 点の定義
│   │   ├── gaussian_voxel_map.hpp      # Gaussian Map (iVox ベース)
│   │   ├── gaussian_map_builder.hpp    # オフライン構築
│   │   └── gaussian_gicp.hpp           # マッチングアルゴリズム
│   └── ...
├── src/
│   └── gaussian/
│       ├── gaussian_voxel_map.cpp
│       ├── gaussian_map_builder.cpp
│       └── gaussian_gicp.cpp
├── ros/
│   ├── gaussian_localization_node.cpp
│   └── gaussian_map_builder_node.cpp
└── launch/
    ├── gaussian_localization.launch.py
    └── gaussian_map_builder.launch.py
```

---

## 9. 移動体・外れ値対策

AKF-LIO の移動体対策を Gaussian GICP に統合する。

### 9.1 多段階リジェクション

```cpp
struct OutlierRejectionConfig {
    // 幾何学的リジェクション
    double min_eigenvalue_ratio = 0.04;    // 最小固有値比（平面度）
    double max_tangent_distance = 1.0;     // 接平面内最大距離
    double max_normal_distance_ratio = 1.0/9.0; // 法線方向最大距離比

    // 統計的リジェクション
    double mahalanobis_threshold = 7.82;   // χ² 3DoF 95%
    double max_residual = 0.5;             // 最大残差 [m]

    // 時間的一貫性
    int min_consistent_frames = 3;         // 最小一貫フレーム数
};
```

### 9.2 幾何学的リジェクション

```cpp
bool GaussianGICP::isValidCorrespondence(
    const Eigen::Vector3d& query,
    const GaussianPoint& gaussian) {

    Eigen::Vector3d diff = query - gaussian.mean;

    // 固有値分解（共分散から平面性を確認）
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(gaussian.cov);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    Eigen::Matrix3d eigenvectors = solver.eigenvectors();

    // 1. 平面度チェック（λ2 が十分大きいか）
    if (eigenvalues(1) < config_.min_eigenvalue_ratio) {
        return false;  // 平面として信頼できない
    }

    // 2. 接平面内距離チェック
    Eigen::Vector3d normal = eigenvectors.col(0);
    Eigen::Vector3d tangent1 = eigenvectors.col(1);
    Eigen::Vector3d tangent2 = eigenvectors.col(2);

    double dist_tangent = std::pow(tangent1.dot(diff), 2) / eigenvalues(1) +
                          std::pow(tangent2.dot(diff), 2) / eigenvalues(2);

    if (dist_tangent > config_.max_tangent_distance * config_.max_tangent_distance) {
        return false;  // 接平面内で離れすぎ
    }

    // 3. 法線方向距離チェック（距離依存閾値）
    double dist_normal = std::abs(normal.dot(diff));
    double distance_from_sensor = query.norm();
    double max_normal_dist = config_.max_normal_distance_ratio * std::sqrt(distance_from_sensor);

    if (dist_normal > max_normal_dist) {
        return false;  // 法線方向に離れすぎ（移動体の可能性）
    }

    return true;
}
```

### 9.3 指数重み付け（AKF-LIO 式）

```cpp
double GaussianGICP::computeAdaptiveWeight(
    const GaussianPoint& gaussian,
    double point_to_plane_dist,
    double mahalanobis_dist) {

    // 1. 残差ベースの uncertainty（AKF-LIO 式）
    double residual_uncertainty = point_to_plane_dist * point_to_plane_dist;

    // 2. 指数重み付け（移動体の影響を指数的に抑制）
    double exp_weight = std::exp(-config_.t_ratio_b * residual_uncertainty);

    // 3. マップ側の uncertainty による重み
    double map_uncertainty_weight = 1.0 / (1.0 + gaussian.uncertainty);

    // 4. 平面の厚さ（薄いほど信頼できる）
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(gaussian.cov);
    double thickness = solver.eigenvalues()(0);  // 最小固有値
    double thickness_weight = 1.0 / (1.0 + thickness);

    // 5. 観測回数による信頼度
    double observation_weight = std::min(
        1.0,
        static_cast<double>(gaussian.pt_num) / config_.min_observation_count);

    return exp_weight * map_uncertainty_weight * thickness_weight * observation_weight;
}
```

### 9.4 マップ Gaussian の Uncertainty 更新

```cpp
void GaussianGICP::updateMapUncertainty(
    const std::vector<Correspondence>& correspondences,
    const Eigen::Matrix4d& final_T,
    const pcl::PointCloud<pcl::PointXYZ>& source) {

    for (const auto& corr : correspondences) {
        // 最終残差を計算
        Eigen::Vector3d p_src(source.points[corr.source_idx].x,
                               source.points[corr.source_idx].y,
                               source.points[corr.source_idx].z);
        Eigen::Vector3d transformed = final_T.block<3,3>(0,0) * p_src +
                                      final_T.block<3,1>(0,3);

        // Point-to-plane 残差
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(corr.target_gaussian.cov);
        Eigen::Vector3d normal = solver.eigenvectors().col(0);
        Eigen::Vector3d diff = transformed - corr.target_gaussian.mean;
        double p2pl = std::abs(normal.dot(diff));

        // マップの Gaussian を更新
        GaussianPoint& map_g = map_->getGaussian(corr.target_gaussian_idx);

        // Uncertainty の適応的更新（AKF-LIO 式）
        // 新しい残差² と既存の uncertainty の加重平均
        double new_uncertainty = p2pl * p2pl;
        double w_old = static_cast<double>(map_g.use_num);
        double w_new = 1.0;

        if (w_old + w_new > 0) {
            map_g.uncertainty = (w_old * map_g.uncertainty + w_new * new_uncertainty) /
                               (w_old + w_new);
        }

        map_g.use_num++;

        // 最大 use_num の制限（古い情報の影響を制限）
        if (map_g.use_num > config_.max_use_num) {
            map_g.use_num = config_.max_use_num;
        }
    }
}
```

### 9.5 動的物体の間接的除去

移動体は以下のメカニズムで間接的に除去される：

```
┌─────────────────────────────────────────────────────────────────┐
│  移動体からの点群                                                │
├─────────────────────────────────────────────────────────────────┤
│  1. マップとの対応探索 → 対応が見つからない or 大きなマハラノビス距離  │
│  2. 対応が見つかった場合 → 大きな point-to-plane 残差              │
│  3. 大きな残差 → 高い uncertainty (p2pl²)                        │
│  4. 高い uncertainty → 指数的に低い重み exp(-t_ratio_b * uncertainty) │
│  5. 低い重み → 最適化への寄与が minimal                           │
└─────────────────────────────────────────────────────────────────┘
```

### 9.6 パラメータ設定

```yaml
gaussian_gicp:
  ros__parameters:
    # 幾何学的リジェクション
    min_eigenvalue_ratio: 0.04
    max_tangent_distance: 1.0
    max_normal_distance_ratio: 0.111  # 1/9

    # AKF 重み付け
    t_ratio_b: 1.0                    # 指数重み係数
    init_uncertainty: 0.01            # 初期 uncertainty
    max_use_num: 100                  # 最大使用回数

    # 統計的リジェクション
    mahalanobis_threshold: 7.82       # χ² 3DoF 95%
    max_residual: 0.5                 # 最大残差 [m]
```

---

# Advanced Features

---

## 10. 変化検出（移動体・新規物体）

残差ベースのクラスタリングにより、移動体および既存マップにない新規物体を検出する。

### 10.1 点群分類

```
┌─────────────────────────────────────────────────────────────────┐
│  入力: リアルタイム点群 + Gaussian Map                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  各点について:                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 対応あり & 低残差  → MATCHED（既存環境）                      │ │
│  │ 対応あり & 高残差  → MOVING_OBJECT（移動体）                  │ │
│  │ 対応なし          → NEW_OBJECT（新規物体）                   │ │
│  │ マップ範囲外       → OUT_OF_MAP                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                         ↓                                        │
│  高残差 + 対応なし の点をクラスタリング                           │
│                         ↓                                        │
│  時間的一貫性でフィルタ（ノイズ除去）                             │
│                         ↓                                        │
│  出力: 移動体 / 新規物体 のクラスタ                               │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 データ構造

```cpp
struct PointClassification {
    enum class Category {
        MATCHED,           // 既存マップと一致
        MOVING_OBJECT,     // 移動体（高残差）
        NEW_OBJECT,        // 新規物体（マップにない）
        OUT_OF_MAP         // マップ範囲外
    };

    int point_idx;
    Category category;
    double confidence;
    double residual;
    Eigen::Vector3d position;
};

struct DetectedCluster {
    pcl::PointCloud<pcl::PointXYZ> points;
    Eigen::Vector3d centroid;
    Eigen::Vector3d bbox_min, bbox_max;
    PointClassification::Category type;
    double confidence;
    int track_id;
    int consistent_frames;
};
```

### 10.3 変化検出クラス

```cpp
class ChangeDetector {
public:
    struct Config {
        // 残差閾値
        double match_residual_threshold = 0.1;     // マッチ判定
        double moving_residual_threshold = 0.3;    // 移動体判定
        double search_radius = 2.0;                // 対応探索半径

        // クラスタリング
        double cluster_tolerance = 0.5;            // クラスタ間距離
        int min_cluster_size = 10;                 // 最小点数

        // 時間的フィルタ
        int min_consistent_frames = 3;             // 最小一貫フレーム数
        double tracking_distance_threshold = 1.0;  // トラッキング距離

        // マップ境界
        double map_boundary_margin = 1.0;
    };

    // 点群を分類
    std::vector<PointClassification> classifyPoints(
        const pcl::PointCloud<pcl::PointXYZ>& scan,
        const Eigen::Matrix4d& pose);

    // クラスタを抽出
    std::vector<DetectedCluster> detectClusters(
        const std::vector<PointClassification>& classifications,
        const pcl::PointCloud<pcl::PointXYZ>& scan);

    // 時間的トラッキング更新
    void updateTracking(const std::vector<DetectedCluster>& clusters,
                        double timestamp);

    // 確定した検出結果を取得
    std::vector<DetectedCluster> getConfirmedDetections();

private:
    GaussianVoxelMap* map_;
    Config config_;
    std::vector<Track> tracks_;
};
```

### 10.4 分類アルゴリズム

```cpp
std::vector<PointClassification> ChangeDetector::classifyPoints(
    const pcl::PointCloud<pcl::PointXYZ>& scan,
    const Eigen::Matrix4d& pose) {

    std::vector<PointClassification> results;
    results.reserve(scan.size());

    Eigen::Matrix3d R = pose.block<3,3>(0,0);
    Eigen::Vector3d t = pose.block<3,1>(0,3);

    for (size_t i = 0; i < scan.size(); ++i) {
        Eigen::Vector3d pt_local(scan.points[i].x,
                                  scan.points[i].y,
                                  scan.points[i].z);
        Eigen::Vector3d pt_world = R * pt_local + t;

        PointClassification cls;
        cls.point_idx = i;
        cls.position = pt_world;

        // 1. マップ範囲チェック
        if (!map_->isInBounds(pt_world, config_.map_boundary_margin)) {
            cls.category = PointClassification::Category::OUT_OF_MAP;
            cls.confidence = 1.0;
            cls.residual = -1.0;
            results.push_back(cls);
            continue;
        }

        // 2. 近傍 Gaussian を探索
        std::vector<GaussianPoint> nearby;
        map_->getGaussiansInRadius(pt_world, config_.search_radius, nearby);

        if (nearby.empty()) {
            // 対応なし → 新規物体
            cls.category = PointClassification::Category::NEW_OBJECT;
            cls.confidence = 0.9;
            cls.residual = config_.search_radius;
            results.push_back(cls);
            continue;
        }

        // 3. 最近傍との残差を計算
        double min_residual = std::numeric_limits<double>::max();
        for (const auto& g : nearby) {
            // Point-to-plane 残差
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(g.cov);
            Eigen::Vector3d normal = solver.eigenvectors().col(0);
            double residual = std::abs(normal.dot(pt_world - g.mean));
            min_residual = std::min(min_residual, residual);
        }

        cls.residual = min_residual;

        // 4. 残差に基づいて分類
        if (min_residual < config_.match_residual_threshold) {
            cls.category = PointClassification::Category::MATCHED;
            cls.confidence = 1.0 - min_residual / config_.match_residual_threshold;
        } else if (min_residual < config_.moving_residual_threshold) {
            cls.category = PointClassification::Category::MOVING_OBJECT;
            cls.confidence = 0.7;
        } else {
            cls.category = PointClassification::Category::NEW_OBJECT;
            cls.confidence = 0.9;
        }

        results.push_back(cls);
    }

    return results;
}
```

### 10.5 クラスタリングとトラッキング

```cpp
std::vector<DetectedCluster> ChangeDetector::detectClusters(
    const std::vector<PointClassification>& classifications,
    const pcl::PointCloud<pcl::PointXYZ>& scan) {

    // 1. 非マッチ点を抽出
    pcl::PointCloud<pcl::PointXYZ> candidate_points;
    std::vector<PointClassification::Category> point_types;

    for (const auto& cls : classifications) {
        if (cls.category == PointClassification::Category::MOVING_OBJECT ||
            cls.category == PointClassification::Category::NEW_OBJECT) {
            candidate_points.push_back(scan.points[cls.point_idx]);
            point_types.push_back(cls.category);
        }
    }

    if (candidate_points.size() < config_.min_cluster_size) {
        return {};
    }

    // 2. ユークリッドクラスタリング
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(candidate_points.makeShared());

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(config_.cluster_tolerance);
    ec.setMinClusterSize(config_.min_cluster_size);
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(candidate_points.makeShared());
    ec.extract(cluster_indices);

    // 3. クラスタを構造体に変換
    std::vector<DetectedCluster> clusters;
    for (const auto& indices : cluster_indices) {
        DetectedCluster cluster;

        // 点群とタイプの集計
        int moving_count = 0, new_count = 0;
        for (int idx : indices.indices) {
            cluster.points.push_back(candidate_points.points[idx]);
            if (point_types[idx] == PointClassification::Category::MOVING_OBJECT) {
                moving_count++;
            } else {
                new_count++;
            }
        }

        // 多数決でタイプ決定
        cluster.type = (moving_count > new_count) ?
            PointClassification::Category::MOVING_OBJECT :
            PointClassification::Category::NEW_OBJECT;

        // 重心と BoundingBox
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        cluster.bbox_min = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::max());
        cluster.bbox_max = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::lowest());

        for (const auto& pt : cluster.points) {
            Eigen::Vector3d p(pt.x, pt.y, pt.z);
            sum += p;
            cluster.bbox_min = cluster.bbox_min.cwiseMin(p);
            cluster.bbox_max = cluster.bbox_max.cwiseMax(p);
        }
        cluster.centroid = sum / cluster.points.size();

        cluster.confidence = static_cast<double>(
            std::max(moving_count, new_count)) / indices.indices.size();

        clusters.push_back(cluster);
    }

    return clusters;
}
```

### 10.6 出力インターフェース

```cpp
void ChangeDetector::publishResults(
    const std::vector<PointClassification>& classifications,
    const std::vector<DetectedCluster>& clusters,
    const pcl::PointCloud<pcl::PointXYZ>& scan,
    const rclcpp::Time& stamp) {

    // 1. 分類済み点群（色分け）
    pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;
    for (const auto& cls : classifications) {
        pcl::PointXYZRGB pt;
        pt.x = scan.points[cls.point_idx].x;
        pt.y = scan.points[cls.point_idx].y;
        pt.z = scan.points[cls.point_idx].z;

        switch (cls.category) {
            case PointClassification::Category::MATCHED:
                pt.r = 0; pt.g = 255; pt.b = 0;    // 緑
                break;
            case PointClassification::Category::MOVING_OBJECT:
                pt.r = 255; pt.g = 0; pt.b = 0;    // 赤
                break;
            case PointClassification::Category::NEW_OBJECT:
                pt.r = 0; pt.g = 0; pt.b = 255;    // 青
                break;
            case PointClassification::Category::OUT_OF_MAP:
                pt.r = 128; pt.g = 128; pt.b = 128; // 灰
                break;
        }
        colored_cloud.push_back(pt);
    }

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(colored_cloud, cloud_msg);
    cloud_msg.header.stamp = stamp;
    cloud_msg.header.frame_id = "map";
    classified_cloud_pub_->publish(cloud_msg);

    // 2. 検出クラスタの BoundingBox
    vision_msgs::msg::Detection3DArray detections;
    detections.header.stamp = stamp;
    detections.header.frame_id = "map";

    for (const auto& cluster : clusters) {
        if (cluster.consistent_frames < config_.min_consistent_frames) continue;

        vision_msgs::msg::Detection3D det;
        det.bbox.center.position.x = cluster.centroid.x();
        det.bbox.center.position.y = cluster.centroid.y();
        det.bbox.center.position.z = cluster.centroid.z();
        det.bbox.size.x = cluster.bbox_max.x() - cluster.bbox_min.x();
        det.bbox.size.y = cluster.bbox_max.y() - cluster.bbox_min.y();
        det.bbox.size.z = cluster.bbox_max.z() - cluster.bbox_min.z();

        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id =
            (cluster.type == PointClassification::Category::MOVING_OBJECT) ?
            "moving" : "new";
        hyp.hypothesis.score = cluster.confidence;
        det.results.push_back(hyp);

        detections.detections.push_back(det);
    }

    detections_pub_->publish(detections);
}
```

---

## 11. シームレスローカライゼーション

マップ内外を問わず連続的に自己位置推定を行う。

### 11.1 アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                    Seamless Localization                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [LiDAR点群] → [LIO オドメトリ] ─────────────────┐               │
│                      ↓                          │               │
│              [Gaussian Map 照合]                 │               │
│                      ↓                          ↓               │
│              ┌──────────────────┐      ┌──────────────┐         │
│              │  マップ内         │      │  マップ外    │         │
│              │  (GICP マッチング) │      │  (LIO のみ)  │         │
│              └────────┬─────────┘      └──────┬───────┘         │
│                       ↓                        ↓                │
│              ┌──────────────────────────────────┐               │
│              │     適応的フュージョン            │               │
│              │  pose = α * map_pose             │               │
│              │       + (1-α) * lio_pose         │               │
│              │  (α = map_confidence)            │               │
│              └──────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 データ構造

```cpp
struct LocalizationResult {
    Eigen::Matrix4d pose;
    Eigen::Matrix<double, 6, 6> covariance;

    double map_confidence;      // 0.0 (マップ外) ~ 1.0 (マップ内)
    bool in_map;

    int matched_points;
    int total_points;

    enum class Mode {
        MAP_BASED,      // マップベースローカライゼーション
        LIO_ONLY,       // LIO のみ
        TRANSITIONING   // 遷移中
    } mode;
};

struct MapBoundaryStatus {
    enum class Zone {
        INSIDE,         // 完全にマップ内
        BOUNDARY,       // 境界付近
        OUTSIDE,        // 完全にマップ外
        RETURNING       // マップに戻りつつある
    } zone;

    double coverage_ratio;
    Eigen::Vector3d nearest_map_point;
};
```

### 11.3 シームレスローカライザ

```cpp
class SeamlessLocalizer {
public:
    struct Config {
        // マップ内判定
        double map_coverage_threshold = 0.3;    // マップ内判定閾値
        double min_match_ratio = 0.2;           // 最小マッチ率

        // 遷移制御
        double smooth_transition_rate = 0.1;    // 信頼度変化率
        double hysteresis_margin = 0.05;        // ヒステリシス

        // 不確実性
        double uncertainty_scale_outside = 2.0; // マップ外の不確実性スケール

        // マップ拡張（オプション）
        bool enable_map_extension = false;
        double min_lio_confidence = 0.8;
    };

    LocalizationResult localize(
        const pcl::PointCloud<pcl::PointXYZ>& scan,
        const Eigen::Matrix4d& lio_pose,
        const Eigen::Matrix<double, 6, 6>& lio_covariance);

private:
    // 姿勢のフュージョン（SE3 上での補間）
    Eigen::Matrix4d fusePoses(
        const Eigen::Matrix4d& map_pose,
        const Eigen::Matrix4d& lio_pose,
        double alpha);

    // 共分散のフュージョン（情報行列で加重平均）
    Eigen::Matrix<double, 6, 6> fuseCovariances(
        const Eigen::Matrix<double, 6, 6>& map_cov,
        const Eigen::Matrix<double, 6, 6>& lio_cov,
        double alpha);

    // マップ境界状態の判定
    MapBoundaryStatus checkMapBoundary(
        const Eigen::Vector3d& position,
        const pcl::PointCloud<pcl::PointXYZ>& scan);

    std::shared_ptr<GaussianGICP> gaussian_gicp_;
    Config config_;
    double map_confidence_ = 0.0;  // 状態として保持
};
```

### 11.4 ローカライゼーションアルゴリズム

```cpp
LocalizationResult SeamlessLocalizer::localize(
    const pcl::PointCloud<pcl::PointXYZ>& scan,
    const Eigen::Matrix4d& lio_pose,
    const Eigen::Matrix<double, 6, 6>& lio_covariance) {

    LocalizationResult result;
    result.total_points = scan.size();

    // 1. Gaussian Map とのマッチングを試行
    auto gicp_result = gaussian_gicp_->align(scan, lio_pose);
    result.matched_points = gicp_result.num_correspondences;

    // 2. マップカバレッジ（マッチ率）を計算
    double match_ratio = static_cast<double>(gicp_result.num_correspondences) /
                        static_cast<double>(scan.size());

    // 3. マップ内/外の判定（ヒステリシス付き）
    double threshold = config_.map_coverage_threshold;
    if (result.in_map) {
        threshold -= config_.hysteresis_margin;  // マップ内なら閾値を下げる
    } else {
        threshold += config_.hysteresis_margin;  // マップ外なら閾値を上げる
    }

    result.in_map = (match_ratio >= threshold) && gicp_result.converged;

    // 4. 信頼度の滑らかな更新
    double target_confidence = result.in_map ?
        std::min(1.0, match_ratio / config_.map_coverage_threshold) : 0.0;

    map_confidence_ += config_.smooth_transition_rate *
                       (target_confidence - map_confidence_);
    map_confidence_ = std::clamp(map_confidence_, 0.0, 1.0);
    result.map_confidence = map_confidence_;

    // 5. モード決定
    if (map_confidence_ > 0.8) {
        result.mode = LocalizationResult::Mode::MAP_BASED;
    } else if (map_confidence_ < 0.2) {
        result.mode = LocalizationResult::Mode::LIO_ONLY;
    } else {
        result.mode = LocalizationResult::Mode::TRANSITIONING;
    }

    // 6. 姿勢と共分散のフュージョン
    if (map_confidence_ > 0.01 && gicp_result.converged) {
        result.pose = fusePoses(
            gicp_result.transformation,
            lio_pose,
            map_confidence_);

        result.covariance = fuseCovariances(
            gicp_result.covariance,
            lio_covariance,
            map_confidence_);
    } else {
        // 完全にマップ外
        result.pose = lio_pose;
        result.covariance = lio_covariance * config_.uncertainty_scale_outside;
    }

    return result;
}
```

### 11.5 姿勢フュージョン

```cpp
Eigen::Matrix4d SeamlessLocalizer::fusePoses(
    const Eigen::Matrix4d& map_pose,
    const Eigen::Matrix4d& lio_pose,
    double alpha) {

    // 回転: 球面線形補間 (SLERP)
    Eigen::Quaterniond q_map(map_pose.block<3,3>(0,0));
    Eigen::Quaterniond q_lio(lio_pose.block<3,3>(0,0));
    Eigen::Quaterniond q_fused = q_lio.slerp(alpha, q_map);

    // 並進: 線形補間
    Eigen::Vector3d t_fused =
        (1.0 - alpha) * lio_pose.block<3,1>(0,3) +
        alpha * map_pose.block<3,1>(0,3);

    Eigen::Matrix4d fused = Eigen::Matrix4d::Identity();
    fused.block<3,3>(0,0) = q_fused.toRotationMatrix();
    fused.block<3,1>(0,3) = t_fused;

    return fused;
}

Eigen::Matrix<double, 6, 6> SeamlessLocalizer::fuseCovariances(
    const Eigen::Matrix<double, 6, 6>& map_cov,
    const Eigen::Matrix<double, 6, 6>& lio_cov,
    double alpha) {

    // 情報行列（逆共分散）での加重平均
    Eigen::Matrix<double, 6, 6> map_info = map_cov.inverse();
    Eigen::Matrix<double, 6, 6> lio_info = lio_cov.inverse();

    Eigen::Matrix<double, 6, 6> fused_info =
        alpha * map_info + (1.0 - alpha) * lio_info;

    return fused_info.inverse();
}
```

### 11.6 オンラインマップ拡張（オプション）

```cpp
void SeamlessLocalizer::extendMap(
    const pcl::PointCloud<pcl::PointXYZ>& scan,
    const Eigen::Matrix4d& pose,
    const LocalizationResult& loc_result) {

    if (!config_.enable_map_extension) return;
    if (loc_result.in_map) return;
    if (lio_confidence_ < config_.min_lio_confidence) return;

    // 新規エリアを Gaussian Map に追加
    pcl::PointCloud<pcl::PointXYZ> transformed;
    pcl::transformPointCloud(scan, transformed, pose);

    map_builder_->addPointCloud(transformed, pose);

    RCLCPP_INFO(node_->get_logger(),
        "Extended map: +%zu points at (%.1f, %.1f)",
        scan.size(), pose(0,3), pose(1,3));
}
```

### 11.7 パラメータ設定

```yaml
seamless_localization:
  ros__parameters:
    # マップ内判定
    map_coverage_threshold: 0.3
    min_match_ratio: 0.2
    hysteresis_margin: 0.05

    # 遷移制御
    smooth_transition_rate: 0.1

    # 不確実性
    uncertainty_scale_outside: 2.0

    # マップ拡張
    enable_map_extension: false
    min_lio_confidence: 0.8

    # Gaussian GICP 設定（継承）
    voxel_resolution: 0.5
    mahalanobis_threshold: 7.82
```

---

## 12. iVox 実装比較

AKF-LIO、Faster-LIO、small_gicp における iVox（Incremental Voxel）の使い方の違いを整理する。

### 12.1 各ライブラリの概要

| 項目 | AKF-LIO | Faster-LIO | small_gicp |
|------|---------|------------|------------|
| ベース | Faster-LIO | オリジナル | オリジナル |
| 目的 | LIO + Gaussian Map | LIO（高速化） | 点群レジストレーション |
| iVox 役割 | Gaussian 格納 + 近傍探索 | 点群格納 + 近傍探索 | ボクセル化 + 共分散計算 |

### 12.2 Faster-LIO の iVox

```cpp
// Faster-LIO: 点群を直接格納し、KNN 探索を高速化
template <typename PointT>
class IVox {
    // ボクセル内に点群を格納
    std::unordered_map<KeyType, NodeType> grids_map_;

    // 近傍探索（ユークリッド距離）
    bool GetClosestPoint(const PointT& pt, PointVector& closest_pt,
                         int max_num, double max_range);
};
```

**特徴:**
- 点群を直接格納
- LRU キャッシュで古いボクセルを削除
- ユークリッド距離ベースの KNN
- Point-to-Plane 残差計算は探索後に別途実行

### 12.3 AKF-LIO の iVox（Gaussian 拡張）

```cpp
// AKF-LIO: Gaussian 情報を持つ点を格納
struct PointXYZIRTtimeRing {
    PCL_ADD_POINT4D;
    float intensity;
    double time;

    // Gaussian 情報
    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
    int pt_num;
    int use_num;
    double uncertainty;  // 残差ベース
};

// 探索時にマハラノビス距離を使用
void KNNPointMAL(std::vector<DistPoint>& candidates,
                 const PointType& pt, int max_num, double max_range) {
    for (auto& p : points_) {
        // マハラノビス距離で評価
        double mal_dist = computeMahalanobis(pt, p);
        if (mal_dist < threshold) {
            candidates.push_back({p, mal_dist});
        }
    }
}
```

**特徴:**
- 点に Gaussian 情報（共分散、uncertainty）を付加
- **マハラノビス距離**による対応点探索
- Pseudo-merge で近い Gaussian を統合
- 残差ベースの uncertainty 更新

### 12.4 small_gicp の iVox（GaussianVoxelMap）

```cpp
// small_gicp: ボクセル単位で共分散を計算
class GaussianVoxelMap {
    struct VoxelInfo {
        Eigen::Vector4d mean;      // ボクセル内点群の平均
        Eigen::Matrix4d cov;       // ボクセル内点群の共分散
        int num_points;            // 点数
    };

    std::unordered_map<VoxelKey, VoxelInfo> voxels_;

    // VGICP 用の分布を取得
    bool get_voxel(const Eigen::Vector4d& pt, VoxelInfo* voxel) const;
};
```

**特徴:**
- ボクセル = 1 Gaussian（ボクセル内の統計量）
- レジストレーション専用（LIO なし）
- ボクセル解像度が Gaussian のサイズを決定
- 動的更新なし（バッチ処理）

### 12.5 主要な違いまとめ

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        iVox の使い方比較                                 │
├───────────────┬─────────────────┬─────────────────┬─────────────────────┤
│               │  Faster-LIO     │  AKF-LIO        │  small_gicp         │
├───────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ 格納データ     │ 点群            │ Gaussian 点     │ ボクセル統計量       │
├───────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ 距離関数       │ ユークリッド     │ マハラノビス     │ ユークリッド         │
├───────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ 共分散計算     │ 探索後に毎回    │ 事前計算+更新    │ ボクセル構築時       │
├───────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ 動的更新       │ LRU 削除のみ    │ Pseudo-merge    │ なし                │
│               │                 │ + Uncertainty   │                     │
├───────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ Gaussian 粒度  │ N/A             │ 点単位          │ ボクセル単位         │
├───────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ 用途          │ LIO             │ LIO + Mapping   │ Registration        │
└───────────────┴─────────────────┴─────────────────┴─────────────────────┘
```

### 12.6 提案手法での統合

本設計では、AKF-LIO の利点を small_gicp に取り込む：

```cpp
// 提案: small_gicp + AKF-LIO 式 Gaussian
class GaussianVoxelMap {
    struct GaussianPoint {
        Eigen::Vector3d mean;
        Eigen::Matrix3d cov;
        Eigen::Vector3d normal;
        int pt_num;
        int use_num;
        double uncertainty;    // AKF-LIO から
        double planarity;
    };

    // ボクセル内に複数 Gaussian を許可（AKF-LIO 式）
    std::unordered_map<KeyType, std::vector<GaussianPoint>> voxels_;

    // マハラノビス距離による探索（AKF-LIO から）
    void getGaussiansWithMahalanobis(
        const Eigen::Vector3d& query,
        double threshold,
        std::vector<GaussianPoint>& results);

    // Pseudo-merge（AKF-LIO から）
    void applyPseudoMerge(double mahalanobis_threshold);

    // Uncertainty 更新（AKF-LIO から）
    void updateUncertainty(size_t idx, double residual);
};
```

**統合のポイント:**
1. **点単位 Gaussian**: small_gicp のボクセル単位ではなく、AKF-LIO の点単位 Gaussian を採用
2. **マハラノビス距離**: 対応点探索に共分散を考慮
3. **Pseudo-merge**: 近い Gaussian を統合してノイズ耐性向上
4. **Uncertainty 追跡**: 残差ベースの信頼度で移動体を間接的に除去

---

## 13. 参考文献

- [AKF-LIO: LiDAR-Inertial Odometry with Gaussian Map by Adaptive Kalman Filter](https://arxiv.org/pdf/2503.06891)
- [Faster-LIO: Lightweight Tightly Coupled Lidar-inertial Odometry using Parallel Sparse Incremental Voxels](https://github.com/gaoxiang12/faster-lio)
- [small_gicp: Efficient Point Cloud Registration](https://github.com/koide3/small_gicp)
