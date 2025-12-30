# AKF-LIO 技術議論まとめ

AKF-LIO の技術的詳細と関連手法についての議論をまとめた文書。

---

## 1. スキャンベース vs ポイントベース LIO

### 1.1 処理方式の比較

| 方式 | 代表的手法 | 処理単位 | 更新頻度 | 特徴 |
|------|-----------|----------|----------|------|
| **スキャンベース** | FAST-LIO2, Faster-LIO, AKF-LIO | 1スキャン（数千〜数万点） | 10Hz 程度 | 計算効率、ノイズ耐性 |
| **ポイントベース** | Point-LIO, MA-LIO, iG-LIO | 1点ずつ | kHz オーダー | 低遅延、急動作対応 |

### 1.2 AKF-LIO の処理フロー（スキャンベース）

```
┌────────────────────────────────────────────────────────────┐
│  SyncPackages()                                            │
│  → LiDAR スキャン全体 + 対応する IMU データを同期           │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│  p_imu_->Process()                                         │
│  → IMU 積分で姿勢を予測                                     │
│  → スキャン全体を undistort（運動歪み補正）                  │
└────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────┐
│  kf_.update_iterated_dyn_share_akf()                       │
│  → スキャン全点で対応探索                                   │
│  → 全点の残差を使って Iterated EKF で状態更新               │
└────────────────────────────────────────────────────────────┘
```

### 1.3 ポイントベース LIO 手法一覧

| 手法 | 年 | 特徴 |
|------|-----|------|
| **Point-LIO** | 2023 | 先駆的手法、1点ずつ状態更新 |
| **MA-LIO** | 2024 | Multi-Agent 対応 |
| **iG-LIO** | 2024 | Incremental Gaussian |
| **LOG-LIO** | 2024 | Local Gaussian |
| **Traj-LIO** | 2024 | 連続軌道表現 |

---

## 2. LiTAMIN2 のセンサモデルベース共分散

### 2.1 1点からの共分散推定

LiTAMIN2 は**センサモデル**を使って1点から共分散を推定する。

```
┌─────────────────────────────────────────────────────────────┐
│  LiDAR センサモデル                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  観測: (r, θ, φ) → (x, y, z)                                │
│                                                              │
│  不確実性の源:                                               │
│  ├─ σ_r : 距離測定誤差（距離に比例）                         │
│  ├─ σ_θ : 水平角分解能（センサ固有）                         │
│  └─ σ_φ : 垂直角分解能（センサ固有）                         │
│                                                              │
│  追加要因:                                                   │
│  └─ 入射角: 斜めに当たるほど不確実性増大                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 共分散計算の実装

```cpp
// LiTAMIN2 式: センサモデルから点ごとの共分散を計算
Eigen::Matrix3d computeCovariance(const Eigen::Vector3d& point) {
    // 1. 球座標に変換
    double r = point.norm();
    double theta = std::atan2(point.y(), point.x());      // 水平角
    double phi = std::asin(point.z() / r);                // 垂直角

    // 2. 球座標系での不確実性
    double sigma_r = range_sigma_base + range_sigma_rate * r;
    double sigma_theta = horizontal_resolution;
    double sigma_phi = vertical_resolution;

    // 3. 球座標系での共分散（対角行列）
    Eigen::Matrix3d cov_spherical = Eigen::Matrix3d::Zero();
    cov_spherical(0, 0) = sigma_r * sigma_r;
    cov_spherical(1, 1) = (r * sigma_theta) * (r * sigma_theta);
    cov_spherical(2, 2) = (r * sigma_phi) * (r * sigma_phi);

    // 4. ヤコビアン: 球座標 → カルテシアン座標
    Eigen::Matrix3d J = computeJacobian(r, theta, phi);

    // 5. 誤差伝播: Σ_xyz = J * Σ_spherical * J^T
    return J * cov_spherical * J.transpose();
}
```

### 2.3 共分散計算アプローチの比較

| 手法 | 共分散の意味 | 計算方法 | 1点で可能？ |
|------|-------------|----------|------------|
| **LiTAMIN2** | センサ観測の不確実性 | センサモデル | ✅ Yes |
| **VGICP/small_gicp** | 局所形状（平面性） | 近傍点の統計 | ❌ No |
| **AKF-LIO** | 累積観測の統計 + 不確実性 | 複数観測の統合 | ❌ No |

---

## 3. ハイブリッドアプローチ（iG-LIO / LOG-LIO）

### 3.1 iG-LIO (Incremental Gaussian LIO)

```
┌─────────────────────────────────────────────────────────────────┐
│  iG-LIO: 点単位処理 + 逐次 Gaussian 更新                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  各点について:                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. IMU 伝播 → 点の時刻まで状態予測                          ││
│  │ 2. 点をワールド座標に変換                                    ││
│  │ 3. 最近傍ボクセルの Gaussian と照合                          ││
│  │ 4. 残差で状態更新（点単位 EKF）                              ││
│  │ 5. Gaussian を逐次更新（Welford's algorithm）               ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  特徴: Point-LIO の高頻度 + VGICP の Gaussian 表現              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 LOG-LIO (Local Gaussian LIO)

```
┌─────────────────────────────────────────────────────────────────┐
│  LOG-LIO: 局所 Gaussian + 適応的処理                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2段階処理:                                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ [高頻度] 点単位の簡易マッチング → 姿勢更新                   ││
│  │          ・最近傍点との距離で残差                            ││
│  │          ・軽量な状態更新                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ [低頻度] 局所領域で Gaussian を構築 → 精密化                 ││
│  │          ・局所点群から共分散計算                            ││
│  │          ・Gaussian-to-Gaussian マッチング                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  特徴: 計算効率と精度のバランス                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 手法比較

| 項目 | iG-LIO | LOG-LIO | 提案ハイブリッド |
|------|--------|---------|-----------------|
| **点単位更新** | ✅ 毎点 | ✅ 毎点（簡易） | ✅ 毎点 |
| **Gaussian 更新** | 毎点（逐次） | バッチ（局所） | バッチ（スキャン） |
| **共分散計算** | Welford's | 局所点群統計 | センサモデル + 統計 |
| **AKF-LIO 要素** | ❌ | ❌ | ✅ Uncertainty, Pseudo-merge |

### 3.4 提案ハイブリッド設計

```cpp
// 提案: iG-LIO/LOG-LIO + AKF-LIO の利点
class ProposedHybridLIO {
    void processPoint(const PointType& pt, double timestamp) {
        // [iG-LIO 類似] 点単位で状態更新
        propagateIMU(timestamp);
        auto nearest = findNearestGaussian(pt);
        if (nearest.valid) {
            updateStateWithPoint(pt, nearest);
        }
    }

    void processScan(const CloudType& scan) {
        // [AKF-LIO 独自] Pseudo-merge
        applyPseudoMerge();

        // [AKF-LIO 独自] Uncertainty 更新
        for (auto& corr : correspondences_) {
            double residual = computeResidual(corr);
            corr.gaussian->uncertainty =
                alpha * corr.gaussian->uncertainty +
                (1 - alpha) * residual * residual;
        }
    }
};
```

---

## 4. AKF-LIO の Gaussian Map 構造

### 4.1 結論: 階層的でもマルチスケールでもない

AKF-LIO の Gaussian Map は**単一解像度のフラットハッシュマップ**。

```
┌─────────────────────────────────────────────────────────────────┐
│  AKF-LIO の Gaussian Map                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  構造: 単一解像度のフラットハッシュマップ                         │
│                                                                  │
│  std::unordered_map<KeyType, NodeType> grids_map_;              │
│                                                                  │
│  ┌─────┬─────┬─────┬─────┐                                      │
│  │     │     │     │     │  ← すべて同じサイズ                   │
│  ├─────┼─────┼─────┼─────┤     (ivox_grid_resolution = 0.5m)   │
│  │     │     │     │     │                                      │
│  ├─────┼─────┼─────┼─────┤                                      │
│  │     │     │     │     │                                      │
│  └─────┴─────┴─────┴─────┘                                      │
│                                                                  │
│  階層構造: なし（Octree ではない）                               │
│  マルチスケール: なし                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 誤解しやすい機能

| 機能 | 説明 | 階層的？ |
|------|------|----------|
| `nearby_type` (0,6,18,26) | 探索する隣接ボクセル数 | ❌ 単一解像度内の探索範囲 |
| `adap_voxel_size_en` | 適応的ボクセルサイズ | ❌ **スキャンのダウンサンプリング用** |
| LRU キャッシュ | 古いボクセルの削除 | ❌ メモリ管理のみ |

### 4.3 階層的手法との比較

| 手法 | 構造 | 特徴 |
|------|------|------|
| **AKF-LIO** | フラットハッシュ | 単一解像度、高速探索 |
| **Octomap** | Octree | 階層的、可変解像度 |
| **VDBFusion** | VDB (sparse octree) | 階層的、高速 |
| **SHINE-Mapping** | Multi-resolution hash | マルチスケール |

---

## 5. マハラノビス距離

### 5.1 基本概念

```
┌─────────────────────────────────────────────────────────────────┐
│  ユークリッド距離 vs マハラノビス距離                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ユークリッド距離:              マハラノビス距離:                │
│  d = ||x - μ||                 d = √((x-μ)ᵀ Σ⁻¹ (x-μ))         │
│                                                                  │
│      ● ● ●                         ●                            │
│    ● ● ● ● ●                     ●   ●                          │
│    ● ● ○ ● ●    vs             ●   ○   ●                        │
│    ● ● ● ● ●                     ●   ●                          │
│      ● ● ●                         ●                            │
│                                                                  │
│    等距離 = 円                  等距離 = 楕円（共分散の形状）     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 数式と実装

```cpp
// ユークリッド距離
double euclidean_distance(const Eigen::Vector3d& x, const Eigen::Vector3d& mu) {
    return (x - mu).norm();
}

// マハラノビス距離
double mahalanobis_distance(const Eigen::Vector3d& x,
                            const Eigen::Vector3d& mu,
                            const Eigen::Matrix3d& cov) {
    Eigen::Vector3d diff = x - mu;
    double d_squared = diff.transpose() * cov.inverse() * diff;
    return std::sqrt(d_squared);
}
```

### 5.3 χ² 分布との関係

マハラノビス距離² は χ²(k) 分布に従う（k = 次元数）

| 信頼度 | χ²(3) 閾値 | 意味 |
|--------|-----------|------|
| 90% | 6.25 | 90%の点がこの内側 |
| 95% | 7.82 | 95%の点がこの内側 ← **AKF-LIO デフォルト** |
| 99% | 11.34 | 99%の点がこの内側 |

### 5.4 AKF-LIO での用途

1. **対応点探索（KNNPointMAL）**: 共分散を考慮した最近傍探索
2. **Pseudo-merge 判定**: 同じ平面上かどうかの判定
3. **外れ値除去**: χ² 検定による統計的リジェクション

---

## 6. AKF-LIO の法線推定（角での優位性）

### 6.1 VGICP の問題: ボクセル境界での法線崩壊

```
┌─────────────────────────────────────────────────────────────────┐
│  VGICP (ボクセル単位 Gaussian)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    壁A        角        壁B                                      │
│    ││       ╱│╲        ││                                       │
│    ▼▼    ╱   ▼   ╲     ▼▼                                       │
│   ════  ┌─────────┐   ════                                      │
│   正常  │ 1ボクセル │   正常                                     │
│   法線  │ = 混合法線│   法線                                     │
│         │   ↙↓↘   │                                             │
│         └─────────┘                                             │
│              ↑                                                   │
│         角のボクセルで2平面が混ざる → 法線が不正確               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 AKF-LIO の解決策

```
┌─────────────────────────────────────────────────────────────────┐
│  AKF-LIO (点単位 Gaussian + マハラノビス距離 Pseudo-merge)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    壁A        角        壁B                                      │
│    ●●       ╱●╲        ●●                                       │
│    ●●      ╱ ● ╲       ●●                                       │
│    ▼▼    ╱   ▼   ╲     ▼▼                                       │
│   ════       ═══       ════                                     │
│                                                                  │
│   各点が独自の Gaussian を持つ                                   │
│                                                                  │
│   Pseudo-merge の条件:                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ マハラノビス距離 < 閾値 の場合のみマージ                   │   │
│   │                                                          │   │
│   │ 同じ平面上の点: マハラノビス距離 小 → マージされる        │   │
│   │ 異なる平面の点: マハラノビス距離 大 → マージされない      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 マハラノビス距離による選択的マージ

```cpp
// 同じ平面上: 共分散の方向が一致 → マハラノビス距離 小
//
//    pt1 ●────────● pt2
//        ↑        ↑
//        共分散が平面に沿う → マージされる ✓

// 異なる平面上: 共分散の方向が不一致 → マハラノビス距離 大
//
//    pt1 ●
//        │╲
//        │ ╲
//        │  ● pt2
//    法線方向に大きな差 → マージされない ✓
```

### 6.4 比較まとめ

| 項目 | VGICP | AKF-LIO |
|------|-------|---------|
| **Gaussian 粒度** | ボクセル単位 | 点単位 |
| **角での挙動** | 異なる平面が混合 | 各平面で別々に維持 |
| **マージ条件** | 空間的近さのみ | マハラノビス距離（形状考慮） |
| **法線精度** | 角で劣化 | 角でも正確 |

---

## 7. ポイントベース AKF-LIO の可能性

### 7.1 課題

| 課題 | 説明 |
|------|------|
| 1点での共分散 | 通常は複数点が必要 → **LiTAMIN2 で解決可能** |
| Pseudo-merge タイミング | スキャン単位でバッチ処理 |
| Uncertainty 更新粒度 | 点単位 vs スキャン単位 |

### 7.2 LiTAMIN2 + AKF-LIO 統合案

```cpp
class PointBasedGaussianLIO {
    void processPoint(const PointType& pt, double timestamp) {
        // 1. センサモデルから点の共分散を計算（LiTAMIN2 式）
        Eigen::Matrix3d pt_cov = computeSensorCovariance(pt);

        // 2. マップの Gaussian を探索（AKF-LIO 式マハラノビス距離）
        auto nearest = findNearestGaussianMahalanobis(pt, pt_cov);

        if (nearest.valid) {
            // 3. 点とマップ Gaussian の融合
            Eigen::Matrix3d combined_cov = pt_cov + nearest.gaussian->cov;
            double residual = computeMahalanobisResidual(pt, nearest, combined_cov);

            // 4. 状態更新
            updateStateWithPoint(pt, residual, combined_cov);

            // 5. マップ Gaussian をベイズ更新
            updateGaussianBayesian(nearest.gaussian, pt, pt_cov);
        }
    }
};
```

### 7.3 メリット・デメリット

| 項目 | スキャンベース（現状） | ポイントベース（提案） |
|------|----------------------|---------------------|
| **遅延** | 100ms（10Hz） | 1ms 以下 |
| **急動作対応** | △ | ◎ |
| **Gaussian 精度** | ◎ 多点から計算 | △ 逐次更新で誤差蓄積 |
| **計算効率** | ◎ バッチ処理 | △ オーバーヘッド |

---

## 8. GICP/VGICP vs AKF-LIO のマハラノビス距離

### 8.1 核心的な違い

標準的な GICP/VGICP もマハラノビス距離を使用するが、**使用箇所が異なる**。

```
┌─────────────────────────────────────────────────────────────────┐
│  マハラノビス距離の使用箇所                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                        対応点探索        コスト関数              │
│                        (Nearest Neighbor) (Optimization)         │
│  ─────────────────────────────────────────────────────────────  │
│  GICP/VGICP           ユークリッド距離    マハラノビス距離       │
│  AKF-LIO              マハラノビス距離    マハラノビス距離       │
│                       ↑                                         │
│                       ここが違う！                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 詳細比較

| 項目 | GICP/VGICP (small_gicp) | AKF-LIO |
|------|------------------------|---------|
| **対応点探索** | ユークリッド距離 KNN | マハラノビス距離 |
| **コスト関数** | マハラノビス距離 | マハラノビス距離 |
| **Gaussian 粒度** | ボクセル単位 | 点単位 |
| **動的更新** | なし（バッチ） | Pseudo-merge + Uncertainty |

### 8.3 なぜ対応点探索が重要か

```
┌─────────────────────────────────────────────────────────────────┐
│  角での対応点探索の違い                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  VGICP (ユークリッド距離で探索):                                 │
│                                                                  │
│      壁A ●                                                       │
│          │╲    ○ query                                          │
│          │ ╲  ╱ ← ユークリッド距離が最小の点を選択              │
│          │  ╲╱                                                   │
│      壁B ●───●                                                   │
│              ↑                                                   │
│         この点が選ばれる可能性                                   │
│         （異なる平面でも空間的に近ければ対応）                   │
│                                                                  │
│  → 対応が決まった後にマハラノビス距離でコスト計算               │
│  → 既に間違った対応を選んでいる可能性                           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AKF-LIO (マハラノビス距離で探索):                               │
│                                                                  │
│      壁A ●    ← この点の共分散は壁Aに沿った楕円                 │
│          │╲    ○ query                                          │
│          │ ╲                                                     │
│          │  ╲   マハラノビス距離で「遠い」と判定                │
│      壁B ●───● ← この点の共分散は壁Bに沿った楕円               │
│               ↑                                                  │
│          マハラノビス距離で「近い」と判定                        │
│          → 正しい対応を選択                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.4 コード比較

```cpp
// VGICP (small_gicp): ユークリッド距離で探索
std::vector<Correspondence> findCorrespondences(const PointCloud& source) {
    for (auto& pt : source) {
        // ユークリッド距離で最近傍を探索
        auto nearest = kdtree.nearestNeighbor(pt);  // ← ユークリッド

        if (nearest.distance < max_dist) {
            correspondences.push_back({pt, nearest});
        }
    }
    return correspondences;
}

// コスト関数ではマハラノビス距離を使用
double computeCost(const Correspondence& corr) {
    Eigen::Vector3d diff = corr.source - corr.target.mean;
    Eigen::Matrix3d combined_cov = corr.source_cov + corr.target_cov;
    return diff.transpose() * combined_cov.inverse() * diff;  // ← マハラノビス
}
```

```cpp
// AKF-LIO: マハラノビス距離で探索
std::vector<Correspondence> findCorrespondences(const PointCloud& source) {
    for (auto& pt : source) {
        std::vector<DistPoint> candidates;
        for (auto& map_pt : nearby_points) {
            Eigen::Vector3d diff = pt.pos - map_pt.mean;
            // マハラノビス距離で評価
            double mal_dist = diff.transpose() * map_pt.cov.inverse() * diff;
            if (mal_dist < threshold) {
                candidates.push_back({map_pt, mal_dist});
            }
        }
        // マハラノビス距離が最小の点を選択
        auto best = std::min_element(candidates.begin(), candidates.end());
        correspondences.push_back({pt, best->point});
    }
    return correspondences;
}
```

### 8.5 AKF-LIO のその他の利点

```
┌─────────────────────────────────────────────────────────────────┐
│  AKF-LIO の追加機能                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 点単位 Gaussian（ボクセル単位ではない）                      │
│     → 角や段差でも正確な法線を維持                              │
│                                                                  │
│  2. Pseudo-merge（マハラノビス距離ベース）                       │
│     → 同じ平面上の点のみマージ                                  │
│     → 異なる平面は分離を維持                                    │
│                                                                  │
│  3. Uncertainty 追跡                                             │
│     → 残差² を蓄積して信頼度を計算                              │
│     → 移動体の影響を自動的に抑制                                │
│                                                                  │
│  4. 指数重み付け                                                 │
│     weight = exp(-t_ratio_b * uncertainty)                       │
│     → 不確実な点は最適化への寄与を減らす                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.6 まとめ

| 機能 | GICP/VGICP | AKF-LIO |
|------|-----------|---------|
| 対応点探索 | ユークリッド | **マハラノビス** |
| Gaussian 粒度 | ボクセル | **点単位** |
| 動的 Gaussian 更新 | ❌ | **✅ Pseudo-merge** |
| 不確実性追跡 | ❌ | **✅ Uncertainty** |
| 移動体対策 | 外れ値除去のみ | **✅ 指数重み付け** |

**AKF-LIO の本質的な優位性**:
1. **対応点探索からマハラノビス距離を使用** → 正しい対応を見つけやすい
2. **点単位 Gaussian** → 角での法線精度
3. **動的な Uncertainty 追跡** → 移動体・ノイズへの耐性

---

## 9. LiTAMIN2 センサモデル + ポイントベース LIO

### 9.1 現状: 直接的な組み合わせは存在しない

| 手法 | 共分散計算方法 | ポイントベース？ |
|------|---------------|-----------------|
| **LiTAMIN2** | センサモデル | ❌ スキャン単位レジストレーション |
| **Point-LIO** | なし（点のみ） | ✅ |
| **iG-LIO** | Welford's（逐次統計） | ✅ |
| **LOG-LIO** | 局所点群統計 | ✅ |
| **MA-LIO** | なし | ✅ |

### 9.2 各手法の共分散アプローチ

```
┌─────────────────────────────────────────────────────────────────┐
│  ポイントベース LIO の共分散計算                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Point-LIO:                                                      │
│  └─ 共分散を使わない。点と平面の距離のみ                         │
│                                                                  │
│  iG-LIO:                                                         │
│  └─ Welford's algorithm で逐次更新                              │
│     → 複数点が蓄積されてから有効                                 │
│                                                                  │
│  LOG-LIO:                                                        │
│  └─ 局所ウィンドウの点群から計算                                 │
│     → バッチ処理が必要                                           │
│                                                                  │
│  LiTAMIN2 式:                                                    │
│  └─ センサモデルから1点で計算可能 ← これを使った LIO がない！    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 研究ギャップ: 未開拓の組み合わせ

```
┌─────────────────────────────────────────────────────────────────┐
│  提案: LiTAMIN2 センサモデル + ポイントベース LIO                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  既存手法の問題:                                                 │
│  ├─ Point-LIO: 共分散なし → 形状情報を活用できない              │
│  ├─ iG-LIO: 点が蓄積するまで共分散が不正確                      │
│  └─ LOG-LIO: バッチ処理が必要で遅延                             │
│                                                                  │
│  LiTAMIN2 センサモデルの利点:                                    │
│  ├─ 1点目から正確な共分散が利用可能                             │
│  ├─ 距離・入射角依存の不確実性をモデル化                        │
│  └─ 追加計算コストが小さい                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.4 提案設計: Sensor-Aware Point-LIO

```cpp
// LiTAMIN2 センサモデル + Point-LIO の統合
class SensorAwarePointLIO {
    SensorCovarianceModel sensor_model_;  // LiTAMIN2 由来

    void processPoint(const PointType& pt, double timestamp) {
        // 1. IMU 伝播
        propagateIMU(timestamp);

        // 2. センサモデルから点の共分散を計算（LiTAMIN2 式）
        //    → 1点目から使える！
        Eigen::Matrix3d pt_cov = sensor_model_.computeCovariance(pt);

        // 3. マップから最近傍を探索
        auto nearest = findNearestInMap(pt);

        if (nearest.valid) {
            // 4. 共分散を考慮した残差計算
            Eigen::Matrix3d combined_cov = pt_cov;
            if (nearest.has_covariance) {
                combined_cov += nearest.cov;
            }

            // 5. 情報行列で重み付けした状態更新
            Eigen::Matrix3d info = combined_cov.inverse();
            updateStateWithInformation(pt, nearest, info);
        }
    }
};
```

### 9.5 期待される利点

| 項目 | Point-LIO | iG-LIO | 提案手法 |
|------|-----------|--------|----------|
| **初期共分散** | ❌ なし | △ 不正確 | ✅ センサモデルから |
| **入射角考慮** | ❌ | ❌ | ✅ |
| **距離依存不確実性** | ❌ | ❌ | ✅ |
| **計算オーバーヘッド** | 最小 | 小 | 小 |
| **1点目からの精度** | △ | △ | ✅ |

### 9.6 実装難易度

```
低 ←──────────────────────────────────────→ 高

Point-LIO      提案手法        iG-LIO      LOG-LIO
(共分散なし)   (センサモデル)   (逐次統計)   (局所バッチ)

実装は比較的シンプル:
- LiTAMIN2 のセンサモデル部分を抽出
- Point-LIO の状態更新に組み込み
- 情報行列で重み付け
```

---

## 10. 参考文献

- [AKF-LIO](https://arxiv.org/pdf/2503.06891) - Gaussian Map + Adaptive Kalman Filter
- [Faster-LIO](https://github.com/gaoxiang12/faster-lio) - iVox ベース高速 LIO
- [Point-LIO](https://github.com/hku-mars/Point-LIO) - ポイントベース LIO
- [LiTAMIN2](https://github.com/koide3/LiTAMIN2) - センサモデルベース共分散
- [iG-LIO](https://github.com/zijiechenrobotics/ig_lio) - Incremental Gaussian LIO
- [small_gicp](https://github.com/koide3/small_gicp) - 高速 GICP/VGICP
