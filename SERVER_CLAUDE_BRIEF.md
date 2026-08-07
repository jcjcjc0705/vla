# 簡報:在 Isaac Sim 裡訓練 OMX 抓取策略 → 微調 GR00T

> 這份是給**跑在 RTX Pro 6000 伺服器上的 Claude** 的完整簡報。它是自足的——你看不到
> 產生它的那段對話,需要的資訊都在這裡。

---

## 0. 你在哪、你能碰什麼

你在一台 **RTX Pro 6000(Blackwell, 96 GB)伺服器**上。

| | 伺服器(你) | 使用者的筆電 |
|---|---|---|
| GPU | RTX Pro 6000, 96 GB | RTX 2060, 6 GB |
| Isaac Sim | **5.1.0 原生安裝**,headless + WebRTC(已驗證可串流到筆電) | 也有一份 |
| 實體 OMX 手臂 | **沒有** | leader + follower 都接在這裡 |
| 角色 | 場景、資料收集、訓練、評估 | 只當觀看端 |

**你完全碰不到實體機器人,這個階段也不需要。** 整個專案是 sim-only。

**Isaac Sim 安裝(已確認)**

| | 路徑 | 版本 |
|---|---|---|
| 伺服器 | `/home/pochun/isaac_sim_5.1` | `5.1.0-rc.19+release.26219.9c81211b.gl` |
| 筆電 | `~/Desktop/isaac_sim_5.1` | 同上 |

兩台**版本完全一致**,目錄結構也一致(伺服器少一個使用者自寫的 `start_isaac_ros.sh`)。
`python.sh`、`kit/`、`VERSION`、`setup_ros_env.sh`、`isaac-sim.compatibility_check.sh`
都在。使用者**沒有 sudo**,Isaac 裝在自己的 home 底下;同一台機器上別的使用者可能有
別的版本,**只用 `/home/pochun/isaac_sim_5.1`**。

**伺服器環境(已實測)**:Ubuntu 24.04.4 LTS、系統內建 Python 3.12.3、
driver 580.159.03(支援到 CUDA 13.0)、GPU 97,887 MiB。
工作目錄 `~/omx_vla/`。§2 列出要準備的東西。

**`docker run --gpus all` 可用** —— nvidia-container-toolkit 已設好,不需要 sudo。
所以 **ML 那一層跑在容器裡**(符合使用者既有的 image + compose + `utils.sh` 做法),
**Isaac 留在原生**(搬進容器要處理 Vulkan/RTX/串流埠,是另一個工程,而且原生現在就能跑)。
兩者靠 ROS 2 溝通(§4.3),容器加 `--network host` 與相同的 `ROS_DOMAIN_ID` 即可。

⚠️ **GPU 是共用的**,同機器上其他使用者可能在跑 Isaac(觀察到約 5.5 GB / 22% 佔用)。
還有約 92 GB 可用,GR00T 微調建議 40 GB,可以並存,不必等對方。

---

## 1. 目標

讓 OMX-F(OpenManipulator-X follower)在 Isaac 裡把桌面上的方塊夾起來。

**硬性要求:模型不能知道方塊在哪,必須從影像自己找出來。**

- 腳本專家用 ground-truth 方塊座標算 IK — **只發生在產生資料時**
- 策略只吃影像 + 關節狀態 — 座標從未進入模型

這是標準的 privileged expert → visuomotor student。

最終要微調 `nvidia/GR00T-N1.7-3B`,但**先用 ACT 驗證資料可學性**。

---

## 2. 要準備的東西

**兩個行程,一條 ROS。** Isaac 原生跑在 host 上(它要 GPU、RTX、WebRTC);所有控制
程式跑在 `omx_bridge_image` 容器裡(它要 rclpy、MoveIt)。兩者靠 ROS 2 Jazzy 上的
topic / service 溝通,`ROS_DOMAIN_ID` 一致即可。§4.3 說明為什麼是這個形狀。

```bash
# 1. 這個 repo
git clone https://github.com/jcjcjc0705/vla.git

# 2. 機器人資料層 —— 原生 Isaac 要 host 上的 USD/URDF 檔案
git clone https://gitlab.screamtrumpet.csie.ncku.edu.tw/pochun/omx_bridge_image.git
#    沒 clone 也行,直接從 image 取出來:
#    docker run --rm -v "$PWD:/out" \
#      registry.screamtrumpet.csie.ncku.edu.tw/pochun/omx_bridge_image:latest cp -r /assets /out

# 3. 同步引擎 —— **host 端**的 Isaac 腳本要它才知道關節順序
#    HTTPS 不通就用 SSH:
#    ssh://git@gitlab.screamtrumpet.csie.ncku.edu.tw:722/pochun/sim_real_bridge_image.git
git clone https://gitlab.screamtrumpet.csie.ncku.edu.tw/pochun/sim_real_bridge_image.git

# 4. 控制層的 image(引擎、profile、jog、MoveIt、Isaac prim 服務介面全在裡面)
docker pull registry.screamtrumpet.csie.ncku.edu.tw/pochun/omx_bridge_image:latest
```

| 東西 | 用途 |
|---|---|
| `omx_bridge_image/assets/omx_f.usd` | Isaac 場景(8.2 MB,已 flatten)。**host 上要有**,Isaac 原生讀它 |
| `omx_bridge_image/assets/omx_f/omx_f.urdf` + `meshes/` | `sim/ik.py` 讀幾何,MoveIt 也用同一份的絕對路徑版本 |
| `sim_real_bridge_image/bridge/` | `profile.py` —— 純 Python + yaml,無 ROS 相依。**只有 host 端需要** |
| `omx_bridge_image` image | 容器裡的一切:`sim_real_bridge` 引擎、profile、`jog`、`ik_target`、`moveit_kin`、`isaac_prim` |

**不要抄一份 `profile.py`。** 全鏈只能有一份關節順序的真相。`sim/task_config.py`
自己會找對地方 —— 容器裡 import 已安裝的 ROS 套件,host 上沿
`task/pick_cube.task.yaml` 的 `paths.sim_real_bridge` candidates 找 checkout。
兩條路指向同一份檔案的兩個副本,內容必須一致,所以**兩個都跟著 tag 走,不要各自改**。

**設定這台機器**(0.2.2 起 `.env` 不進 repo,要自己從範本建):

```bash
cd omx_bridge_image
cp docker/compose/.env.example docker/compose/.env
# 填 ISAAC_SIM_PATH=/home/pochun/isaac_sim_5.1
# ROS_DOMAIN_ID 維持 1
```

`scripts/isaac.sh` 與 compose 讀的是同一份 `.env`,所以 host 上的 Isaac 跟容器裡的
節點不會分岔。**注意 `scripts/` 沒有烤進 image** —— 它只在 git clone 裡,所以
`omx_bridge_image` 這個 repo 一定要 clone,不能只 `docker pull`。

**起 Isaac:**

```bash
cd omx_bridge_image
bash scripts/isaac.sh --streaming       # headless + WebRTC,使用者從筆電連
```

⚠️ **一定要用這支 script,不要直接跑 `isaac-sim.sh`。** host 只要 source 過任何
ROS,Isaac 就會去那份 ROS 裡找 `isaac_ros2_messages`、找不到就放棄,而**沒有任何一個
apt 裝的 ROS 有這個套件**(它只在 Isaac 自己的 bundle 裡)。症狀是場景看起來完全正常
但 prim 服務一個都沒開,錯誤只是一行沒人會看的 warning。source 了 jazzy 也一樣壞 ——
這跟發行版無關。

⚠️ **eco mode 預設是關的**(要的話加 `--eco`)。它會壓算繪,而這台機器上算繪速率正是
要量的東西 —— 見 §10。

驗證兩邊都通:

```bash
# 容器裡
bash docker/scripts/vla.sh                 # 進 shell(compose 掛 vla/ 進 /vla)
r                                          # 第一次要 build
python3 -c "from sim_real_bridge.profile import load_profile; \
  print(load_profile('/profile/omx_f.profile.yaml').joints)"
# 期待: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper_joint_1']

# host 上,用 Isaac 的直譯器 —— 印出它解析到的每一條路徑
bash sim/isaac_python.sh sim/task_config.py
```

⚠️ host 上跑的話,`sim_real_bridge` 要沿 `paths.sim_real_bridge` 找得到 checkout。
找不到時 `task_config` 會直接說是哪一條路徑失敗。

`omx_arm_image`(真臂驅動)**不需要**,這台機器沒有真手臂。

---

## 3. 已驗證的事實 — 這些不是假設

以下都是直接讀 USD 與 Isaac 擴充原始碼查證出來的。**其中幾個會讓你在不知情時做錯事。**

### 3.1 `omx_f.usd` 不能用 `AddReference` — 必須用 sublayer

這份 USD 是 flatten 過的,根層有 16 個 `Flattened_Prototype_1..16` prim spec,
外加根層的 `/visuals`、`/colliders`、`/meshes`。`/omx_f/link6/collisions` 內部參照
`</Flattened_Prototype_5>`。

**用 `AddReference` 引入 default prim `/omx_f` 會讓這些全部懸空,靜默丟失所有視覺與
碰撞幾何。** 用 sublayer:

```python
stage = Usd.Stage.CreateNew(out)
stage.GetRootLayer().subLayerPaths = ["<相對路徑>/omx_f.usd"]
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetDefaultPrim(stage.GetPrimAtPath("/omx_f"))
```

驗證:composed stage 的 prim 數要跟原本一致(原本 traverse 約 235 個 prim)。

### 3.2 **從這個 repo 不要修改 `omx_f.usd`**

它是 sim↔real 校正鏈的一部分,使用者的 `jog` / `to_sim` / `to_real` 都依賴它。
所有任務增修都寫成 `pick_cube.usd` 裡的 override。

它**不是**永遠不變的檔案 —— `omx_bridge_image` 0.2.0 在它自己的 repo 裡加了
`/ik_target`(隱藏的除錯用目標方塊)、一個 TF 發佈節點,以及 `ROS2ServicePrim`。
兩件事因此成立:

- **`pick_cube.usd` 會繼承那三樣**,所以 `build_scene.py` 不再自己加
  `ROS2ServicePrim` —— 兩個節點會搶同一組服務名。
- 要動 `omx_f.usd` 就去 `omx_bridge_image` 那個 repo 動、發 tag、重出 image。
  不要從 `vla/` 這邊改。

### 3.3 夾爪力量偏弱 —— 是**參數問題**,不是風險

所有 drive 是 `type = acceleration`、stiffness 625、damping 40。
`gripper_joint_1` 的 `JointEquivalentInertia = 4.51e-5`,
所以力矩 ≈ `4.51e-5 × 625 × err` ≈ `0.028 × err` N·m。
0.3 rad 過衝時約 8.5 mN·m,指尖法向力約 **0.14 N** —— 對應可夾持 **約 25–30 g**。

**這不代表夾不起來。** 計畫的方塊是 **15 g**,已在預算內、還有約 2 倍餘裕。
而且這是 sim,方塊的質量/摩擦/尺寸與 drive 的 stiffness **全都是你設的參數**——
跟真手臂不同(那邊改不了馬達扭矩)。

真的不夠力時,依序轉這幾個旋鈕(**全部只在 `pick_cube.usd` 的 override 或
`task/pick_cube.task.yaml` 裡**,不要動 `omx_f.usd`):方塊變輕 → 摩擦調高 →
stiffness 625 → ~5000(damping 隨之 ~100)。

### 3.4 手指 collider 是 `convexHull` —— 這個「變輕」解決不了

`/colliders/link6/follower_07_gripper_motorized/node_STL_BINARY_` 與 link7 的對應者,
`physics:approximation = convexHull`。

實際量過兩根手指的網格:**扁平比只有 3.7 : 2.1 : 1**(塊狀零件,不是薄板),
且只有 **3–6%** 的頂點貼在最外側的兩個面上。所以凸包**不會**保留出平整的夾持面。

這是唯一一個「把方塊變輕」解決不了的項目——接觸幾何不好時,方塊會從指間滑掉或轉出去,
跟重量無關。

> **實測結論:留在 `convexHull`,不要換。** 兩種都試過了,理由寫在
> `task/pick_cube.task.yaml` 的 `overrides.finger_colliders` 註解裡,摘要:
>
> - **兩者都是過度近似**,凸塊的聯集必然 ⊇ 原始網格,分解只是換個地方胖。
>   夾得很穩的時候兩側**都**看得到縫 —— 那是 collider 比視覺網格胖的顯示落差,
>   不是沒接觸。
> - **功能上等價**:analytic 與 moveit 兩種解算器下都是 20/20。
> - 分解讓相機從 12.0 掉到 9.4 Hz,而且**視覺判讀失效**(凸塊比網格胖又不被算繪)。
>   這一段開發裡用眼睛看是最有效的診斷管道。
>
> 接觸幾何若真的變成瓶頸,下一步是 `overrides.pads`(明確給一個平整夾持面),
> **不是換近似方式**。

### 3.5 `gripper_joint_2` 的 mimic 剛度 —— **M1 唯一真正的障礙,已解決**

`/omx_f/joints/gripper_joint_2` 的 `apiSchemas` 是**空的** —— 它**沒有任何 drive**,
純粹靠 `PhysxMimicJoint:rotZ` 約束跟隨 `gripper_joint_1`。

原廠 `naturalFrequency = 25.0` 在 1/120 s 的步長下**太軟**:夾到東西時被動指會被
物體頂開、嚴重落後,結果**只有一根手指在出力**,另一根等於不存在。

**已實測解法:`naturalFrequency` 25 → 1000**(寫在 `task/pick_cube.task.yaml` 的
`overrides.mimic_joint`)。改完在 GUI 裡用 `jog` 手動驗證,方塊確實被夾起並抬離桌面。
調硬也更貼近真實 —— 真手臂第二指是齒輪連動的剛性連結,不是彈簧。

⚠️ **這個問題對「調 `gripper_joint_1` 的 drive stiffness」完全免疫。** 實測把它從
625 拉到 20000(32 倍)、方塊從 15 g 減到 5 g,掉落距離一毫米都沒變 —— 因為出力的
那根手指本來就夠力。**如果日後又出現夾不住,先看被動指跟不跟得上,不要再去調 drive。**

### 3.5b `gripper_joint_2` 是**真的** PhysX mimic joint

`/omx_f/joints/gripper_joint_2` 帶 `PhysxMimicJointAPI:rotZ`,
`referenceJoint → gripper_joint_1`,`gearing = 1.0`,`naturalFrequency = 25.0`,
`dampingRatio = 0.8`。

URDF 註解說它「visual only」,那句是針對 ros2_control 而言。**在 USD 裡它是物理耦合的、
會實際參與夾取。** 原廠是軟彈簧,受力時被動指會落後、造成單邊夾持而滑脫 —— 已由 §3.5
的 `naturalFrequency` 25 → 1000 解決。`expert_node` 每一集都會回報這個誤差
(`mimic_error()`),它變大就是被動指又跟不上了。

它佔一個真實的 articulation DOF,所以 `/joint_states` 會發 **7** 個名字,不是 6 個。

### 3.6 home 姿態夾爪是**閉合**的

q=0 時 `link6`(y ∈ [−0.0025, 0.0176])與 `link7`(y ∈ [−0.0209, −0.0007])的
world bbox 相接。`gripper_joint_1` 軸是 +Z、`link6` 在 +y 側,所以**正值 = 開**。
這是推導出來的,信心高,但花 30 秒在 GUI 確認一下。

### 3.7 Python 版本分裂 — 這決定了整個檔案配置

- **Isaac Sim 5.1 = CPython 3.11**(`kit/python/bin/python3.11`,extscache 都是 `cp311`)
- **LeRobot ≥ 0.6 要求 Python ≥ 3.12**
- `rclpy` 也不能從 Isaac 的 python 匯入(Jazzy 是 3.12)

**所以 torch / lerobot / rclpy 都裝不進 Isaac 的直譯器。強制多行程。**

實際切出來的行程長這樣:

| 行程 | Python | 跑什麼 |
|---|---|---|
| Isaac(host 原生) | 3.11 | 模擬。ActionGraph 收發 ROS,見 §4.2 |
| `omx_bridge_image` 容器 | 3.12(Jazzy) | 專家、錄製、`jog`、`ik_target` —— 全部用 rclpy |
| ML 容器 / venv | 3.12 | torch + lerobot,訓練與轉檔 |

**Isaac 與控制層之間的界線是 ROS,不是自己寫的 RPC。** Isaac 的 ROS 2 bridge 本來
就用 Jazzy(用 `omx_bridge_image/scripts/isaac.sh` 起 Isaac,它會強制用 Isaac 自帶的
jazzy bundle),所以兩邊講同一套訊息,不需要橋接層。

**不要**為了統一而升級到 Isaac Sim 6.0。這在動工前評估過了,見 §9 第 7 條。
另外注意:**就算 Python 版本一致,把訓練堆疊裝進 Isaac 的直譯器仍然是壞主意**
(Isaac 6.0 自己綁 torch 2.10,會跟 lerobot 的 pin 打架)。那個行程邊界本來就該存在。

### 3.8 Blackwell = sm_120 — **已在這台機器上實測通過**

2026-08-04 在伺服器上驗證(容器內):

```
torch     : 2.11.0+cu128
cuda      : 12.8
capability: (12, 0)          ← sm_120
device    : NVIDIA RTX PRO 6000 Blackwell Workstation Edition
matmul    : -318418.875      ← kernel 真的執行了
```

**這條路已知可行,照做就好。** 不是傳言的 sm_122。需要 **torch ≥ 2.7.0 +
cu128(或更新)的 index**。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

pip 的 torch wheel **自帶完整 CUDA runtime**,所以**不需要**安裝 CUDA toolkit
(系統的或 conda 的都不用),只需要驅動。
⚠️ Isaac 附的 `environment.yml` 釘 `cuda-toolkit=11.8` —— **那個在 Blackwell 上不能用**,
不要拿它建 conda 環境來跑 torch。

**絕對不要讓任何 `requirements.txt` 去釘 torch 版本。** 先單獨從 cu128 索引裝 torch,
再裝其他東西。驗證要**實際跑一次 kernel**,只印版本會騙人:

```python
import torch
print(torch.__version__, torch.cuda.get_device_capability())   # 期待 (12, 0)
a = torch.randn(4096, 4096, device="cuda"); (a @ a).sum().item()
```

### 3.9 場景數據

Z-up、`metersPerUnit = 1.0`、地面在 z=0。
articulation root 是 `/omx_f/root_joint`(fixed base),但既有的
`IsaacArticulationController` 用 `targetPrim=/omx_f` 且能運作,所以
`SingleArticulation(prim_path="/omx_f")` 應該是對的 —— 第一次跑時用 `arm.dof_names` 確認。

EE 在 home 的位置 `(0.3129, −0.0016, 0.2107)`;肩部在 `(−0.011, 0, 0.0975)`,
最大伸展約 **0.40 m**。方塊生成環帶 `r ∈ [0.16, 0.26]` 有充足餘裕。

`physicsScene`:`enableGPUDynamics = False`、TGS、32 position iterations、CCD 開。

### 3.10 5 軸的根本限制

`joint1` 底座偏航;`joint2/3/4` 是三個**平行**俯仰軸,在 joint1 選定的垂直平面內構成
平面 3R;`joint5` 前臂滾轉。

- 給得出位置 3 DOF + 該平面內的接近俯仰角。
- **給不出獨立的 EE 偏航** —— 接近方位角永遠是 `atan2(y, x)`。
- **但由上而下抓取時限制消失**:俯仰 90° 朝下時,`joint5` 的滾轉軸剛好垂直,它**就變成**
  夾爪繞垂直軸的偏航。任意 `(x, y)`、任意方塊朝向的垂直夾取完全可行。

**把任務限定為由上而下抓取。** 不要要求手臂做不到的姿態 —— 解算器會回 `None`,
或更糟,回一個看似合理的錯姿態。

---

## 4. 架構決策(照做,別重新發明)

### 4.1 原生 Isaac,不引入 Isaac Lab

NVIDIA 的 LeIsaac 綁 Isaac Lab 2.1 + Isaac Sim 4.5/5.0,這裡是 5.1.0。借它的想法,
不用它的程式碼。

⚠️ **Isaac 內建的 `PickPlaceController` / Lula IK / `ArticulationKinematicsSolver` /
`ParallelGripper` 一個都沒用到**,不要再去接它們 —— 它們都住在 Isaac 的直譯器裡,
而現在控制程式在容器裡。運動學見 §5。

headless 串流:配方在
`standalone_examples/api/isaacsim.simulation_app/livestream.py`。使用者的 AppImage
連得上,M0 已驗證。

### 4.2 Isaac 端用 ActionGraph,控制流在容器裡

**Isaac 不跑任何自己寫的控制邏輯。** 它的 ActionGraph 只做四件事,全是 stock 節點:

| 節點 | 作用 |
|---|---|
| `ROS2PublishJointState` / `ROS2SubscribeJointState` | `/joint_states` 出、`/joint_command` 進 |
| `ROS2PublishTransformTree` | 方塊(與 `/ik_target`)的位姿走 TF 出來 |
| `ROS2ServicePrim` | 從外面讀寫 prim 屬性 —— 這是**放置方塊**唯一的路 |
| `IsaacCreateRenderProduct` + `ROS2CameraHelper` | 兩台相機的影像 |

回合重置、方塊隨機化、成功判定、狀態機全部在容器裡的 Python,透過上面那四樣操作
Isaac。**你手動開 Isaac、載場景、按 Play,然後在容器裡 `ros2 run`** —— Isaac 那邊
不需要 standalone script。

### 4.3 一切都走 ROS

錄製、專家、鍵盤控制,全部經由 `/sync/command` → `sync_node` → `/joint_command`。

> ⚠️ **這推翻了本文件早期版本的 §4.3。** 那時寫的是「錄資料不經過 ROS,直接在 Isaac
> 行程內驅動 articulation」,理由是時間對齊。實際做下去換掉它的原因是:
>
> - **reset 不必在 Isaac 端做。** `ROS2ServicePrim` 的 `set_prim_attribute` 可以從
>   外面寫 `xformOp:translate`,方塊真的會動(實測過)。這是原本判斷「純 ROS 不存在」
>   的那個前提,而它是錯的。
> - **行程內的路要 rclpy 才能跟 bridge 共用**,而 rclpy 進不了 Isaac 的 3.11。走
>   ROS 之後 `jog`、專家、未來的 policy 全部共用同一條命令路徑,M7 不再是一個階段。
> - **時間對齊改用記錄而不是保證**:`state` 與 `action` 取自控制迴圈的同一次迭代,
>   所以 `action[t]` 確實是 `state[t]` 當下發出的命令。影像做不到 —— 它們以算繪的
>   步調自己到達 —— 所以每一幀記下**當時用的是哪張影像、那張多舊**,由轉檔器決定
>   怎麼處理過期影像。假裝同時發生比記下延遲更糟。

⚠️ 影像的 `header.stamp` 是 Isaac 的**模擬時間**,跟控制迴圈的時鐘不是同一個。
拿兩者相減會得到十億秒等級的數字。用**到達的 wall time**,`recorder.py` 就是這樣做的。

### 4.4 原始傾印 → 轉檔,不要直接寫 LeRobot

`recorder.py` 寫 `data/raw/ep_XXXXX/{frames.npz, img_*.png, meta.json}`;
轉檔器(3.12)產生 dataset。

LeRobot 一年內走過 v2 → v2.1 → v3.0,而 **GR00T 要的是 LeRobot v2 變體 +
`meta/modality.json`**,ACT 走 v3.0。留一份格式無關的原始傾印,換格式是重跑
`convert.py`,不是重錄 200 集。

影像在寫入前降到 `cameras.record_resolution`(320×240)。用算繪解析度存的話一集
約 47 MB,200 集就是 9 GB 的、策略根本不會用到那個尺寸的像素。

### 4.5 相機:錄兩台,訓練時再選

- `cam_front` — 胸口高度的固定相機,**內參對齊 Intel RealSense D455**
  (使用者有一台;換算工具在筆電的 `camera/cameracalibration/isaac_camera.py`,
  公式是 `focal_length = fx × horizontal_aperture / image_width`,
  diagonal aperture 預設 7.2 mm)。起點:`eye≈(0.75, −0.45, 0.55)`,
  `target≈(0.21, 0, 0.03)`,算繪 640×480,資料集降到 320×240。
- `cam_wrist` — 掛在 `/omx_f/link6`,朝夾爪前方。

**不需要額外的環境相機** —— 底座固定,「胸口相機」與「環境相機」幾何上等價。

⚠️ 真手臂上**沒有**腕上相機。M5 必須跑「只用 front」vs「front + wrist」的對照,
好讓使用者在花錢買腕上相機之前知道它值不值。

ROS 相機發布**現在就在用**(錄製就靠它)。headless 正確的接法:
```
OnPlaybackTick -> IsaacCreateRenderProduct(cameraPrim, w, h) -> ROS2CameraHelper(type="rgb")
```
用 `isaacsim.core.nodes.IsaacCreateRenderProduct`,**不要**用
`IsaacCreateViewport`/`IsaacGetViewportRenderProduct`(那需要 headless 下不存在的
viewport)。`build_scene.py` 的 `add_camera_nodes()` 已經是這個形狀。

⚠️ **筆電上相機只發得出約 12 Hz,而控制迴圈是 30 Hz** —— 大約每兩幀重複一次像素。
`recorder.py` 把每一幀用的影像索引與**當時的影像年齡**都記下來了,轉檔器要決定怎麼
處理。這台機器算繪快得多,先量再決定:`meta.json` 裡的 `image_age_s`。

---

## 5. 運動學:兩個解算器,不是 Lula

⚠️ **Lula 從沒用過,也不要再去試。** 它住在 Isaac 的直譯器裡,而控制程式在容器裡。
`omx_f_descriptor.yaml` 不存在,也不需要。實際有兩個可互換的解算器:

| 解算器 | 在哪 | 用途 |
|---|---|---|
| **MoveIt**(主力) | image 裡的 `omx_bridge_app.moveit_kin.MoveItKin` | ROBOTIS 官方的 `moveit/` 設定,KDL plugin |
| **解析解**(對照) | `sim/ik.py` 的 `OMXKinematics` | 閉式解,已對 Isaac 驗到 0.024 mm |

`ros2 run omx_vla_app expert --ros-args -p ik:=moveit|analytic|position_only` 切換。
兩者都跑到 20/20。

```python
kin = MoveItKin()                    # 容器裡,約 1.3 秒啟動
kin.seed = q_measured                # ⚠️ 每次都餵量到的關節值
q = kin.ik([0.22, 0.10, 0.12])       # 夾持點的世界座標(公尺)→ 5 個弧度,或 None
```

⚠️ **`position_only_ik: True`**(官方設定,不是我們加的)。5 軸手臂做不到任意 6-DOF
姿態,所以只解位置、放掉朝向。夾爪俯仰角是算完才知道的(抓取瞬間落在 32–44°,
斜著伸進去),`joint5` 也不受控制。**斜抓是可行的**,實測抓得起來。

⚠️ **零空間有 2 個自由度而且沒有選擇準則**,答案幾乎完全由種子決定。讓求解器接著自己
上一次的答案跑會漂,而且漂得很混亂 —— 曾經只是改了 collider 的近似方式(碰不到
運動學),平均抓取俯仰角就從 44° 變成 7°。

**解析解的那個坑**(如果要重寫):`joint2 → joint3` 的偏移向量是
`(0.0415, 0, 0.11315)`,**偏離垂直 20.15°**(OpenManipulator-X 的經典肘偏置)。
沒把這個常數帶進去,每個解都會系統性偏掉。`sim/ik.py` 從 URDF 讀,不寫死。

**自碰撞**:`moveit_kin` 解出來的姿態會先過自碰撞才回傳,碰到就擾動種子重解
(預設 8 次)。Isaac 的 articulation 是 `enabledSelfCollisions = False`,穿模不會有
任何人抗議,所以這道關卡是唯一擋得住的地方。

### 場景與隨機化(都在 `task/pick_cube.task.yaml`,不要寫進程式碼)

**方塊**:2.5 cm、15 g,**自訂 `PhysicsMaterial(static_friction=1.0,
dynamic_friction=0.9)`** —— 預設的 0.2 太滑。

**隨機化**:`sim/spawn.py` 是唯一的取樣器,用 `np.random.default_rng(seed)`。
**不要**用 `omni.replicator` —— 它是為 SDG 擷取迴圈設計的,會跟控制迴圈打架。
定案的環帶:`r ∈ U[0.16, 0.24]`、`θ ∈ U[−50°, 50°]`、`yaw ∈ U[−45°, 45°]`。
`holdout_theta_deg: [[-35,-20],[20,35]]` 是**內側**的兩條保留帶(M5 的驗收用),
不是外緣 —— 外緣同時也是最遠、最難的地方,拿它當保留區會把「泛化差」跟「本來就難」
混在一起。

**成功判定**在 `success:` 底下:抬高 `lift_height: 0.05` m、方塊離夾爪
`max_ee_distance: 0.06` m 以內、連續成立 `hold_steps: 15`(0.5 s)。
距離條件擋掉「方塊被彈飛」,連續 15 步擋掉瞬間彈跳,**兩個都要**。

---

## 6. 資料集格式

> **這段已對 lerobot 0.6.1 校正過**(2026-08-07 實跑)。原本寫的
> `"dtype": "video"` 配 `use_videos=False` 會**直接 raise**,而 `shape` 用 list
> 會讓每一幀都被拒。實作見 `ml/convert.py`。

```python
features = {
  "observation.state": {"dtype": "float32", "shape": (6,),
      "names": ["joint1","joint2","joint3","joint4","joint5","gripper_joint_1"]},
  "action":            {"dtype": "float32", "shape": (6,), "names": [...同上...]},
  # ⚠️ "image" 不是 "video":dataset_metadata.py 明確 raise 「features contain
  #    video keys but use_videos is False」。
  # ⚠️ shape 必須是 **tuple**:validate_feature_numpy_array 用
  #    `ndarray.shape != expected_shape` 比較,list 永遠不相等,錯誤訊息還會印出
  #    兩個看起來一模一樣的 shape。
  "observation.images.front": {"dtype": "image", "shape": (3, 240, 320),
      "names": ["channels","height","width"]},
  "observation.images.wrist": {...同上...},
}
ds = LeRobotDataset.create(repo_id="screamlab/omx_pick_cube", fps=30,
                           features=features, root=..., robot_type="omx_f",
                           use_videos=False)
for f in frames: ds.add_frame({**f, "task": "pick up the cube"})
ds.save_episode()
ds.finalize()      # 必須 —— 少了它 parquet footer 永遠不會寫入
```

**可以放心的事**:`validate_feature_image_or_video` 同時接受 `(C,H,W)` 與
`(H,W,C)`,所以 PNG 讀出來直接餵,不必 transpose。

⚠️ **推論時正規化不在 policy 裡。** checkpoint 帶著
`policy_preprocessor_*` / `policy_postprocessor_*`,正確流程是
`pre(obs) → policy.select_action → post(action)`,用
`make_pre_post_processors(policy_cfg=..., pretrained_path=...)` 建。直接呼叫
`select_action` 會餵進未正規化的弧度與像素、拿回未反正規化的動作 —— **手臂照樣會動,
不會有任何錯誤**。preprocessor 內含 `to_batch_processor` 與 `device_processor`,
所以觀測要給**未加 batch、CPU 上**的張量。

⚠️ **lerobot 0.6.x 把功能拆成 extras**:要 `lerobot[dataset,training]`,
否則 import 或訓練時才報 ImportError。

- **`observation.state` 是量測到的關節位置**(`arm.get_joint_positions()[dof_ix]`),
  **不是**命令值。存成命令值的話 ACT 會學到恆等映射 —— 訓練分數很漂亮、實際什麼也不會。
- **`action` 是絕對關節目標**,不是增量。絕對值才對得上 `/sync/command` 的語意,
  M7 才會只是五行改動。
- `task` 是常數字串(API 要求),**不要**讓它長出詞彙表 —— 這階段不做語言條件。
- `fps = 30`;物理 1/120,每 4 步算繪並記錄一次。
- **`use_videos=False`** —— 200 × ~150 frames × 2 相機 × 320×240 PNG ≈ 5 GB。
  存 PNG 而非 MP4 可以把影片解碼移出 dataloader,對這種規模的資料集那才是真正的瓶頸。
- **只有成功的回合進資料集。** 行為克隆沒有機制利用失敗。失敗留在 `data/raw/` 供除錯。

---

## 7. 里程碑

### M0 — 環境查證(半天,不寫程式)
`/etc/os-release`、`nvidia-smi`、`uv python install 3.12`、從 cu128 裝 torch 並**實際跑
一次 kernel**(§3.8)。跑 `livestream.py` 讓使用者用 AppImage 連上。順手測
`docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`。
✅ matmul 真的在 GPU 上跑完,且使用者從筆電看得到 headless standalone Isaac。
> `docker --gpus` **不是關鍵路徑**。Isaac 原生跑、錄製原生跑、訓練用 uv venv 就好
> (免 sudo、比容器簡單)。失敗不改變任何計畫,測只是為了知道。

### M1 — 任務場景,不含學習(1 天)—— 這是在**找參數**,不是風險關卡
`build_scene.py` → `pick_cube.usd`(sublayer,§3.1)。加方塊、加兩台相機。
然後**手動把夾爪移到方塊上並夾起來**。調 gripper stiffness、摩擦、
`convexHull` → `convexDecomposition`,直到夾得住。
✅ 方塊被夾起來過,且成功的數值寫進 `task/pick_cube.task.yaml`。

**為什麼要在錄資料之前做:** 不是因為可能做不到——sim 裡每個失敗模式都有便宜的解
(見 §3.3、§3.4),**沒有任何結果會殺掉這個專案**。真正的理由是**改了方塊質量或
collider 之後,先前錄的資料就作廢**。重錄 200 集才貴,先把旋鈕轉對很便宜。

心態是「花半小時定下參數」,不是「怕做不到」。卡超過一兩小時就直接用最粗暴的解
(方塊再變輕、加 box pad),不要在這裡精雕細琢。

> **M1 已完成。** 定案參數在 `task/pick_cube.task.yaml`:mimic
> `natural_frequency: 1000`、`grasp_offset: [-0.006, -0.0002, 0.0]`(EE 座標系)。
> 回歸驗證改成跑專家本身:`ros2 run omx_vla_app expert --ros-args -p episodes:=20`
> —— 它每一集都完整做一次抓取與抬升,改任何物理參數後跑一次即可。

### M2 — 腳本專家(1–2 天)—— ✅ **已完成**
> 狀態機在 `sim/expert.py`(`APPROACH → DESCEND → CLOSE → LIFT → HOLD → DONE`),
> ROS 執行器在 `src/omx_vla_app/omx_vla_app/expert_node.py`。
>
> ```bash
> ros2 run omx_vla_app expert --ros-args -p episodes:=20            # 預設 moveit
> ros2 run omx_vla_app expert --ros-args -p episodes:=20 -p ik:=analytic
> ros2 run omx_vla_app expert --ros-args -p episodes:=20 -p holdout:=true
> ```
> **實測 moveit 20/20、analytic 20/20**,命令與量測的落差約 2.6–2.8 mm。
>
> ⚠️ **同一時間只能跑一個。** 兩個行程一起發 `/sync/command` 會互相污染,而且結果
> 看起來只是「比較不穩」而不是壞掉。跑之前先 `pgrep -f omx_vla_app`。
>
> 兩個花了很多時間才找到的東西,不要退掉:
> - **到位判定要看量測值,不能只看命令。** 只看命令的話手臂會落後十幾 mm(方塊才
>   25 mm),夾合就發生在方塊上緣;靠夾合的 0.5 秒去追是時序巧合,重力力矩大的姿態
>   就會失敗。`expert.settle_tol` / `settle_max_ticks` 就是這件事。
> - **MoveIt 的種子每個 tick 從量到的關節值重設**,理由見 §5。

### M3 — 走通骨架(1 天)**最小的端到端證明** —— 錄製半邊已完成
> `recorder.py` 已經在跑:`ros2 run omx_vla_app record --ros-args -p episodes:=5`
> 寫出 `data/raw/ep_XXXXX/`,只留成功的回合。筆電上已產出 3 集。

還沒做的是 `convert.py` → `lerobot-train --steps=200` → `eval.py` 驅動 Isaac。
✅ 手臂會因為訓練過的 checkpoint 而動。**它一定會失敗,那就是預期結果。**
重點是每個接縫都通、每個 tensor shape 都對、`finalize()` 產出的 dataset 載得回來。
**不要為了「省時間」跳過這步直接錄 200 集。**

> **這是移植到這台機器之後的第一件事。** 需要 py3.12 + torch cu128,筆電做不了。

### M4 — 真正的資料集(半天,無人值守)
200 集成功回合,完整隨機化。**專家要注入雜訊**:waypoint ±5 mm、接近高度 ±2 cm、
夾合時機 ±3 frame、每步小高斯雜訊 —— 你要的是一個分布,不是一條軌道。
腳本專家的資料天生近乎單模態,不注入雜訊會讓 ACT 學得很乾淨但完全不會泛化。
**保留一塊訓練不用的位置區域當測試集。** 兩台相機都存。
✅ dataset 載得回、約 30k frames、兩個影像 key 都在、抽樣播放正常。

### M5 — 訓 ACT,在沒看過的位置評估
先 50 集看趨勢,再 200 集。**跑兩組:只用 `front` vs `front + wrist`。**
```bash
lerobot-train --dataset.repo_id=screamlab/omx_pick_cube --policy.type=act \
  --batch_size=64 --steps=60000 --num_workers=8 --output_dir=outputs/act_v1
```
ACT 預設(resnet18 + ImageNet 權重、`dim_model=512`、`n_obs_steps=1`、
`chunk_size=100`、`use_vae=True`、`lr=1e-5`)就是好起點。
在 96 GB 上約 10–20 steps/s,60k 步約 1–2 小時 —— **這是估計,先跑 500 步量實際速度**。
低於 8 steps/s 就是 dataloader 瓶頸(提高 `num_workers`、確認 `use_videos=False`)。

✅ **保留區域的成功率明顯 >0,且與訓練區域差距不大。**
⚠️ 訓練區高、保留區接近 0 = **模型在背軌跡而不是看影像**。這是這類專案最常見的假成功。
發生的話回頭加大隨機化與專家雜訊,不要往下走。

### M6 — 微調 GR00T N1.7
`nvidia/GR00T-N1.7-3B`、`--embodiment-tag NEW_EMBODIMENT`(就是為自訂手臂設計的)。
從同一份 `data/raw/` 轉出 GR00T 要的格式 + `meta/modality.json`。
微調建議 40 GB+ VRAM(你有 96 GB)。GR00T 自己釘 Python 3.12 / CUDA 12.8 /
PyTorch 2.7 —— 剛好就是 Blackwell 需要的。
先 `gr00t/eval/open_loop_eval.py` 做離線檢查,再接 inference server/client 進 Isaac。
⚠️ GR00T 內建的 sim benchmark 是 LIBERO / SimplerEnv / RoboCasa,**不含 Isaac Sim**,
閉迴路評估要自己寫 client。
✅ 保留區域成功率與 ACT 對照。

### M7 — ~~接回 ROS seam~~ **已經是現況了**
> 專家、錄製、`eval` 全部已經走 `/sync/command`,而且 `expert_node.py` 就是照
> `jog.py` 的寫法:行程內起 `SyncNode` 強制 `mode:=command`、背景執行緒 spin、
> 只用公開介面。
>
> **所以 `policy_node` 不是一個里程碑,是把 `expert_node` 裡叫專家的那一行換成
> 叫策略。** ~~剩下的差異只有推論要在哪個行程跑~~ —— **這題已經有答案了**:
>
> 推論就跑在控制容器裡。做了一個衍生 image `omx_vla_image`
> (`FROM omx_bridge_image` + torch cu128 + lerobot),所以 torch 與 rclpy 共用同一個
> 直譯器,`ml/eval.py` 一個行程就能跑策略又發 `/sync/command`。實測 M3 走通:
> 3 集各跑滿 450 步、`/sync/command` 上量到 455 則無 NaN 的命令、手臂確實因
> checkpoint 而動。
>
> ⚠️ 但 lerobot 逼著 numpy 升到 2.x,而那會讓 `moveit_py` **segfault**(不變條件 11),
> 所以 `ik:=moveit`、`jog`、`ik_target`、`monitor` 要在另一個容器
> (`omx_vla_ctrl`,`docker/compose/docker-compose-ctrl.yml`)。`eval.py` 用 analytic
> 解算器,不受影響。

**指向真手臂只差 `targets:=real` 加一台真相機。** ⚠️ 這台機器沒有真手臂,而且
`expert_node.py` 寫死 `targets:=sim` —— 那一行是唯一擋住的東西,不要拿掉。

### M8(日後)— 人類示範
在**使用者的筆電**上跑輕量 Isaac + leader 臂收人類示範(leader 接在筆電),
格式相同、併進同一個 dataset。腳本資料近乎單模態,人類資料能補上多樣性。

---

## 8. 檔案配置

```
vla/                           # https://github.com/jcjcjc0705/vla
├── README.md                  # 專案總覽(給人看的)
├── SERVER_CLAUDE_BRIEF.md     # 這份
├── task/pick_cube.task.yaml   # 唯一手改的設定:方塊尺寸/質量/摩擦、生成環帶、
│                              # 相機姿態、夾爪開合值、drive override、成功門檻、fps
├── assets/pick_cube.usd       # 產生的(gitignored);sublayer 指向 omx_f.usd
├── sim/                       # 用 Isaac 的直譯器跑 — CPython 3.11,無 ros、無 torch
│   ├── task_config.py         # 讀 task yaml + sim_real_bridge.profile(關節順序)
│   ├── build_scene.py         # 產生 pick_cube.usd(純 USD,不用 SimulationApp)
│   ├── ik.py                  # 解析解運動學 + --check / --map / --isaac 驗證模式
│   ├── expert.py              # 抓放狀態機(只有邏輯,執行在 ROS 那邊)
│   ├── spawn.py               # 唯一的取樣器(含保留帶)
│   ├── app.py  scene.py       # 行程內的 Isaac 路徑,只剩 ik.py --isaac 在用
│   └── isaac_python.sh
├── src/omx_vla_app/           # ROS 2 py3.12 — 在 omx_bridge_image 容器裡 build
│   └── omx_vla_app/
│       ├── expert_node.py     # 專家與錄製的執行器(entry: expert / record)
│       ├── moveit_ik.py       # MoveItKin 的適配層(FK 走解析解)
│       └── recorder.py        # → data/raw/
├── docker/                    # compose 掛 vla/ 進 omx_bridge_image 容器
└── data/                      # gitignored
```

~~**還沒建的**:`convert.py`、訓練腳本、`eval.py`~~ —— **已經建好了**(2026-08-07):

```
ml/                            # 學習層。跑在 omx_vla_image 容器裡
├── convert.py                 # data/raw/ -> LeRobotDataset
└── eval.py                    # checkpoint -> /sync/command
docker/compose/
├── docker-compose-vla.yml     # omx_vla_image  — 轉檔/訓練/推論/ik:=analytic
└── docker-compose-ctrl.yml    # omx_bridge_image — ik:=moveit/jog/ik_target/monitor
LAPTOP_CLAUDE_BRIEF.md         # 給筆電那邊的增量簡報
```

訓練沒有自己的腳本 —— 就是 `lerobot-train` CLI,參數記在 `LAPTOP_CLAUDE_BRIEF.md` §2。

**環境全部在 docker 裡,沒有 venv 也沒有 conda。** 新 repo
`gitlab.screamtrumpet.csie.ncku.edu.tw/pochun/omx_vla_image` 只放環境
(Dockerfile + CI),這隻手臂的程式碼全部留在 `vla`。⚠️ 為什麼是兩個容器而不是一個,
見不變條件 11。

---

## 9. 不變條件(違反了會很難查)

1. **從這個 repo 不修改 `omx_f.usd`** —— 它是 sim↔real 校正鏈的一部分。所有任務
   增修都是 `pick_cube.usd` 裡的 override。要動它就去 `omx_bridge_image` 那個 repo,
   見 §3.2。
2. **絕不對 `omx_f.usd` 用 `AddReference`** —— 用 sublayer,見 §3.1。
3. **關節順序只有一個來源** —— `import sim_real_bridge.profile`,不要抄常數。
4. **`action` 必須是合法的 `/sync/command` payload** —— 6 個關節、同順序、弧度、
   USD 座標、絕對值。
5. **`observation.state` 是量測值,不是命令值。**
6. **torch 一定單獨從 cu128 索引裝**,不讓任何 requirements 檔釘它。
7. **不要升級 Isaac Sim。** 版本已定為 `5.1.0-rc.19`,兩台機器一致。
   這是**動工前評估過的決定,不是慣性**——6.0 確實是 Python 3.12 且能原生 source
   ROS 2 Jazzy,但那兩個好處現在都已經用別的方式拿到了(控制層在容器、Isaac 用自帶
   的 jazzy bundle),所以升級的理由比當初更弱。此外:
   (a) 它目前是 **Early Developer Release、不是 GA**;
   (b) binding 的 public import path 搬了位置,§5 查證過的每個 API 都要重驗;
   (c) 5.1 已在這台 Blackwell 上實際跑起來過,相容性風險已退場;
   (d) 兩台機器共用 `omx_f.usd`,版本分家會製造難查的差異。
   第一個 ML 專案不該蓋在 pre-release 上——除錯時會分不出 bug 是 Isaac 的還是自己的。
   **6.0 到 GA 之後,在 M6/M7 時再重新評估。** `rc.19 → 5.1.0 GA` 也不要換,
   除非真的撞到可歸因於 RC 的 bug。
8. **只有成功的回合進資料集。**
9. **控制迴圈用模擬時間,不用 wall clock。** `expert_node.ExpertClient.wait_sim()`。
   Isaac 在 headless streaming 下**不限速**(實測 2.42x,且隨 GPU 負載浮動);
   `time.sleep(1/30)` 會讓每個 tick 之間走掉 80 ms 模擬時間,種子軌跡一變,
   position-only IK 的俯仰角就漂出可解範圍。**11/20 vs 19/20。**
   ⚠️ kit 的限速參數(`rateLimitEnabled` / `rateLimitFrequency` /
   `useFixedTimeStepping` / `useFastMode`)四個都試過,headless 下全部無效,
   不要再試那條路。
10. **`expert.reset()` 之前要把 IK 種子重設成量到的關節值。** 種子每個 tick 重設
    那行在迴圈**裡面**,而規劃在迴圈**外面** —— 少了這行,規劃用的是上一集最後
    留下的姿態,失敗會成塊出現(觀察到連續 3 集與連續 9 集),因為失敗的那集不進
    迴圈、種子就永遠不會更新。**跟機器速度無關,筆電上一樣會發作。**
11. **MoveIt 與 lerobot 不能裝在同一個 python 環境。** lerobot 要 `numpy>=2.0`,
    而 `moveit_py` 是 C++ binding、numpy 2.0 破壞了 C ABI。**它 import 正常、
    執行時 segfault,沒有任何訊息。** 所以有兩個容器:`omx_vla`(學習層,numpy 2.x)
    與 `omx_vla_ctrl`(控制層,numpy 1.x + MoveIt)。⚠️ **不要用 import 檢查來判斷
    MoveIt 可不可用** —— 這個錯誤已經犯過一次。

## 10. 尚未查證的事(依賴前先確認)

> 2026-08-07 在伺服器上實測後,前三題有答案了。

- ✅ **轉檔器該怎麼處理過期影像 → 不必特別處理,但不是零重複。** 這台機器實測
  `image_age_s` 平均 4.59 ms、最大 31.07 ms,而**唯一影像/幀 = 0.865** ——
  相機在模擬時間上約 26 Hz 對 30 Hz 控制迴圈,13.5% 的 tick 重複像素(筆電是
  12 Hz 對 30 Hz,約六成)。`convert.py` 照 index 展開、不丟不插值,只在統計裡報告。
  ⚠️ `recorder.py` 的 age 仍是 wall clock 記的,而控制週期現在是 sim time(§4.3),
  兩者不同基準 —— `0.865` 那個數字才是不依賴時鐘換算的。
- ✅ **訓練速度 → 2 step/s,dataloader 是瓶頸。** 200 步、batch 8、`num_workers=4`:
  `updt_s 0.028` vs `data_s 0.472`,dataloader 佔 94%。`use_videos=False` 已生效,
  要調的是 `num_workers` 與 batch。**M5 開始前先量。**
- ✅ **推論跑在哪個行程 → 就在控制容器裡。** 見 M7。
- **GR00T 要的 LeRobot 格式版本與 lerobot 0.6 寫出的 v3.0 是否有落差**(§4.4 的設計
  正是為此)—— 還沒查。
- ⚠️ **新的未解問題:兩台機器的光照可能不同。** 這台的 Isaac 是從筆電複製過來的,
  `omni.kit.stage_templates` 裡的資源路徑仍寫死 `/home/jcjcjc/Desktop/isaac_sim_5.1/...`,
  所以預設 stage 的 DomeLight HDR 與一張地板材質**在這台載不到**(每次開 Isaac 都會刷
  Error,連 `ik.py --isaac` 開的全新行程也一樣)。算繪本身正常、有光有陰影、相機影像
  可用,但 **M8 要把筆電的人類示範併進同一個 dataset 之前,必須比對兩台的 `cam_front`**。

已經查證掉、不用再查的:ROS distro(容器裡是 Jazzy,Isaac 用自帶的 jazzy bundle)、
`SingleArticulation(prim_path="/omx_f")`(對,但現在只有 `ik.py --isaac` 在用)、
Lula(不用了,見 §5)、**kit 的限速參數**(`rateLimitEnabled` / `rateLimitFrequency` /
`useFixedTimeStepping` / `useFastMode` 四個都試過,headless streaming 下全部無效,
見 §9.9)。

## 11. 參考

- NVIDIA/Isaac-GR00T — https://github.com/NVIDIA/Isaac-GR00T
- GR00T N1.5 微調 SO-101 — https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning
- GR00T in LeRobot — https://huggingface.co/blog/nvidia/nvidia-isaac-gr00t-in-lerobot
- LeIsaac(Isaac 裡遙操作 → GR00T)— https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/
- LeRobotDataset v3.0 — https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3
- sm_120 確認 — https://github.com/pytorch/pytorch/issues/157549
