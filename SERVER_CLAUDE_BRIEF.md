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
工作目錄 `~/omx_vla/`,兩個 repo 已 clone 完成。

**`docker run --gpus all` 可用** —— nvidia-container-toolkit 已設好,不需要 sudo。
所以 **ML 那一層跑在容器裡**(符合使用者既有的 image + compose + `utils.sh` 做法),
**Isaac 留在原生**(搬進容器要處理 Vulkan/RTX/串流埠,是另一個工程,而且原生現在就能跑)。
兩者用 §4.4 的 zmq 邊界溝通,容器加 `--network host` 即可。

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

## 2. 先 clone

```bash
# 機器人資料層 — USD 場景、URDF+meshes、profile
git clone https://gitlab.screamtrumpet.csie.ncku.edu.tw/pochun/omx_bridge_image.git

# 同步引擎 — 只需要其中的 sim_real_bridge/profile.py
git clone ssh://git@gitlab.screamtrumpet.csie.ncku.edu.tw:722/pochun/sim_real_bridge_image.git
# 沒有 SSH key 就改 HTTPS:
# git clone https://gitlab.screamtrumpet.csie.ncku.edu.tw/pochun/sim_real_bridge_image.git
```

| 檔案 | 用途 |
|---|---|
| `omx_bridge_image/assets/omx_f.usd` | Isaac 場景(8.2 MB,已 flatten) |
| `omx_bridge_image/assets/omx_f/omx_f.urdf` + `meshes/` | Lula IK 的輸入 |
| `omx_bridge_image/profile/omx_f.profile.yaml` | 關節名稱與順序 |
| `sim_real_bridge_image/bridge/sim_real_bridge/profile.py` | 純 Python + yaml,**無 ROS 相依** |

**不要抄一份 `profile.py`**,加進 `PYTHONPATH`。全鏈只能有一份關節順序的真相:

```bash
export PYTHONPATH="$PWD/sim_real_bridge_image/bridge:$PYTHONPATH"
python -c "from sim_real_bridge.profile import load_profile; \
  print(load_profile('omx_bridge_image/profile/omx_f.profile.yaml').joints)"
# 期待: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper_joint_1']
```

這行同時驗證了 clone 成功、以及 `profile.py` 在 Isaac 的 Python 3.11 下 import 得起來。

`omx_arm_image`(真臂驅動)**不需要**。

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

### 3.2 **絕對不要修改 `omx_f.usd`**

它是 sim↔real 校正鏈的一部分,使用者的 `jog` / `to_sim` / `to_real` 工具都依賴它。
所有場景增修都寫成 `pick_cube.usd` 裡的 override。

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
跟重量無關。解法依序試:`convexHull` → `convexDecomposition`;仍是圓角就在
`link6`/`link7` 底下各加一片薄的 `FixedCuboid` 當夾持墊(~5 行,常見且正當的做法)。

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
會實際參與夾取。** 但它是軟彈簧,受力時被動指會落後,可能造成單邊夾持而滑脫 —— M1 要觀察。

它佔一個真實的 articulation DOF,所以 `/joint_states` 會發 **7** 個名字,不是 6 個。

### 3.6 home 姿態夾爪是**閉合**的

q=0 時 `link6`(y ∈ [−0.0025, 0.0176])與 `link7`(y ∈ [−0.0209, −0.0007])的
world bbox 相接。`gripper_joint_1` 軸是 +Z、`link6` 在 +y 側,所以**正值 = 開**。
這是推導出來的,信心高,但花 30 秒在 GUI 確認一下。

### 3.7 Python 版本分裂 — 這決定了整個檔案配置

- **Isaac Sim 5.1 = CPython 3.11**(`kit/python/bin/python3.11`,extscache 都是 `cp311`)
- **LeRobot ≥ 0.6 要求 Python ≥ 3.12**
- `rclpy` 也不能從 Isaac 的 python 匯入(Humble 是 3.10、Jazzy 是 3.12)

**所以 torch / lerobot 裝不進 Isaac 的直譯器。強制雙行程。**

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

**把任務限定為由上而下抓取。** 不要要求手臂做不到的姿態 —— Lula 會回 `succ=False`,
或更糟,回一個看似合理的錯姿態。

---

## 4. 架構決策(照做,別重新發明)

### 4.1 原生 Isaac standalone,不引入 Isaac Lab

NVIDIA 的 LeIsaac 綁 Isaac Lab 2.1 + Isaac Sim 4.5/5.0,這裡是 5.1.0。
Isaac 已內建需要的每一塊:

| 元件 | 位置 |
|---|---|
| `PickPlaceController`(通用十階段抓放狀態機) | `isaacsim.robot.manipulators.controllers` |
| Lula IK 0.10.1 | `isaacsim.robot_motion.lula` |
| `ArticulationKinematicsSolver` | `isaacsim.robot_motion.motion_generation` |
| `ParallelGripper` | `isaacsim.robot.manipulators.grippers` |

headless 也看得到:配方在
`standalone_examples/api/isaacsim.simulation_app/livestream.py` —
`SimulationApp({"headless": True, ...})` + `enable_extension("omni.services.livestream.nvcf")`。
使用者的 AppImage 連得上。

### 4.2 GUI ActionGraph 不夠用,用 standalone Python

回合重置、方塊隨機化、成功判定都是 Python 控制流。用 OmniGraph 表達它們需要自訂 OGN
節點或 `ScriptNode` —— 嚴格來說更多工、更難除錯。而且伺服器是 headless 的,
透過 WebRTC 做 GUI 編輯很痛苦。

### 4.3 錄資料**不經過 ROS**

直接在 Isaac 行程內驅動 articulation。理由是**時間對齊**:寫 action → `world.step()`
→ 讀 observation,`action[t]` 真的就是施加在 `state[t]` 的動作。

走 ROS 的話:`sync_node` 的固定 50 Hz 重發會把動作邊緣抹成階梯、影像與關節狀態在不同
時鐘上——等於教模型去擬合 DDS 抖動。而且 **reset 本來就只能在 Isaac 端做**,
「純 ROS」這個選項並不存在。

**但 seam 要保在 schema 層**:錄下的 `action` 就是合法的 `/sync/command` payload
(同樣 6 個關節名、同順序、弧度、USD 座標)。這靠 import `sim_real_bridge.profile`
取關節順序來機械性地保證。M7 再把 ROS 接回來。

### 4.4 原始傾印 → 轉檔,不要直接寫 LeRobot

Isaac(3.11)寫 `data/raw/ep_XXXXX/{frames.npz, img_*.png, meta.json}`;
轉檔器(3.12 venv)產生 dataset。

被 Python 版本差**強制**,但本來就是更好的設計:LeRobot 一年內走過 v2 → v2.1 → v3.0,
而 **GR00T 要的是 LeRobot v2 變體 + `meta/modality.json`**,ACT 走 v3.0。
留一份格式無關的原始傾印,換格式是重跑 `convert.py`,不是重錄 200 集。

3.11/3.12 的界線只由**一個 ~40 行的 zmq request/reply** 跨越,送 `{image, state}`
回 `action`。**不要在這裡耍聰明**(ONNX 匯出可行但麻煩且無益——sim 時間你控制得了,
延遲無所謂)。這個界線在 M7 會被 ROS topic 對重新實作,兩者形狀相同。

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

ROS 相機發布(**M7 才需要**,錄製不用)——headless 正確的接法:
```
OnPlaybackTick -> IsaacCreateRenderProduct(cameraPrim, w, h) -> ROS2CameraHelper(type="rgb")
```
用 `isaacsim.core.nodes.IsaacCreateRenderProduct`,**不要**用
`IsaacCreateViewport`/`IsaacGetViewportRenderProduct`(那需要 headless 下不存在的
viewport)。範例:`standalone_examples/testing/isaacsim.ros2.bridge/test_camera_tf_delay.py`。
把它放在 `/World/task/CameraGraph`,**不要**動 `/omx_f/ActionGraph`。

---

## 5. 關鍵 API(已查證簽章)

```python
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera

open_stage(str(ASSETS / "pick_cube.usd"))
world = World(stage_units_in_meters=1.0, physics_dt=1/120.0, rendering_dt=1/30.0)
arm  = world.scene.add(SingleArticulation(prim_path="/omx_f", name="omx_f"))
cube = world.scene.add(SingleRigidPrim(prim_path="/World/task/cube", name="cube"))
ee   = SingleXFormPrim(prim_path="/omx_f/link5/end_effector_link")
cam  = Camera(prim_path="/World/task/cam_front", resolution=(640, 480), frequency=30)
world.reset(); cam.initialize()
dof_ix = [arm.dof_names.index(j) for j in CANONICAL_JOINTS]   # 7 個 DOF 裡取 6 個
```

**IK**(`ArticulationKinematicsSolver.compute_inverse_kinematics` 的實際簽章):
```python
compute_inverse_kinematics(target_position, target_orientation=None,
                           position_tolerance=None, orientation_tolerance=None)
    -> (ArticulationAction, bool)
```
`target_orientation=None` 給位置-only IK,`orientation_tolerance` 可放寬 —— 5 軸手臂
兩者都用得上。

**Lula 需要一份 `robot_descriptor.yaml`(要你手寫,~20 行)**,範本在
`standalone_examples/api/isaacsim.robot.manipulators/ur10e/rmpflow/robot_descriptor.yaml`:
```yaml
api_version: 1.0
cspace: [joint1, joint2, joint3, joint4, joint5]
default_q: [0.0, 0.0, 0.0, 0.0, 0.0]
acceleration_limits: [10, 10, 10, 10, 10]
jerk_limits: [10000, 10000, 10000, 10000, 10000]
cspace_to_urdf_rules:
  - {name: gripper_joint_1, rule: fixed, value: 0.0}
  - {name: gripper_joint_2, rule: fixed, value: 0.0}
collision_spheres: {}      # 純 IK 空的就好,只有 RMPflow 需要
```
EE frame 名稱是 `end_effector_link` —— 用 `solver.get_all_frame_names()` 確認。

⚠️ **Lula 吃不吃這份 URDF 尚未實測。** `<ros2_control>` 是 urdfdom 會忽略的未知元素、
`<mimic>` 是標準的,*應該*沒問題。若它拒絕,就做一份只有運動學的
`omx_f_kinematics.urdf`。

**退路:解析解 IK**(~40 行)。`q1 = atan2(y, x)`,投影到垂直平面,沿接近方向退掉腕長
(0.0287 + 0.09193 = 0.1206 m),用餘弦定理解 L₂ = 0.1205、L₃ = 0.162 的 2R。
**陷阱:joint2→joint3 的偏移向量是 `(0.0415, 0, 0.11315)`,偏離垂直 20.15°**
(OpenManipulator-X 的經典肘偏置)。沒把這個常數帶進去,每個解都會系統性偏掉。
Lula 免費處理它,所以 Lula 優先。

**方塊**:`DynamicCuboid(size=0.025, mass=0.015)`,**務必自訂
`PhysicsMaterial(static_friction=1.0, dynamic_friction=0.9, restitution=0.0)`** ——
預設的 `static_friction=0.2` 太滑。

**隨機化**:用 `np.random.default_rng(seed)`,五行就好。**不要**用
`omni.replicator` / `isaacsim.replicator.domain_randomization` —— 它們是為 SDG 擷取
迴圈設計的,會跟手寫的 stepping loop 打架。
取樣環帶:`r ∈ U[0.16, 0.26]`、`θ ∈ U[−50°, 50°]`、`yaw ∈ U[−45°, 45°]`、`z = 0.0125`。
把 `seed` 與取樣結果存進 episode metadata,任何一集都能重現。

**成功判定**(幾何,不需要接觸感測器):
```python
p, _ = cube.get_world_pose(); ee_p, _ = ee.get_world_pose()
lifted = p[2] > CUBE_HALF + 0.05
held   = np.linalg.norm(p - ee_p) < 0.06
success = lifted and held      # 必須連續成立 15 步(0.5 s)
```
`held` 擋掉「方塊被彈飛」;連續 15 步擋掉瞬間彈跳。**兩個都要**。

---

## 6. 資料集格式

```python
features = {
  "observation.state": {"dtype": "float32", "shape": (6,),
      "names": ["joint1","joint2","joint3","joint4","joint5","gripper_joint_1"]},
  "action":            {"dtype": "float32", "shape": (6,), "names": [...同上...]},
  "observation.images.front": {"dtype": "video", "shape": (3, 240, 320),
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

> **M1 已完成 2026-08-05。** 定案參數在 `task/pick_cube.task.yaml`:mimic
> `natural_frequency: 1000`、`grasp_offset: [-0.006, -0.0002, -0.011]`(EE 座標系)。
> 隨時可用 `bash sim/isaac_python.sh sim/grasp_test.py` 一鍵回歸驗證 ——
> 改任何物理參數後都該跑一次,不用再開 GUI。

### M2 — 腳本專家(1–2 天)
Lula IK + 抓放狀態機。**先跑 1000 個取樣姿態的 IK,回報失敗率**,再錄任何東西。
狀態機:`HOME → 開夾爪 → 方塊上方 +0.10 m → 下降到抓取高度 → 夾合(目標略超過接觸,
如 −0.05 rad)→ 停 10 步 → 抬升 +0.15 m → 保持 15 步 → 判定`。
waypoint 之間在**關節空間**以約 0.6 rad/s 內插,讓動作串流平滑。
✅ 20 個隨機方塊位置中成功 ≥18。**還沒有 dataset。**

### M3 — 走通骨架(1 天)**最小的端到端證明**
錄 **5** 集 → `convert.py` → `lerobot-train --steps=200` → `serve.py` 載 checkpoint
→ `eval.py` 驅動 Isaac。
✅ 手臂會因為訓練過的 checkpoint 而動。**它一定會失敗,那就是預期結果。**
重點是每個接縫都通、每個 tensor shape 都對、`finalize()` 產出的 dataset 載得回來。
**不要為了「省時間」跳過這步直接錄 200 集。**

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

### M7 — 接回 ROS seam(1 天)
`policy_node` 訂閱相機與 `/joint_states`、發布 `/sync/command`。
**完全複製 `omx_bridge_image/src/omx_bridge_app/omx_bridge_app/jog.py` 的寫法**:
行程內起 `SyncNode` 強制 `mode:=command`、背景執行緒 spin、只用公開介面。
✅ 同一個策略透過 bridge 驅動 Isaac,成功率相當。
**這一步之後,指向真手臂只差一個 `targets:=real` 加一台真相機。**

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
├── assets/
│   ├── pick_cube.usd          # 產生的;sublayer 指向 omx_f.usd
│   └── omx_f_descriptor.yaml  # Lula cspace(手寫)
├── sim/                       # Isaac python.sh — CPython 3.11,無 ros、無 torch
│   ├── config.py              # 讀 task yaml + sim_real_bridge.profile(關節順序)
│   ├── build_scene.py  scene.py  ik.py  expert.py
│   ├── record.py              # 專家 → data/raw/
│   ├── eval.py                # 策略 → 成功率(訓練區 vs 保留區)
│   └── policy_client.py       # zmq client,跨 3.11/3.12 界線
├── ml/                        # py3.12 uv venv — torch(單獨從 cu128 裝)+ lerobot
│   ├── convert.py             # data/raw/ → LeRobot(ACT v3.0 / GR00T v2+modality.json)
│   ├── train_act.sh  train_groot.sh
│   └── serve.py               # 策略後面掛 zmq
├── src/vla_app/               # ROS 2 py3.12 — M7 才用
│   └── policy_node.py
└── data/                      # gitignored
```

---

## 9. 不變條件(違反了會很難查)

1. **絕不修改 `omx_f.usd`** —— 它是 sim↔real 校正鏈的一部分。所有增修都是
   `pick_cube.usd` 裡的 override。
2. **絕不對 `omx_f.usd` 用 `AddReference`** —— 用 sublayer,見 §3.1。
3. **關節順序只有一個來源** —— `import sim_real_bridge.profile`,不要抄常數。
4. **`action` 必須是合法的 `/sync/command` payload** —— 6 個關節、同順序、弧度、
   USD 座標、絕對值。
5. **`observation.state` 是量測值,不是命令值。**
6. **torch 一定單獨從 cu128 索引裝**,不讓任何 requirements 檔釘它。
7. **不要升級 Isaac Sim。** 版本已定為 `5.1.0-rc.19`,兩台機器一致。
   這是**動工前評估過的決定,不是慣性**——6.0 確實是 Python 3.12(可消掉 §4.4 的 zmq
   邊界)且能原生 source ROS 2 Jazzy(讓 M7 變簡單),但:
   (a) 它目前是 **Early Developer Release、不是 GA**;
   (b) binding 的 public import path 搬了位置,§5 查證過的每個 API 都要重驗;
   (c) 5.1 已在這台 Blackwell 上實際跑起來過,相容性風險已退場;
   (d) 兩台機器共用 `omx_f.usd`,版本分家會製造難查的差異。
   第一個 ML 專案不該蓋在 pre-release 上——除錯時會分不出 bug 是 Isaac 的還是自己的。
   **6.0 到 GA 之後,在 M6/M7 時再重新評估。** `rc.19 → 5.1.0 GA` 也不要換,
   除非真的撞到可歸因於 RC 的 bug。
8. **只有成功的回合進資料集。**

## 10. 尚未查證的事(依賴前先確認)

- Lula 能不能直接吃這份 URDF(§5)
- `SingleArticulation(prim_path="/omx_f")` 對不對(用 `arm.dof_names` 確認)
- 伺服器的 ROS distro(影響 M7 落在 Humble 還是 Jazzy)。Ubuntu 已確認是 24.04.4,
  系統內建 Python 3.12.3
- M5 的訓練速度估計(10–20 steps/s 是推測,先跑 500 步量)
- GR00T 要的 LeRobot 格式版本與 lerobot 0.6 寫出的 v3.0 是否有落差(§4.4 的設計正是為此)

## 11. 參考

- NVIDIA/Isaac-GR00T — https://github.com/NVIDIA/Isaac-GR00T
- GR00T N1.5 微調 SO-101 — https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning
- GR00T in LeRobot — https://huggingface.co/blog/nvidia/nvidia-isaac-gr00t-in-lerobot
- LeIsaac(Isaac 裡遙操作 → GR00T)— https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/
- LeRobotDataset v3.0 — https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3
- sm_120 確認 — https://github.com/pytorch/pytorch/issues/157549
