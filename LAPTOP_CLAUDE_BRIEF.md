# 簡報:給筆電上的 Claude —— 伺服器這邊的進度與兩個必須知道的修正

> **先讀 `SERVER_CLAUDE_BRIEF.md`。** 那份是專案的完整背景(物理參數、架構決策、
> 每個踩過的坑),仍然有效。**這一份只寫「自從那份寫完之後,在伺服器上發生了什麼」**,
> 以及筆電這邊該同步什麼。
>
> 日期:2026-08-07。伺服器 = `screamviolin`(140.116.82.227),RTX Pro 6000 96 GB。

---

## 0. 一句話總結

**M3 完成了**(`convert.py` → `lerobot-train --steps=200` → `eval.py` 讓手臂因 checkpoint
而動),過程中在伺服器上發現並修掉**兩個 bug**,兩個都動到 `expert_node.py`。
其中一個在筆電上不會發作,另一個**會**。

---

## 1. ⚠️ 兩個修正 —— pull 之後你會拿到,先知道為什麼

兩個都在 `src/omx_vla_app/omx_vla_app/expert_node.py`。**不要退掉。**

### 1.1 控制迴圈改用模擬時間,不再用 wall clock

`run_episode` 裡原本是 `time.sleep(period)`,現在是 `client.wait_sim(period)`。

**為什麼:Isaac 在 headless streaming 下不限速。** 伺服器實測模擬跑 **2.42 倍**真實時間,
所以 wall-clock 的控制迴圈每個 tick 之間物理實際走了 **80.6 ms 模擬時間**而不是 33.3 ms。
後果不是「有點不準」,而是:

```
                    成功率    夾合俯仰角        落後命令
wall clock          11/20     -12.2~72.0°      7.4 / 21.3 mm
sim time            17/20      -4.5~70.6°      3.4 / 13.1 mm
sim time + §1.2     19/20      46.1~59.1°      2.8 /  4.5 mm
```

時序偏移改變了 position-only IK 的種子軌跡(§5:零空間 2 DOF、沒有選擇準則),
俯仰角就散掉,散到負值之後大片位置根本無解 —— 9 集連續「抓取點無解」。

⚠️ **kit 參數那條路是死的。** `rateLimitEnabled`、`rateLimitFrequency`、
`useFixedTimeStepping`、`useFastMode` 四個都用命令列傳過(確認有進 kit 的 cmdline),
**全部無效**。`base.kit:151` 為了 MGPU 穩定性關掉了 `present.enabled`,headless 下
沒有 swapchain present 可以掛節流。不要再試這條。

**筆電上這個修正不會改變行為** —— RTX 2060 算繪兩台相機本身就是限速器,realtime factor
大約 1.0。但**它讓兩台機器產出同構的資料**,這是 M8(把筆電的人類示範併進同一個 dataset)
的前提:兩台速度不同,只有 sim time 對齊才併得起來。

實作上它靠的是 `/joint_states` —— 那個 topic 由 `omx_f.usd` 的 `on_playback_tick` 驅動、
由 `isaac_read_simulation_time` 蓋章,所以「等它推進 1/30」就等於「等兩個 playback tick」,
不必寫死 tick 頻率。

⚠️ **realtime factor 不是常數。** 伺服器上跑訓練搶 GPU 時,Isaac 從 145 FPS 掉到 72 FPS,
factor 從 2.42 掉到 1.2。GPU 是共用的(§0),所以「當下有沒有別人在用這張卡」本來會
悄悄改變錄出來的資料。改用 sim time 之後這個變數消失了。

`wait_sim` 另外帶一個 `loop_overruns` 計數器:如果某個 tick 自己的運算(主要是 IK)
就花掉超過一個控制週期的模擬時間,那迴圈等於又退化成「盡快跑」。它會在每次 run 的
總結印出來。目前 moveit / analytic 都是 0。

### 1.2 `expert.reset()` 之前要重設 IK 種子 —— **這個在筆電上也會發作**

`run_episode` 在 `expert.reset(...)` 前面多了:

```python
if hasattr(kin, "seed"):
    kin.seed = client.joints()[:5]
```

**為什麼:** `expert.reset()` 內部會解 IK(`sim/expert.py:76` 的 `self.kin.solve(...)`),
但種子每個 tick 重設那行在**迴圈裡面**,而 reset 在迴圈**外面**。所以規劃用的種子是
**上一集最後一個 tick 留下的**——手臂舉在空中抓著方塊的姿態。

失敗因此是**成塊的**,不是零星的:一集結束得不好 → 下一集規劃失敗 → 那集不進迴圈 →
種子沒被更新 → 再下一集用同一個壞種子 → 繼續失敗,直到某次擾動把它撞出來。
伺服器上觀察到 14–16 集連續倒在一個異常的第 13 集後面,修 §1.1 之前更是連續 9 集。

這是簡報 M2 那條原則(「MoveIt 的種子每個 tick 從量到的關節值重設」)漏掉的一條路徑,
**跟機器速度無關**。筆電上碰巧沒撞到,不代表不存在。修完之後手臂在 reset 時一定在 home,
所以**每一集的規劃變成可重現的**,不再取決於上一集怎麼結束。

---

## 2. 伺服器上已經驗證跑通的功能(逐項實跑)

| 功能 | 指令 | 伺服器結果 |
|---|---|---|
| 設定解析 | `bash sim/isaac_python.sh sim/task_config.py` | ✅ 6 關節、路徑全解析 |
| 建場景 | `bash sim/isaac_python.sh sim/build_scene.py` | ✅ prim 238→249,mimic 25→1000 |
| 解析解自檢 | `sim/ik.py --check 1000` | ✅ 100%,FK 往返 < 1 µm |
| 可達範圍 | `sim/ik.py --map` | ✅ 環帶最高懸停 **110 mm** |
| 對 Isaac 驗運動學 | `sim/ik.py --isaac 20` | ✅ **0.025 mm**(簡報記 0.024) |
| 專家 moveit | `expert -p episodes:=20` | **19/20**,俯仰 46.1~59.1° |
| 專家 analytic | `expert -p ik:=analytic -p episodes:=20` | **20/20**,俯仰 88.6~89.6° |
| 專家 position_only | `expert -p ik:=position_only -p episodes:=10` | **10/10**,俯仰 36.3~44.7° |
| 保留區取樣 | `expert -p holdout:=true -p episodes:=10` | **10/10**,θ 全落在保留帶 |
| 階段截圖 | `expert -p save_frames:=true` | ✅ 每集 5 階段 × 2 相機 |
| 錄製 | `record -p episodes:=20` | ✅ 19 集 / 3836 幀 |
| 轉檔 | `python ml/convert.py` | ✅ 載得回,shape 全對 |
| 訓練 | `lerobot-train --steps=200` | ✅ loss 23.8 → 3.38 |
| 評估 | `python ml/eval.py --checkpoint ...` | ✅ 455 則命令、無 NaN、手臂動了 |
| `jog` / `monitor` | `ros2 run omx_bridge_app jog sim` | ⏸ 需互動終端,伺服器上的 agent 驗不了 |
| `ik_target` | `ros2 run omx_bridge_app ik_target` | ⏸ 需 GUI 拖曳,同上 |

**三個解算器的差異值得記住**(同一組 seed、同一個場景):

- `analytic` 夾爪固定朝下 → 俯仰 **89°**,沒有零空間,最穩,20/20。
- `position_only` (DLS 替身) → 俯仰 **36~45°**,正好是簡報記的 32–44°。
- `moveit` (真的 KDL) → 俯仰 **46~59°**,19/20。比簡報記的筆電值(32–44°)高,
  因為 §1.2 把種子從「上一集殘留」換成了「home」,零空間落點自然不同 ——
  **但現在它是可重現的**。

⚠️ `--map` 說整個環帶都能懸停的最高高度是 **110 mm**,但簡報 §spawn 的註解寫
「r=0.24 最高只能懸停 85mm」。**這兩個數字對不上,請在筆電上跑一次 `--map` 對照。**
若筆電也是 110,那簡報那句要更新。

---

## 3. 簡報 §10「尚未查證的事」——有答案了

### 3.1 轉檔器該怎麼處理過期影像 → **不必特別處理,但不是零重複**

伺服器上錄 19 集實測:

```
image_age_s   平均 4.59 ms   最大 31.07 ms
唯一影像/幀   0.865            ← 86.5% 的 tick 拿到全新像素
```

相機在**模擬時間**上約 **26 Hz** 對 30 Hz 控制迴圈,13.5% 的 tick 重複像素。
筆電是 12 Hz 對 30 Hz(約六成重複),所以伺服器好很多,但不是 1:1。
`convert.py` 目前**照 index 展開、不丟不插值**,只在統計裡報告。

⚠️ **`recorder.py` 的 age 仍然是 wall clock 記的**,而控制週期現在是 sim time。
在 2.42x 下一個控制週期在 wall 上只有 13.8 ms,所以拿 4.59 ms 直接跟 33.33 ms 比會
**過度樂觀**。`0.865` 那個數字不依賴時鐘換算,是目前唯一能直接採信的。

**這是一個還沒做的修正**:控制迴圈改用 sim time 之後,§4.3 那句「影像 stamp 與控制
迴圈是兩個時鐘、不能相減」的前提已經不成立 —— 影像的 `header.stamp` 跟 `/joint_states`
的 stamp 現在同基準,`age = sim_now - img_stamp` 才是對的。改動落在 `recorder.py` 與
`expert_node.py` 的 `_on_image`。**還沒動,因為它只影響診斷數字的判讀,不影響資料本身。**

### 3.2 訓練速度 → **2 step/s,dataloader 是瓶頸**

200 步、batch 8、`num_workers=4`、19 集 3836 幀:

```
updt_s: 0.028      ← GPU 更新
data_s: 0.472      ← dataloader   ★ 佔 94%
```

簡報 M5 估 10–20 step/s 並說「低於 8 就是 dataloader 瓶頸」——確認是。`use_videos=False`
已經生效,所以要調的是 `num_workers`(以及 M5 時 batch 拉大)。**M5 開始前先量。**

---

## 4. 簡報 §6 的程式碼片段跟 lerobot 0.6.1 對不上(兩處,照抄會直接報錯)

```python
# §6 寫的                          # 實際要的
"dtype": "video"   + use_videos=False   →   "dtype": "image" + use_videos=False
"shape": [3,240,320]  (list)            →   "shape": (3,240,320)  (tuple)
```

1. `dataset_metadata.py:850` 明確 raise:features 含 video key 但 `use_videos=False`。
2. `validate_feature_numpy_array` 用 `actual_shape != expected_shape` 比較,
   `ndarray.shape` 是 tuple,list 永遠不相等 → 每一幀都被拒,錯誤訊息還會印出兩個
   看起來一模一樣的 shape。

另外**確認過可以放心的**:`validate_feature_image_or_video` 同時接受 `(C,H,W)` 與
`(H,W,C)`,所以 PNG 讀出來直接餵,不必 transpose。

還有兩個 0.6.1 的坑:

- **正規化搬出 policy 了。** checkpoint 裡有 `policy_preprocessor_*` /
  `policy_postprocessor_*`,正確流程是
  `preprocessor(obs) → policy.select_action → postprocessor(action)`,
  用 `make_pre_post_processors(policy_cfg=..., pretrained_path=...)` 建。
  直接呼叫 `select_action` 會餵進未正規化的弧度與像素、拿到未反正規化的動作 ——
  **手臂照樣會動,不會有任何錯誤**。preprocessor 內含 `to_batch_processor` 與
  `device_processor`,所以觀測要給**未加 batch、CPU 上**的張量。
- **功能拆成 extras**:`pip install lerobot` 之後還要 `lerobot[dataset]` 與
  `lerobot[training]`,否則 import 或訓練時才報。

---

## 5. 環境:伺服器上怎麼跑起來的(筆電對照用)

```bash
# 1. 一次性:.env(0.2.2 起不進 repo)
cd omx_bridge_image && cp docker/compose/.env.example docker/compose/.env
#    ISAAC_SIM_PATH=/home/pochun/isaac_sim_5.1 , ROS_DOMAIN_ID=1

# 2. Isaac(host 原生)
bash scripts/isaac.sh --streaming        # 然後手動載 vla/assets/pick_cube.usd、按 Play

# 3. 控制層容器
cd ../vla && docker compose --env-file docker/compose/.env \
  -f docker/compose/docker-compose-vla.yml up -d
docker exec omx_vla bash -c 'source /workspaces/rebuild_colcon.rc'   # 第一次要 build
```

⚠️ **容器裡 `bash -lc` 不會 source `~/.bashrc`**,所以 ROS 沒被載入、`ros2` 不在 PATH。
非互動執行一律要明寫:

```bash
docker exec omx_vla bash -c 'source /opt/ros/jazzy/setup.bash \
  && source /workspaces/install/setup.bash && <指令>'
```

**ML 那一層在一個新 image 裡,而且環境全部收進 docker —— 沒有 venv,沒有 conda。**

新 repo:`gitlab.screamtrumpet.csie.ncku.edu.tw/pochun/omx_vla_image`

```
sim_real_bridge_image      機器人無關的同步引擎
└── omx_bridge_image       這隻手臂的資料:profile、assets、MoveIt、jog
    └── omx_vla_image      + torch(cu128) + lerobot[dataset,training]   ← 新增
```

### ⚠️ 但是要兩個容器,因為 MoveIt 與 lerobot 無法共存

**lerobot 硬性要求 `numpy>=2.0`,而 `moveit_py` 是 C++ binding、numpy 2.0 破壞了
C ABI。** 實測(2026-08-07,伺服器):

| numpy | 指令 | 結果 |
|---|---|---|
| 2.2.6 | `expert`(預設 `ik:=moveit`) | **Segmentation fault** |
| 1.26.4(強制 PYTHONPATH) | 同上 | 20/20 |
| 2.2.6 | `expert -p ik:=analytic` | 正常 |

⚠️ **它 import 完全正常,執行時才 segfault,沒有任何訊息。** 這份簡報早期的版本就是
靠 import 檢查下結論說 MoveIt 沒事 —— 那是錯的,已更正。這是二進位相容性,不是設定
問題,真解是拿 numpy 2.x 重編譯 moveit,不在守備範圍。

所以界線用**兩個容器**畫死,而不是靠誰記得設環境變數:

| 容器 | compose | image | numpy | 做什麼 |
|---|---|---|---|---|
| `omx_vla` | `docker-compose-vla.yml` | omx_vla_image | 2.x | 轉檔 / 訓練 / 推論 / `ik:=analytic` |
| `omx_vla_ctrl` | `docker-compose-ctrl.yml` | omx_bridge_image | 1.x | `ik:=moveit`、`jog`、`ik_target`、`monitor` |

```bash
bash docker/scripts/vla.sh     # 學習層
bash docker/scripts/ctrl.sh    # 控制層(要 MoveIt 時)
```

⚠️ 兩個容器可以並存,但**一次只能有一個行程發 `/sync/command`** —— 兩邊都要 pgrep。

⚠️ **`vla.sh` / `ctrl.sh` 的 `up` 模式在你離開那個 shell 時會 `docker compose down`**
(見 `utils.sh` 的 `cleanup`)。容器會消失,這是設計不是 bug。

**`eval.py` 用 analytic 解算器,所以推論完全不受這個衝突影響。** 而且 `analytic`
專家實測 **20/20**,比 `moveit` 的 19/20 還好,俯仰角也更穩(88.6~89.6° vs 46~59°)。

> **這修正了簡報 M7 的假設。** M7 說推論「要嘛把 torch 裝進衍生 image,要嘛另開
> 一個容器」—— 答案是前者。理由是 `eval.py` 必須在**同一個行程**裡跑策略(torch)
> 並發布 `/sync/command`(rclpy),兩者得共用直譯器。
>
> 中途曾經用 host 的 venv 掛進容器跑通過(host 與 image 都是 Ubuntu 24.04 /
> py3.12.3 / glibc 2.39,連 patch 都一樣)。**那能動是巧合不是設計**,base image
> 的基底一動就會壞在 `symbol not found`,而且 7.6 GB 的東西完全不在版本控制裡。
> 已廢棄並刪除。

⚠️ **筆電不需要 pull `omx_vla_image`。** RTX 2060 6 GB 跑不動訓練,M8 的人類示範只
需要 `omx_bridge_image`。clone 那個 repo 是為了看程式碼與同步進度。

⚠️ **另一個 pip 的坑(已處理)**:apt 裝的 python 套件沒有 RECORD 檔,pip 卸載不了
(`error: uninstall-no-record-file`)。Dockerfile 用 `--ignore-installed` 把新版寫進
`/usr/local`(sys.path 優先),而且**只針對會衝突的那幾個**
(`numpy packaging pillow setuptools pyyaml psutil`)—— 對整包 lerobot 下
`--ignore-installed` 會連 torch 一起重裝、且會從 default index 拉,正是不變條件 6
要防的事。

⚠️ 學習層容器要 GPU —— `docker-compose-vla.yml` 有
`deploy.resources.reservations.devices`。少了它 lerobot 會**靜默**退回 CPU,
然後在 `.to("cuda")` 才炸出一個看起來無關的 driver 錯誤。

---

## 6. 這次新增/改動的檔案

**新 repo `omx_vla_image`** —— 只有環境,沒有程式碼:

```
Dockerfile        FROM omx_bridge_image + numpy2/torch cu128 + lerobot[dataset,training] + build guard
.gitlab-ci.yml    照抄 omx_bridge_image 的,只在 tag 時 build 並推 :tag 與 :latest
README.md
```

**`vla` repo:**

```
ml/                                  ← 新增。訓練/推論那一層,與 sim/(Isaac 3.11)、src/(ROS) 並列
├── convert.py                       raw dump -> LeRobotDataset
└── eval.py                          checkpoint -> /sync/command
src/omx_vla_app/omx_vla_app/expert_node.py   ← §1 的兩個修正 + loop_overruns 計數
docker/compose/docker-compose-vla.yml        ← 指向 omx_vla_image + GPU reservation
docker/compose/docker-compose-ctrl.yml       ← 新增。控制層(omx_bridge_image,numpy 1.x)
docker/scripts/ctrl.sh                       ← 新增
SERVER_CLAUDE_BRIEF.md                       ← 更新:§6 的 lerobot 片段、§10 的答案、§9 新增兩條
LAPTOP_CLAUDE_BRIEF.md                       ← 這份
```

**職責邊界**:`omx_vla_image` 只放環境(訓練本身是 `lerobot-train` CLI,不需要自己寫的
程式碼);`vla` 放這隻手臂的知識 —— 場景與運動學、腳本專家、ROS 節點、轉檔、推論、測試。
`convert.py` 留在 `vla`,因為它要讀 `data/raw/` 並從 `sim/task_config.py` 取關節順序
(全鏈唯一的真相),它屬於資料層而不是訓練層。

`omx_bridge_image` **完全沒有改動**(`.env` 是 gitignored 的機器設定)。

`data/` 底下(全部 gitignored):

```
data/raw/                    19 集,目前唯一有效的資料(sim time + 種子都修好之後錄的)
data/raw_2.42x_bad/          11 集,時序壞掉時錄的 —— 只留著對照,不要拿去訓練
data/raw_simtime_seedbug/    17 集,時序好了但種子還沒修
data/lerobot/                19 集 / 3836 幀的 LeRobotDataset
outputs/act_m3/              200 步的 checkpoint
```

⚠️ **容器以 root 執行**,所以 `record` 寫出來的檔案是 `root:root`,host 端(以及在
host 上跑的 `convert.py`)會寫不進去。每次錄完跑一次:

```bash
docker exec omx_vla chown -R $(id -u):$(id -g) /vla/data
```

---

## 7. 兩台機器的差異(M8 合併資料前要知道)

| | 筆電 | 伺服器 |
|---|---|---|
| Isaac realtime factor | ~1.0(GPU 就是限速器) | **2.42**,且隨負載浮動 |
| 相機速率(模擬時間) | ~12 Hz | ~26 Hz |
| 每 tick 重複像素 | 約 60% | **13.5%** |
| 控制迴圈時鐘 | — | 兩台都改用 sim time(§1.1)後**一致** |

⚠️ **一個還沒解決的 domain gap:光照。** 伺服器的 Isaac 是從筆電複製過來的,
`omni.kit.stage_templates` 這個擴充套件裡的資源路徑仍寫死
`/home/jcjcjc/Desktop/isaac_sim_5.1/...`,所以預設 stage 的 DomeLight HDR 與一張地板
材質**在伺服器上載不到**(每次開 Isaac 都會刷這幾行 Error,連 `ik.py --isaac` 開的
全新行程也一樣)。算繪本身正常、有光有陰影、相機影像可用,**但兩台的光照很可能不同**。

**請在筆電上存一張 `cam_front` 的影像跟伺服器的比對** —— 併資料集之前這件事要確認。
伺服器這邊的樣本在 `data/diag/`(用 `expert -p save_frames:=true` 產生)。

---

## 8. 現在的狀態與下一步

- **M0/M1/M2/M3/M7 完成。** M3 的三個接縫都通了,而且是用**這台機器上重新錄的**
  19 集資料走完的。
- **M4(200 集無人值守)可以開始了** —— 但先確認 §1 兩個修正都在,不然錄出來的
  200 集會是壞資料。專家的雜訊注入(`task/pick_cube.task.yaml` 的 `expert.noise`)
  已經在設定裡,`record` 會直接用。
- **沒有加任何新功能。** `ml/convert.py` 與 `ml/eval.py` 是 M3 本來就要交付的兩支,
  其餘都是修既有行為或搬環境。
- **環境全部收進 docker,不再有 venv/conda。** 見 §5;這也讓「兩邊透過 git 同步」
  真的成立 —— 環境的定義本身進了版本控制。

**往 GR00T 的方向**(使用者的最終目標,簡報 M6):

- 目前 dataset 是 `use_videos=False`(PNG in parquet)、LeRobot v3.0,這是 ACT 要的。
  **GR00T 要的是 v2 變體 + `meta/modality.json`**,而且會需要影片而不是 PNG。
- 簡報 §4.4 的設計正是為了這件事:格式無關的 `data/raw/` 留著,換格式是**再寫一支
  轉檔器**(或給 `convert.py` 加一個輸出模式),不是重錄。
- 兩台相機的資料**都已經在錄**(`front` + `wrist`),M5 要跑的「只用 front」vs
  「front + wrist」對照,資料面已經備妥。

**已知還沒做的**(不是 bug,是待辦):

1. `recorder.py` 的 image age 改用 sim time(§3.1)。
2. 簡報 `SERVER_CLAUDE_BRIEF.md` 本身要更新:§6 的 lerobot 片段、§10 的兩個答案、
   M7 可以簡化、§1 兩個修正該進 §9 不變條件、`--map` 的 110 vs 85。
3. `jog` / `monitor` / `ik_target` 在伺服器上沒驗過(需要互動終端或 GUI)。
