# OMX 抓取專案 — 給你看的版本

要做的事:讓 omx_follower 在 Isaac 裡學會把方塊夾起來,**模型只能看影像、不能知道方塊
座標**,最後微調 NVIDIA 的 GR00T。

這份是決策與進度用的。實作細節在 [`SERVER_CLAUDE_BRIEF.md`](SERVER_CLAUDE_BRIEF.md),
那份是要複製給伺服器上的 Claude 的。

---

## 一句話架構

> Isaac 裡放一個方塊和五台相機,寫個腳本專家用 IK 抓幾百次錄成資料集,
> 訓練一個只看影像的策略,最後換成 GR00T。全部在伺服器上,真手臂完全不參與。

---

## 為什麼是這個形狀

**為什麼只能在 sim 裡?**
你沒有裝在手臂上的相機。Isaac 的相機看到的是模擬的方塊,拿它驅動真手臂會夾到空氣。
要讓 sim 的方塊對上真方塊,就得先感知真方塊——那又需要真相機。所以閉迴路必須留在 sim。

**那 D455 呢?**
它讓**日後**的 sim2real 可行。現在就把 Isaac 主相機的內參對齊 D455
(`camera/cameracalibration/isaac_camera.py` 就是做這個換算的),日後只要把影像來源
從 Isaac 算繪換成 D455 的 topic,策略不用重訓。

**為什麼腳本專家而不是你手拉示範?**
一晚幾百筆 vs 一下午幾百次手拉,而且你每改一次方塊大小、相機位置或夾爪調校就要重來。
GR00T 只吃 LeRobot dataset,不在乎資料怎麼來。人類示範留到 M8,那時在你筆電上收。

**為什麼先 ACT 再 GR00T?**
兩者吃同一份資料。ACT 小、一兩小時就跑完、好除錯。先用它確認**資料是可學的**,
之後 GR00T 若效果不好,你能立刻分辨是資料問題還是模型問題,而不是卡在 3B 模型裡瞎猜。

**為什麼不用 Isaac Lab / LeIsaac?**
NVIDIA 的 LeIsaac 配方綁 Isaac Lab 2.1 + Isaac Sim 4.5/5.0,你裝的是 5.1.0。
借它的想法就好。(附帶一提,Isaac 內建的 `PickPlaceController` / Lula IK 最後也沒用 ——
控制程式跑在容器裡,碰不到 Isaac 的直譯器。運動學是自己的解析解加 MoveIt。)

---

## 已經定案的決定

| 項目 | 決定 |
|---|---|
| 成功定義 | sim 裡方塊被夾起並抬高 |
| 相機 | 錄五台(四台環境 + 腕上),**訓練時再決定用哪些**。已先轉出 3 台的版本 |
| 示範來源 | 腳本專家(M8 再補人類示範,用你筆電) |
| 模型 | 先 ACT,再 GR00T N1.7 |
| 跑在哪 | 全在伺服器。筆電只當觀看端 |
| 真手臂 | **完全不參與**。但 ⚠️ 見下方「真手臂為什麼不會動」 |
| Isaac 版本 | `5.1.0-rc.19`,**兩台已確認一致**,不重裝、不升 6.0 |

---

## 進度

- [x] **M0** 環境查證 — ✅ **完成**,見下方
- [x] **M1** 場景 + 相機 + 夾起方塊 — ✅(關鍵是 mimic 剛度 25→1000;後來擴到五台相機)
- [x] **M2** 腳本專家 — ✅ **moveit 20/20、analytic 20/20**
- [x] **M3** 走通骨架:錄 → 訓 200 步 → 手臂會動 — ✅ **完成**
- [x] **M4** 錄資料集 — ✅ **500 集跑完,486 集成功**(五台相機全存),
      已轉出 3 台版本 `data/lerobot_3cam`(486 集 / 100,096 幀)
- [ ] **M5** 訓 ACT + **在沒看過的位置評估** ← **現在在這裡**
- [ ] **M6** 微調 GR00T N1.7
- [x] **M7** 接回 `/sync/command` — ✅ 提早做掉了,專家與錄製本來就走這條
- [ ] **M8** 人類示範(日後)

停在任何一步都有東西:M2 有腳本抓放 demo、M5 有訓練好的策略、M7 有可部署的。

---

## ⚠️ 真手臂為什麼不會動

早期的說法是「連 ROS 都不接」。**那已經不成立了** —— 整套現在跑在 ROS 上,而且用
`ROS_DOMAIN_ID=1`,跟真 follower 同一個 DDS domain。專家發的是 `/sync/command`,
那正是 `jog real` 用來驅動真臂的同一個 topic。

真臂不動的唯一原因是 [`expert.py`](src/data_collection/data_collection/expert.py)
起引擎時寫死了 `targets:=sim`。引擎只把命令轉給被指名的端點,`sim` 就只到 Isaac。

**所以:改那一行之前先想清楚,而且不要在真 follower 通電時做實驗。** 這不是理論
風險 —— 筆電上 `omx_follower` 容器常常是開著的。

---

## M0 已完成 — 實測結果

| 項目 | 結果 |
|---|---|
| 伺服器 | `screamviolin`,Ubuntu 24.04.4,**系統內建 Python 3.12.3** |
| 驅動 | 580.159.03(支援到 CUDA 13.0) |
| GPU | RTX PRO 6000 Blackwell,97,887 MiB |
| **torch** | `2.11.0+cu128`,`capability=(12,0)`,**matmul 真的跑出數字** |
| **`docker --gpus all`** | ✅ 可用,nvidia-container-toolkit 已設好,免 sudo |
| repo | 兩個都 clone 完成,`omx_f.usd` 與筆電位元組一致 |

三個附帶結論:

- **不用 uv、不用 conda、也不用 venv** — 環境全部收進 docker。ML 那層是衍生 image
  `omx_vla_image`(`FROM omx_bridge_image` + torch cu128 + lerobot)。pip 的 torch
  wheel 自帶 CUDA runtime,**不需要裝 CUDA toolkit**。
- ⚠️ **Isaac 附的 `environment.yml` 釘 `cuda-toolkit=11.8`,在 Blackwell 上不能用。**
  不要拿它建 conda 環境跑 torch。
- **ML 那層跑容器**(符合你既有做法),**Isaac 留原生**。GPU 是共用的,還有 92 GB,
  可以跟別人並存不用等。

## 下一步:M5 —— 訓 ACT,在沒看過的位置評估

M3 的三個接縫都通了,M4 的資料也錄完了。現在手上有:

```
data/raw/            486 集,五台相機都存,格式無關的原始傾印
data/lerobot_3cam/   486 集 / 100,096 幀,front_left + front_right + wrist
```

接下來:

1. **先量 dataloader 速度再開大訓練。** M3 時量到 2 step/s、dataloader 佔 94%,
   那是 19 集的數字;486 集要重量一次,調 `num_workers` 與 `batch_size`。
2. `lerobot-train --policy.type=act` 跑到收斂。
3. **`eval.py --holdout`** —— 驗收條件是保留區的成功率明顯 >0 且與訓練區差距不大。
   訓練區高、保留區接近 0 就是「模型在背軌跡而不是看影像」,回頭加大隨機化。
4. 相機對照:目前轉的是 3 台,同一份 `data/raw/` 可以再轉 2 台或 5 台的版本比較
   (`ml/convert.py --cameras`),不必重錄。

實作細節在 [`SERVER_CLAUDE_BRIEF.md`](SERVER_CLAUDE_BRIEF.md)。

---

## 會浪費你時間的三個坑

**1. 夾爪參數沒定好就開始錄資料**
指尖法向力約 **~0.14 N**(可夾 25–30 g),而方塊設 15 g——**在預算內**,所以
「夾不起來」本身不太可能發生,而且真發生了在 sim 裡也只是轉旋鈕:方塊變輕、摩擦調高、
stiffness 625→5000。

**唯一「變輕」解決不了的是接觸幾何**:手指 collider 是 `convexHull`,實測扁平比只有
3.7、只有 3–6% 頂點貼在外側面,所以凸包沒有平整夾持面。要用 `convexDecomposition`
或加兩片薄板 pad。

真正會浪費時間的不是這些參數難調,而是**錄完 200 集才發現要改**——改了質量或 collider,
先前的資料就全部作廢。所以 M1 先花半小時把旋鈕轉對。

**2. 模型在背軌跡,不是在看影像**
如果方塊位置隨機化得不夠廣,策略會學成「無視影像、照平均軌跡揮一遍」——訓練分數很漂亮,
但它根本沒在看。

**M5 的驗收條件是在訓練從未見過的位置區域評估。** 這不是可選項。訓練區高、保留區接近 0
就是這個病。

**3. PyTorch 裝到不支援 Blackwell 的版本**
RTX Pro 6000 是 sm_120,需要 **torch ≥ 2.7 + CUDA 12.8**。很多機器人套件的
`requirements.txt` 還釘在 cu118/cu121,裝下去會是 `no kernel image is available`。

**永遠先從 cu128 索引單獨裝 torch,再裝其他東西。** 好消息是 GR00T 自己就釘
torch 2.7 / CUDA 12.8,方向是對的。

---

## 之後才要決定的事

- **要不要買腕上相機** — M5 會跑「只用胸口」vs「胸口+腕上」的對照,差距小就別買。
- **要不要補人類示範** — 腳本專家的資料近乎單模態,若 M5 泛化不好,M8 補人類資料。
- **真手臂的安全夾限** — URDF 限位是 ±360° 全開,擋不住任何東西。sim2real 之前必須補。

---

## 相關路徑

這個 repo 是獨立的(`github.com/jcjcjc0705/vla`),依賴另外兩個 repo 但**不含**它們。

**這隻手臂的資料只有一個來源:`omx_bridge_image` 這個 image。** 不再列每台機器的
checkout 路徑 —— 那個做法讓每台機器可能拿到跟正在跑的 image 不同版本的 USD,而症狀是
「場景看起來對但數字不對」這種最難查的東西。

| 東西 | 從哪來 |
|---|---|
| USD / URDF / meshes / profile | `omx_bridge_image`。容器內烤在 `/`;host 上跑 `bash isaac/fetch_assets.sh` 匯出到 `.image_cache/` |
| 同步引擎(關節順序的真相) | 同上。連 `profile.py` 一起匯出,不抄第二份 |
| 訓練環境(torch + lerobot) | `omx_vla_image`,`FROM omx_bridge_image` |
| 相機內參換算工具 | `camera/cameracalibration/isaac_camera.py`(在使用者筆電的另一個專案裡) |

⚠️ `docker pull` 了新的 `omx_bridge_image` 之後要重跑 `fetch_assets.sh`。`.image_cache/`
是複本、沒有版本標記,不會自己更新。
