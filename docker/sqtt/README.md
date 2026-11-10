# SQTT 采集镜像（kra-sqtt）

在 BW1000 上用 XProf 采集 SQTT，需要另外安装 xprofiler 和 hsa-aqlprofile，配置比较麻烦。这个镜像已经装好了这两个包。
拉取镜像后按下面的参数启动容器，就可以直接运行 `examples/kda/sqtt_capture.sh`。

## 镜像

| 项 | 值 |
|---|---|
| 镜像 | `42.228.13.241:5000/roofline-huc-tools/kra-sqtt:dtk26.04-xprof3.16.0-20261110` |
| 访问范围 | 郑州集群内网 Harbor（`http://42.228.13.241:5000/`，项目 `roofline-huc-tools`），外网无法访问 |
| 基础镜像 | `10.16.1.152:5000/jenkins/model_test_env/vllm:0.18.1-ubuntu22.04-dtk26.04-py3.10-20260518-2242`（DTK 26.04，`/opt/dtk`） |
| 新增内容 | xprofiler 3.16.0、hsa-aqlprofile 3.6.2，安装在 `/opt/rocm-6.3.3`；`PATH` 已包含 `/opt/rocm-6.3.3/bin` |
| 大小 | 约 30.3 GB，其中新增层不到 5 MB；节点上已有基础镜像时，只需拉取新增层 |
| 构建方式 | 用 [`Dockerfile`](Dockerfile) 构建，不是 `docker commit`，所以镜像里没有实验源码、`/tmp` 残留和个人凭据 |

两个安装包的 sha256：

```
94caf7052b853c63352d13c4e5dcf1fb22b56f631dbbb97bc442a669c36632a0  hsa-aqlprofile_3.6.2_amd64.deb
5dee008437ac24fde751d5f76e4ba1b41fc7194cf364f9509860870fb8718779  xprofiler_3.16.0_amd64.deb
```

安装包存放在 bw61 的 `/public/home/tanbo/kra_sqtt_image/` 下。镜像丢失时，可以用它们和 Dockerfile 重新构建。

## 启动容器

镜像只包含容器的文件系统。设备、权限和驱动库都要在 `docker run` 时指定，少了任何一项，都可能找不到卡或 SQTT 采集失败：

```bash
docker login 42.228.13.241:5000          # 首次拉取需要 Harbor 账号
docker pull 42.228.13.241:5000/roofline-huc-tools/kra-sqtt:dtk26.04-xprof3.16.0-20261110

docker run -d --name ${USER}_xprof \
  --device /dev/kfd --device /dev/dri --device /dev/mem \
  --group-add video --cap-add SYS_RAWIO --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --shm-size 64g \
  -v /usr/local/hyhal:/opt/hyhal:ro \
  42.228.13.241:5000/roofline-huc-tools/kra-sqtt:dtk26.04-xprof3.16.0-20261110 sleep infinity

# 自检：能找到 xprof，并且能看到卡
docker exec ${USER}_xprof bash -lc 'command -v xprof && /opt/hyhal/bin/hy-smi'
```

宿主机需要满足以下条件：

- 已安装 HCU 驱动（`hycu` 内核模块），`/dev/kfd` 和 `/dev/dri` 存在；
- 宿主机上有 `/usr/local/hyhal`。它不在镜像里，启动时以只读方式挂载到容器的 `/opt/hyhal`；
- Docker 的 `insecure-registries` 包含 `42.228.13.241:5000`（bw61 已配置）。

容器没有挂载 `/public`。`sqtt_capture.sh` 会用 `docker cp` 把源码复制进容器，采集完再把结果复制出来。

## 采集 SQTT

在**宿主机**上运行，不要在容器里运行。用 `XPROF_CONTAINER` 指定上面启动的容器：

```bash
cd kernel-roofline-analysis-tools
XPROF_CONTAINER=${USER}_xprof bash examples/kda/sqtt_capture.sh TREE TAG T H "kernel1,kernel2" [GPU]
# 例：XPROF_CONTAINER=${USER}_xprof bash examples/kda/sqtt_capture.sh ~/g2_opt demo 8192 12 "p3_wy_s4b_kernel" 0
```

`TREE` 是包含 `csrc/g2` 的源码树。结果写到 `/public/home/tanbo/kra_sqtt/TAG/`，这个路径固定写在脚本的 `OUT=` 一行里。
其他人使用时要把它改成自己的目录，并把 `KRA` 变量指向自己检出的仓库。

## 已验证的环境

| 节点 | 宿主机 | 验证内容 | 结果 |
|---|---|---|---|
| bw61 | Rocky Linux 8.6，内核 4.18.0-372.9.1.el8，8 张 BW1000（gfx936） | 用新镜像启动容器，对 G2 源码采集 `p3_wy_s4b_kernel`（T=8192，H=12，GPU 0） | 与原容器 `glm_bw7_xprof2` 的结果一致：dispatch 107，`.perf` 约 6.0 MB |

其他节点还没有验证，尤其是驱动已升级到更新版本的节点。第一次使用时，请先跑一遍上面的例子，确认能得到 `.perf`。
