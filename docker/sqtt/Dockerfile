# SQTT capture environment for kernel-roofline-analysis-tools (kra).
#
# Base: the DTK 26.04 image used on the BW1000 nodes. On top of it this adds only
# XProf (xprofiler 3.16.0) and hsa-aqlprofile 3.6.2, both installed under /opt/rocm-6.3.3.
# That is everything examples/kda/sqtt_capture.sh needs inside the container.
#
# The two .deb files are not in this repository. They are on bw61 under
# /public/home/tanbo/kra_sqtt_image/ (sha256 in docker/sqtt/README.md); copy them next to
# this Dockerfile, then build (the docker on the BW nodes has no buildx):
#
#   DOCKER_BUILDKIT=0 docker build -t 42.228.13.241:5000/roofline-huc-tools/kra-sqtt:dtk26.04-xprof3.16.0-20261110 .
ARG BASE=10.16.1.152:5000/jenkins/model_test_env/vllm:0.18.1-ubuntu22.04-dtk26.04-py3.10-20260518-2242
FROM ${BASE}
COPY hsa-aqlprofile_3.6.2_amd64.deb xprofiler_3.16.0_amd64.deb /tmp/kra_debs/
RUN dpkg -i /tmp/kra_debs/hsa-aqlprofile_3.6.2_amd64.deb /tmp/kra_debs/xprofiler_3.16.0_amd64.deb \
 && ldconfig && rm -rf /tmp/kra_debs
ENV PATH=/opt/rocm-6.3.3/bin:${PATH}
LABEL org.kra.purpose="SQTT capture (XProf) for kernel-roofline-analysis-tools" \
      org.kra.xprofiler="3.16.0" org.kra.aqlprofile="3.6.2" org.kra.dtk="26.04"
