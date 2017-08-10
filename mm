nvidia-docker run -d -p 6817:6817 -p 6818:6818 -p 2222:22 -v /dockerdata/:/dockerdata/ -h h1 slurm-mm
nvidia-docker run --rm -it -p 6817:6817 -p 6818:6818 -p 2222:22 -v /dockerdata/:/dockerdata/ -h h1 slurm-mm
