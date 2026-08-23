#代码运行前请创建虚拟环境，python 3.9+ 版本，安装如下依赖：

numpy>=1.24
scipy>=1.10
matplotlib>=3.7

#将 code 文件夹下的三个 py 文件拉取好后

bash：cd 到指定目录下
bash：python3 -m venv .venv
bash：source .venv/bin/activate

#运行 additional_test_instances.py，输出 additional_outputs 文件夹

bash：python additional_test_instances.py --problems zdt3,zdt4,mw2,welded_beam,dtlz2,wfg3 --weights 31 --starts 4 --maxiter 100 --pienn-iter 12 --nc 2 --r-factor 0.3 --output-dir additional_outputs

#运行 compare_pienn_se_vprnn.py， 输出 comparison_outpus 文件夹

bash：python compare_pienn_se_vprnn.py --problems zdt3,zdt4,dtlz2,wfg3 --candidates 51 --starts 4 --maxiter 80 --pienn-iter 12 --n-gen 50 --pop-size 50 --cma-maxfevals 100 --cma-pop-size 8 --output-dir comparison_outputs

#运行 sensitivity_analysis.py，输出 sensitivity_outputs 文件夹

bash：python sensitivity_analysis.py --problem zdt4 --weights 11 --starts 4 --maxiter 80 --pienn-iter 12 --nc 2 --r-factor 0.3 --output-dir sensitivity_outputs
