cd /tmp

echo "********** CREATING CONDA ENVIRONMENT **********"
conda create --name volumetric_cpu -f cpu.yml
source activate volumetric_cpu