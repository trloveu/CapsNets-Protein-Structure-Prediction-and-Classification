cd /tmp

echo "********** UPDATING OS **********"
sudo apt-get -y update && sudo apt-get -y upgrade && sudo apt-get -y dist-upgrade

echo "********** DOWNLOADING MINICONDA **********"
# checks if Miniconda3-latest-Linux-x86_64.sh exists, if so, install...
if [ ! -f Miniconda3-latest-Linux-x86_64.sh ]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi

sudo bash -c 'chmod +x Miniconda3-latest-Linux-x86_64.sh'
sudo ./Miniconda3-latest-Linux-x86_64.sh

echo ""
echo "######################################################"
echo "################## !!! ATTENTION !!! #################"
echo "####### CLOSE THE TERMINAL AND OPEN A NEW ONE ########"
echo "####### 			     THEN, RUN              ########"
echo "######################################################"
echo "### conda create --name volumetric_cpu -f cpu.yml ####"
echo "##########   source activate volumetric_cpu    #######"
echo "######################################################"
echo "################## !!! ATTENTION !!! #################"
echo "######################################################"
echo ""