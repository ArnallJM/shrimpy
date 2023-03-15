git clone https://github.com/MIC-DKFZ/nnUNet.git
git clone https://github.com/ArnallJM/shrimpy
(
  cd shrimpy || exit
  git pull
  git checkout dev
)
cp shrimpy/local_artemis_setup.sh local_artemis_setup.sh