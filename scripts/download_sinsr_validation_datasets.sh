mkdir -p data/validation/imagenet256
gdown --no-check-certificate 1NhmpON2dB2LjManfX6uIj8Pj_Jx6N-6l -O data/validation/imagenet256_srx4.zip
unzip data/validation/imagenet256_srx4.zip -d data/validation
gdown --no-check-certificate https://drive.google.com/uc?id=17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM -O "data/validation/RealSR(V3).tar.gz"
tar -xzvf "data/validation/RealSR(V3).tar.gz" -C "data/validation"
mv "data/validation/RealSR(V3)" "data/validation/RealSR_V3_all_images"

path_to_RealSR="data/validation/RealSR_V3_all_images"
path_to_RealSR_reordered="data/validation/RealSR_V3"
factor=4
python scripts/prepare_testing_realsr.py --path_to_realsr ${path_to_RealSR} --path_to_realsr_reordered ${path_to_RealSR_reordered} \
--factor ${factor}
git clone https://github.com/zsyOAOA/ResShift
mv ResShift/testdata/RealSet65 data/validation/RealSet65
rm -rf ResShift