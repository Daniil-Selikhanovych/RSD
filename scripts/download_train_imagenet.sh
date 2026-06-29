mkdir -p data/training/imagenet
wget --no-check-certificate https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -O data/training/imagenet/ILSVRC2012_img_train.tar
mkdir -p data/training/imagenet/train && mv data/training/imagenet/ILSVRC2012_img_train.tar data/training/imagenet/train/ && cd data/training/imagenet/train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ../../..