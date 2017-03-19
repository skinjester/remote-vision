---
name: GoogLeNet_cars

caffemodel: googlenet_finetune_web_car_iter_10000.caffemodel

caffemodel_url: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/googlenet_finetune_web_car_iter_10000.caffemodel

license: non-commercial

caffe_commit: 737ea5e936821b5c69f9c3952d72693ae5843370

gist_id: b90eb88e31cd745525ae

---

##Description

This is the GoogLeNet model pre-trained on ImageNet classification task and fine-tuned on 431 car models in [CompCars dataset](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html). It is described in the [technical report](http://arxiv.org/abs/1506.08959). Please cite the following work if the model is useful for you.

    A Large-Scale Car Dataset for Fine-Grained Categorization and Verification
    L. Yang, P. Luo, C. C. Loy, X. Tang, arXiv:1506.08959, 2015

The bundled model is the iteration 10,000 snapshot.
This model obtains a top-1 accuracy 91.2% and a top-5 accuracy 98.1% on the testing set, using only the center crop.

## License

The data used to train this model comes from the ImageNet project and the CompCars dataset, which distribute their databases to researchers who agree to a following term of access:
"Researcher shall use the Database only for non-commercial research and educational purposes."
Accordingly, this model is distributed under a non-commercial license.