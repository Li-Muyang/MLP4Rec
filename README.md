## MLP4Rec: A Pure MLP Architecture for Sequential Recommendations (IJCAI 2022, Long Oral)

This repo contains a copy of RecBole 1.0.1, with our implementation of MLP4Rec

Current release is our prelimnary version, which suffers from some efficiency issues, we are currently collaborating with RecBole to solve those issues

## Usage

Before running MLP4Rec, please make sure you visit https://recbole.io/ and https://github.com/RUCAIBox/RecBole first to make sure that you have fulfilled the pre-requistes and familar with basic RecBole usage.

Then, you can simply running run_mlp4rec.py from root directory, default dataset is ml-100k, experiment configurations is by default sync with ours https://www.ijcai.org/proceedings/2022/297.

## Report a problem

If you encounter any problem, please contact Muyang Li by muli0371@uni.sydney.edu.au

## Ciation

If you feel our work is insightful, please consider citing us 
```
@article{li2022mlp4rec,
  title={MLP4Rec: A Pure MLP Architecture for Sequential Recommendations},
  author={Li, Muyang and Zhao, Xiangyu and Lyu, Chuan and Zhao, Minghao and Wu, Runze and Guo, Ruocheng},
  journal={arXiv preprint arXiv:2204.11510},
  year={2022}
}
```
If your code involves any content from this repo, please also knidly cite RecBole
```
@inproceedings{recbole1.0,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Zhao, Wayne Xin and Mu, Shanlei and Hou, Yupeng and Lin, Zihan and Chen, Yushuo and Pan, Xingyu and Li, Kaiyuan and Lu, Yujie and Wang, Hui and Tian, Changxin and others},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  year={2021}
}
```
```
@article{recbole2.0,
  title={RecBole 2.0: Towards a More Up-to-Date Recommendation Library},
  author={Zhao, Wayne Xin and Hou, Yupeng and Pan, Xingyu and Yang, Chen and Zhang, Zeyu and Lin, Zihan and Zhang, Jingsen and Bian, Shuqing and Tang, Jiakai and Sun, Wenqi and others},
  journal={arXiv preprint arXiv:2206.07351},
  year={2022}
}
```
