# RMem: Restricted Memory Banks Improve Video Object Segmentation

Junbao Zhou, [Ziqi Pang](https://ziqipang.github.io/), [Yu-Xiong Wang](https://yxw.web.illinois.edu/)

University of Illinois Urbana-Champaign

## Abstract

With recent video object segmentation (VOS) benchmarks evolving to challenging scenarios, we revisit a simple but overlooked strategy: restricting the size of memory banks. This diverges from the prevalent practice of expanding memory banks to accommodate extensive historical information. Our specially designed "memory deciphering" study offers a pivotal insight underpinning such a strategy: expanding memory banks, while seemingly beneficial, actually increases the difficulty for VOS modules to decode relevant features due to the confusion from redundant information. By restricting memory banks to a limited number of essential frames, we achieve a notable improvement in VOS accuracy. This process balances the importance and freshness of frames to maintain an informative memory bank within a bounded capacity. Additionally, restricted memory banks reduce the training-inference discrepancy in memory lengths compared with continuous expansion. This fosters new opportunities in temporal reasoning and enables us to introduce the previously overlooked "temporal positional embedding." Finally, our insights are embodied in "RMem" ("R" for restricted), a simple yet effective VOS modification that excels at challenging VOS scenarios and establishes new state of the art for object state changes (VOST dataset) and long videos (the Long Videos dataset).

## Method Overview

![method](figures/method2.jpg)

## Data preparation

Download the VOST dataset from [vostdataset.org](https://www.vostdataset.org/) , and organize the directory structure as follows:

```bash
├── aot_plus
│   ├── configs
│   ├── dataloaders
│   ├── datasets
│   │   └── VOST
│   │       ├── Annotations
│   │       ├── ImageSets
│   │       ├── JPEGImages
│   │       └── JPEGImages_10fps
│   ├── docker
│   ├── networks
│   ├── pretrain_models
│   └── tools
├── evaluation
└── README.md
```

> hint: you can achieve it by soft link:
> ```bash
> ln -s <your VOST directory>  ./datasets/VOST
> ```

## Checkpoint

## Evaluation