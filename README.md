# ClusterTabNet: Supervised clustering method for table detection and table structure recognition
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/clustertabnet)](https://api.reuse.software/info/github.com/SAP-samples/clustertabnet)

## Description
Implementation of the table detection and table structure recognition deep learning model described in the paper "ClusterTabNet: Supervised clustering method for table detection and table structure recognition" https://arxiv.org/abs/2402.07502

## Requirements
The requirements are detailed in the `requirements.txt` file

## Download and Installation
For sample inference and training, please check out the jupyter notebook: `demo.ipynb`

Download datasets PubTables-1M, pubtabnet, fintabnet, synthtabnet, icdar2019 and format them using notebooks in the `train_data_preparation` folder.

To run the evaluation and further training you can call: <br />
```CUDA_VISIBLE_DEVICES=0 python train/table_extraction.py --output_dir=OUTPUT_DIRECTORY -t=both --ocr_labels_folder=ocr --learning_rate=0.00001 --is_use_4_points --is_use_image_patches --use_dox_datasets --eval_set='test' --checkpoint_path=model_weights/table_recognition.pth```

## Known Issues
No known issues

## How to obtain support
[Create an issue](https://github.com/SAP-samples/<repository-name>/issues) in this repository if you find a bug or have questions about the content.

For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
