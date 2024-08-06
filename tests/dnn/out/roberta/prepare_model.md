# download models
wget https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/roberta/model/roberta-base-11.tar.gz

tar -xvf roberta-base-11.tar.gz

# move the onnx model to the first directory

# get relay model
python run onnx_to_relay.py
