seed1 = """
#[version = "0.0.5"]
def @main(%x: Tensor[(3, 4, 4), float32], %v: Tensor[(3, 4, 4), float32]) {
  %0 = abs(%v);
  %1 = sqrt(%0);
  divide(%x, %1)
}
"""
seed2 = """
#[version = "0.0.5"]
def @main(%x: Tensor[(3, 5), float32], %weight: Tensor[(4, 5), float32], %in_bias: Tensor[(5), float32]) {
  %0 = multiply(%x, meta[relay.Constant][0]);
  %1 = nn.relu(%0);
  %2 = add(%1, %in_bias);
  nn.dense(%2, %weight, units=None)
}

#[metadata]
{
  "root": 1,
  "nodes": [
    {
      "type_key": ""
    },
    {
      "type_key": "Map",
      "keys": [
        "relay.Constant"
      ],
      "data": [2]
    },
    {
      "type_key": "Array",
      "data": [3]
    },
    {
      "type_key": "relay.Constant",
      "attrs": {
        "_checked_type_": "0",
        "data": "0",
        "span": "0",
        "virtual_device_": "4"
      }
    },
    {
      "type_key": "VirtualDevice",
      "attrs": {
        "device_type_int": "-1",
        "memory_scope": "5",
        "target": "0",
        "virtual_device_id": "-1"
      }
    },
    {
      "type_key": "runtime.String"
    }
  ],
  "b64ndarrays": [
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAIgAQAFAAAAAAAAABQAAAAAAAAAL98OP1M3ZD9CSQk/IrUuP216WT8="
  ],
  "attrs": {"tvm_version": "0.12.dev0"}
}

"""
seed3 = """
#[version = "0.0.5"]
def @main(%x: Tensor[(2, 4, 10, 10), float32], %weight, %out_bias: Tensor[(4), float32]) {
  %0 = nn.conv2d(%x, %weight, padding=[1, 1, 1, 1], channels=4, kernel_size=[3, 3]);
  %1 = nn.bias_add(%0, %out_bias);
  %2 = nn.relu(%1);
  multiply(%2, meta[relay.Constant][0])
}

#[metadata]
{
  "root": 1,
  "nodes": [
    {
      "type_key": ""
    },
    {
      "type_key": "Map",
      "keys": [
        "relay.Constant"
      ],
      "data": [2]
    },
    {
      "type_key": "Array",
      "data": [3]
    },
    {
      "type_key": "relay.Constant",
      "attrs": {
        "_checked_type_": "0",
        "data": "0",
        "span": "0",
        "virtual_device_": "4"
      }
    },
    {
      "type_key": "VirtualDevice",
      "attrs": {
        "device_type_int": "-1",
        "memory_scope": "5",
        "target": "0",
        "virtual_device_id": "-1"
      }
    },
    {
      "type_key": "runtime.String"
    }
  ],
  "b64ndarrays": [
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAwAAAAIgAQAEAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAAQAAAAAAAAAOdeCD9FsjQ/c7sCP60pDj8="
  ],
  "attrs": {"tvm_version": "0.12.dev0"}
}
"""
seed4 = """
#[version = "0.0.5"]
def @main(%x: Tensor[(32), float16]) {
  %0 = add(%x, meta[relay.Constant][0]);
  %1 = multiply(%0, meta[relay.Constant][1]);
  add(%1, meta[relay.Constant][2])
}

#[metadata]
{
  "root": 1,
  "nodes": [
    {
      "type_key": ""
    },
    {
      "type_key": "Map",
      "keys": [
        "relay.Constant"
      ],
      "data": [2]
    },
    {
      "type_key": "Array",
      "data": [3, 6, 7]
    },
    {
      "type_key": "relay.Constant",
      "attrs": {
        "_checked_type_": "0",
        "data": "0",
        "span": "0",
        "virtual_device_": "4"
      }
    },
    {
      "type_key": "VirtualDevice",
      "attrs": {
        "device_type_int": "-1",
        "memory_scope": "5",
        "target": "0",
        "virtual_device_id": "-1"
      }
    },
    {
      "type_key": "runtime.String"
    },
    {
      "type_key": "relay.Constant",
      "attrs": {
        "_checked_type_": "0",
        "data": "1",
        "span": "0",
        "virtual_device_": "4"
      }
    },
    {
      "type_key": "relay.Constant",
      "attrs": {
        "_checked_type_": "0",
        "data": "2",
        "span": "0",
        "virtual_device_": "4"
      }
    }
  ],
  "b64ndarrays": [
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAIQAQAgAAAAAAAAAEAAAAAAAAAAcztHOS87JTbUK/o5ozX/Ojs4izknOiI5SjuWN4w2zjo9Ouolvjk6NSg1/zl1Mv85NDuNNZc7VTbjNg804jjOMg==",
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAIQAQAgAAAAAAAAAEAAAAAAAAAANjMpND0tEDbMN0Y5vzNhNa44Qjv/Nu86NTnYMx8wzicyMtwooDjxNTM6ejjVDyc6tzXEIB0h7zR5Otg7+jiNNw==",
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAIQAQAgAAAAAAAAAEAAAAAAAAAAXTjGOhY4pjOaO3A1/zheMls4sTSYOeE51zjuOZs0YiPkNzA7FzrwKNY5OyjUOqg00zgbM8g6hzDtNfk2UjmjOA=="
  ],
  "attrs": {"tvm_version": "0.12.dev0"}
}
"""
seed5 = """
#[version = "0.0.5"]
def @main(%x: Tensor[(32), float32]) {
  %0 = add(meta[relay.Constant][0], %x);
  add(%0, meta[relay.Constant][1])
}

#[metadata]
{
  "root": 1,
  "nodes": [
    {
      "type_key": ""
    },
    {
      "type_key": "Map",
      "keys": [
        "relay.Constant"
      ],
      "data": [2]
    },
    {
      "type_key": "Array",
      "data": [3, 6]
    },
    {
      "type_key": "relay.Constant",
      "attrs": {
        "_checked_type_": "0",
        "data": "0",
        "span": "0",
        "virtual_device_": "4"
      }
    },
    {
      "type_key": "VirtualDevice",
      "attrs": {
        "device_type_int": "-1",
        "memory_scope": "5",
        "target": "0",
        "virtual_device_id": "-1"
      }
    },
    {
      "type_key": "runtime.String"
    },
    {
      "type_key": "relay.Constant",
      "attrs": {
        "_checked_type_": "0",
        "data": "1",
        "span": "0",
        "virtual_device_": "4"
      }
    }
  ],
  "b64ndarrays": [
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAIgAQAgAAAAAAAAAIAAAAAAAAAA7UEtPzWOMD/zwGw+U2k9PtfQYT88TgQ/RIGNPa4bgT6uMnU/KCwMPn9SrT2jsEI/oUFzP4wLNT+rjXM/ZLGWPihFHz/IDxI/Ed9bPtvKGj6iuVg/Ve1jPy4xMD9spMQ9PZQrP3FtKT/t42Y/K8BaOiO31j0H5wA+uTT/PoDvZz4=",
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAIgAQAgAAAAAAAAAIAAAAAAAAAAgrKWPoljGj8wflI+XMijPTD9ST90kR0/IAcKP0imcT4La+4+Q4UIPx7CNT+bwSU/cKxRP9oBaz91UR0/Gng2P7YNFj/OZog+gjsNPtbNzj7y0ls+YUhlP2vF1j1/HiA/fU3TPWQYxj7auxA+K3cIP6pzpj5KrDE/G+0XP2tNyj4="
  ],
  "attrs": {"tvm_version": "0.12.dev0"}
}
"""

seed6="""
#[version = "0.0.5"]
def @main(%x: Tensor[(2, 4, 10, 10, 10), float32], %weight, %out_bias: Tensor[(8), float32]) {
  %0 = nn.conv3d(%x, %weight, padding=[1, 1, 1, 1, 1, 1], channels=8, kernel_size=[3, 3, 3]);
  %1 = expand_dims(%out_bias, axis=1, num_newaxis=3);
  %2 = add(%0, %1);
  %3 = nn.relu(%2);
  multiply(%3, meta[relay.Constant][0])
}

#[metadata]
{
  "root": 1,
  "nodes": [
    {
      "type_key": ""
    },
    {
      "type_key": "Map",
      "keys": [
        "relay.Constant"
      ],
      "data": [2]
    },
    {
      "type_key": "Array",
      "data": [3]
    },
    {
      "type_key": "relay.Constant",
      "attrs": {
        "_checked_type_": "0",
        "data": "0",
        "span": "0",
        "virtual_device_": "4"
      }
    },
    {
      "type_key": "VirtualDevice",
      "attrs": {
        "device_type_int": "-1",
        "memory_scope": "5",
        "target": "0",
        "virtual_device_id": "-1"
      }
    },
    {
      "type_key": "runtime.String"
    }
  ],
  "b64ndarrays": [
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAABAAAAAIgAQAIAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAACAAAAAAAAAAEpcrPy7Yej8nUC8/VL5VPzUEIT8cQXk/tRwrPxnlYz8="
  ],
  "attrs": {"tvm_version": "0.12.dev0"}
}
"""
