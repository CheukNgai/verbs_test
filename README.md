# BERT DISTRIBUTE VERSION

## Download the dataset and BASE BERT model
*   [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)**:
12-layer, 768-hidden, 12-heads, 110M parameters
*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)



## Set up the env var
Note: The code is in ./bert_clustar/bert
To run this code, you should set up such env variables firstly as followed:

```
export BERT_BASE_DIR=/home/ubuntu/bert_clustar/uncased_L-12_H-768_A-12
export SQUAD_DIR=/home/ubuntu/bert_clustar
export RDMA_DEVICE=mlx5_0
export TF_CPU_ALLOCATOR_USE_BFC=1
```

For node1.py(parameter server process), use the CPU memory:
```
export CUDA_VISIBLE_DEVICES=
```

For node2.py(worker process), use the GPU memory:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Run the code with grpc

note: when using Tesla V100, it's suggested to use the batch_size 32.

1.modify the (1)ip:port (2)rpc_layer (3)run_config protocol (4)task type in node1.py(ps) and node2.py(worker)


```
os.environ['TF_CONFIG'] = '{\
"cluster": { \
"worker": ["10.0.24.3:9100"], \
"ps": ["10.0.24.3:9108"]},\
"task": {"type": 
"ps", "index": 0},\
"rpc_layer": "grpc"\
}'
```

and

```
config = tf.estimator.RunConfig(train_distribute=distribute, save_checkpoints_steps=None, protocol='grpc')
```

THEN run this command:

In ps process, run:
```
python node1.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_train=True   --train_file=$SQUAD_DIR/train-v1.1.json   --do_predict=True   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=32   --learning_rate=3e-5   --num_train_epochs=2.0   --max_seq_length=384   --doc_stride=128   --output_dir=/tmp/squad_base/
```

In worker process, run:
```
python node2.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_train=True   --train_file=$SQUAD_DIR/train-v1.1.json   --do_predict=True   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=32   --learning_rate=3e-5   --num_train_epochs=2.0   --max_seq_length=384   --doc_stride=128   --output_dir=/tmp/squad_base/
```

## Run the code with grpc+verbs(GetCPUAllocator ERROR)

note: when using Tesla V100, it's suggested to use the batch_size 32.

1.modify the (1)ip:port (2)rpc_layer (3)run_config protocol (4)task type in node1.py(ps) and node2.py(worker)


```
os.environ['TF_CONFIG'] = '{\
"cluster": { \
"worker": ["10.0.24.3:9100"], \
"ps": ["10.0.24.3:9108"]},\
"task": {"type": 
"ps", "index": 0},\
"rpc_layer": "grpc+verbs"\
}'
```

and

```
config = tf.estimator.RunConfig(train_distribute=distribute, save_checkpoints_steps=None, protocol='grpc+verbs')
```

THEN run this command:

In ps process, run:
```
python node1.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_train=True   --train_file=$SQUAD_DIR/train-v1.1.json   --do_predict=True   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=32   --learning_rate=3e-5   --num_train_epochs=2.0   --max_seq_length=384   --doc_stride=128   --output_dir=/tmp/squad_base/
```

In worker process, run:
```
python node2.py --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_train=True   --train_file=$SQUAD_DIR/train-v1.1.json   --do_predict=True   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=32   --learning_rate=3e-5   --num_train_epochs=2.0   --max_seq_length=384   --doc_stride=128   --output_dir=/tmp/squad_base/
```

