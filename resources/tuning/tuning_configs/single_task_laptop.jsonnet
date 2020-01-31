local SEED = std.parseInt(std.extVar("SEED"));
local LEARNING_RATE = std.extVar("LEARNING_RATE");
local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));
local SHARED_HIDDEN_SIZE = std.parseInt(std.extVar("SHARED_HIDDEN_SIZE"));
# Multiple by 2 because of bi-directional LSTM
local TASK_ENCODER_INPUT_SIZE = SHARED_HIDDEN_SIZE * 2 + 300;
local SENTIMENT_DROPOUT = std.extVar("SENTIMENT_DROPOUT");

{
    "numpy_seed": SEED,
    "pytorch_seed": SEED,
    "random_seed": SEED,
    "dataset_reader": {
        "type": "targeted_sentiment",
        "token_indexers": {
            "tokens": {
            "type": "single_id",
            "lowercase_tokens": true
            }
        },
        "label_namespace": "sentiment_labels"
    },
    "train_data_path": "/home/andrew/Desktop/multitask_negation_for_targeted_sentiment/data/main_task/en/laptop/train.conll", 
    "validation_data_path": "/home/andrew/Desktop/multitask_negation_for_targeted_sentiment/data/main_task/en/laptop/dev.conll",
    "test_data_path": "/home/andrew/Desktop/multitask_negation_for_targeted_sentiment/data/main_task/en/laptop/test.conll",
    "evaluate_on_test": true,
    "model": {
        "type": "shared_crf_tagger",
        "constrain_crf_decoding": true,
        "calculate_span_f1": true,
        "dropout": SENTIMENT_DROPOUT,
        "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
        "include_start_end_transitions": false,
        "label_namespace": "sentiment_labels",
        "label_encoding": "BIOUL",
        "skip_connections": true,
        "verbose_metrics": true,
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 300,
                "pretrained_file": "/home/andrew/Desktop/multitask_negation_for_targeted_sentiment/resources/embeddings/en/glove.840B.300d.txt",
                "trainable": false
            }
        },
        "task_encoder": {
            "type": "lstm",
            "input_size": TASK_ENCODER_INPUT_SIZE,
            "hidden_size": 50,
            "bidirectional": true,
            "num_layers": 1
        },
        "shared_encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": SHARED_HIDDEN_SIZE,
            "bidirectional": true,
            "num_layers": 1
        }
    },
    "iterator": {
            "type": "basic",
            "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "lr": LEARNING_RATE,
            "type": "adam"
        },
        "validation_metric": "+f1-measure-overall",
        "num_epochs": 150,
        "grad_norm": 5.0,
        "patience": 10,
        "num_serialized_models_to_keep": 1,
        "cuda_device": CUDA_DEVICE
    }
}