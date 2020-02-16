{
    "task_negation": {
        "dataset_reader": {
            "type": "negation_speculation",
            "token_indexers": {
                "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
                }
            },
            "label_namespace": "negation_labels"
        },
        "train_data_path": "./data/auxiliary_tasks/en/SFU_train.conll", 
        "validation_data_path": "./data/auxiliary_tasks/en/SFU_dev.conll",
        "test_data_path": "./data/auxiliary_tasks/en/SFU_test.conll",
        "model": {
            "type": "shared_crf_tagger",
            "constrain_crf_decoding": true,
            "calculate_span_f1": true,
            "dropout": 0.5,
            "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
            "include_start_end_transitions": false,
            "label_namespace": "negation_labels",
            "label_encoding": "BIO",
            "skip_connections": true,
            "verbose_metrics": false,
            "task_encoder": {
                "type": "lstm",
                "input_size": 500,
                "hidden_size": 50,
                "bidirectional": true,
                "num_layers": 2
            }
        },
        "trainer": {
            "optimizer": {
                "type": "adam"
            },
            "validation_metric": "+f1-measure-overall",
            "num_epochs": 150,
            "grad_norm": 5.0,
            "patience": 10,
            "num_serialized_models_to_keep": 1,
            "cuda_device": 0
        },
        "evaluate": {"cuda_device": 0}
    },
    "task_sentiment": {
        "dataset_reader": {
            "type": "target_conll",
            "token_indexers": {
                "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
                }
            },
            "label_namespace": "sentiment_labels"
        },
        "train_data_path": "./data/main_task/en/MAMS/train.conll", 
        "validation_data_path": "./data/main_task/en/MAMS/dev.conll",
        "test_data_path": "./data/main_task/en/MAMS/test.conll",
        "model": {
            "type": "shared_crf_tagger",
            "constrain_crf_decoding": true,
            "calculate_span_f1": true,
            "dropout": 0.5,
            "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
            "include_start_end_transitions": false,
            "label_namespace": "sentiment_labels",
            "label_encoding": "BIOUL",
            "skip_connections": true,
            "verbose_metrics": false,
            "task_encoder": {
                "type": "lstm",
                "input_size": 500,
                "hidden_size": 50,
                "bidirectional": true,
                "num_layers": 1
            }
        },
        "trainer": {
            "optimizer": {
                "type": "adam",
                "lr": 0.0028
            },
            "validation_metric": "+f1-measure-overall",
            "num_epochs": 150,
            "grad_norm": 5.0,
            "patience": 10,
            "num_serialized_models_to_keep": 1,
            "cuda_device": 0
        },
        "evaluate": {"cuda_device": 0}
    },
    "shared_values": {
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "pretrained_file": "./resources/embeddings/en/glove.840B.300d.txt",
                "embedding_dim": 300,
                "trainable": false
            }
        },
        "shared_encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 100,
            "bidirectional": true,
            "num_layers": 1
        },
        "iterator": {
            "type": "basic",
            "batch_size": 32
        }
    },
    "trainer": {
        "type": "multi_task_trainer",
        "task_order": ["task_negation", "task_sentiment"],
        "main_task": "task_sentiment"
    }
}