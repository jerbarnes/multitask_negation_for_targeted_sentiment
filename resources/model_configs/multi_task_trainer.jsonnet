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
        "train_data_path": "./data/auxiliary_tasks/en/conandoyle_train.conllu", 
        "validation_data_path": "./data/auxiliary_tasks/en/conandoyle_dev.conllu",
        "test_data_path": "./data/auxiliary_tasks/en/conandoyle_test.conllu",
        "model": {
            "type": "shared_crf",
            "constrain_crf_decoding": true,
            "calculate_span_f1": true,
            "dropout": 0.5,
            "include_start_end_transitions": false,
            "label_namespace": "negation_labels",
            "label_encoding": "BIO",
            "skip_connections": true,
            "verbose_metrics": false,
            "task_encoder": {
                "type": "lstm",
                "input_size": 400,
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
            "cuda_device": 0
        },
        "evaluate": {"cuda_device": 0}
    },
    "task_sentiment": {
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
        "train_data_path": "./data/main_task/en/laptop/train.conll", 
        "validation_data_path": "./data/main_task/en/laptop/dev.conll",
        "test_data_path": "./data/main_task/en/laptop/test.conll",
        "model": {
            "type": "shared_crf",
            "constrain_crf_decoding": true,
            "calculate_span_f1": true,
            "dropout": 0.5,
            "include_start_end_transitions": false,
            "label_namespace": "sentiment_labels",
            "label_encoding": "BIOUL",
            "skip_connections": true,
            "verbose_metrics": false,
            "task_encoder": {
                "type": "lstm",
                "input_size": 400,
                "hidden_size": 50,
                "bidirectional": true,
                "num_layers": 1
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
            "hidden_size": 50,
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