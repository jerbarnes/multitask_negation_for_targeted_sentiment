{
    "task_lextag": {
        "dataset_reader": {
            "type": "streusle",
            "token_indexers": {
                "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
                }
            },
            "tag_name": "LEXTAG",
            "label_namespace": "lextag"
        },
        "train_data_path": "./data/auxiliary_tasks/en/streusle.ud_train.conllulex", 
        "validation_data_path": "./data/auxiliary_tasks/en/streusle.ud_dev.conllulex",
        "test_data_path": "./data/auxiliary_tasks/en/streusle.ud_test.conllulex",
        "model": {
            "type": "shared_crf_tagger",
            "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
            "label_namespace": "lextag",
            "crf": true,
            "include_start_end_transitions": false,
            "constrain_crf_decoding": false,
            "verbose_metrics": false,
            "calculate_span_f1": false,
            "dropout": 0.27,
            "skip_connections": false
        },
        "trainer": {
            "optimizer": {
                "type": "adam",
                "lr": 0.0019
            },
            "validation_metric": "+accuracy",
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
            "dropout": 0.27,
            "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
            "include_start_end_transitions": false,
            "label_namespace": "sentiment_labels",
            "label_encoding": "BIOUL",
            "skip_connections": true,
            "verbose_metrics": false,
            "task_encoder": {
                "type": "lstm",
                "input_size": 430,
                "hidden_size": 50,
                "bidirectional": true,
                "num_layers": 1
            }
        },
        "trainer": {
            "optimizer": {
                "type": "adam",
                "lr": 0.0019
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
            "hidden_size": 65,
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
        "task_order": ["task_lextag", "task_sentiment"],
        "main_task": "task_sentiment"
    }
}