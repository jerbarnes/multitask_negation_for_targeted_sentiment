{
    "task_negation": {
        "dataset_reader": {
            "type": "negation_speculation",
            "token_indexers": {
                "elmo": {
                "type": "elmo_characters",
                "token_min_padding_length": 1
                }
            },
            "label_namespace": "negation_labels"
        },
        "train_data_path": "./data/auxiliary_tasks/en/SFU_train.conllu",
        "validation_data_path": "./data/auxiliary_tasks/en/SFU_dev.conllu",
        "test_data_path": "./data/auxiliary_tasks/en/SFU_test.conllu",
        "model": {
            "type": "shared_crf_tagger",
            "constrain_crf_decoding": true,
            "calculate_span_f1": true,
            "dropout": 0.27,
            "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
            "include_start_end_transitions": false,
            "label_namespace": "negation_labels",
            "label_encoding": "BIO",
            "skip_connections": false,
            "verbose_metrics": false
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
    "task_sentiment": {
        "dataset_reader": {
            "type": "targeted_sentiment",
            "token_indexers": {
                "elmo": {
                "type": "elmo_characters",
                "token_min_padding_length": 1
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
                "input_size": 1154,
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
            "elmo": {
            "type": "bidirectional_lm_token_embedder",
            "archive_file": "./resources/embeddings/en/restaurant_model.tar.gz",
            "bos_eos_tokens": ["<S>", "</S>"],
            "remove_bos_eos": true,
            "requires_grad": false
        }
        },
        "shared_encoder": {
            "type": "lstm",
            "input_size": 1024,
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
        "task_order": ["task_negation", "task_sentiment"],
        "main_task": "task_sentiment"
    }
}
