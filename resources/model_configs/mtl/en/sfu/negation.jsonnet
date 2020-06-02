{
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
    "evaluate_on_test": true,
    "model": {
      "type": "shared_crf_tagger",
      "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
      "label_namespace": "negation_labels",
      "crf": true,
      "label_encoding": "BIO",
      "include_start_end_transitions": false,
      "verbose_metrics": false,
      "dropout": 0.5,
      "text_field_embedder": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "./resources/embeddings/en/glove.840B.300d.txt",
          "embedding_dim": 300,
          "trainable": false
        }
      },
      "task_encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 50,
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
        "type": "adam",
        "lr": 0.001
      },
      "validation_metric": "+f1-measure-overall",
      "num_epochs": 150,
      "num_serialized_models_to_keep": 1,
      "grad_norm": 5.0,
      "patience": 10,
      "cuda_device": 0
    }
}