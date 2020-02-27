{
    "dataset_reader": {
      "type": "streusle",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        }
      },
      "tag_name": "UPOS",
      "label_namespace": "u_pos"
    },
    "train_data_path": "./data/auxiliary_tasks/en/streusle.ud_train.conllulex", 
    "validation_data_path": "./data/auxiliary_tasks/en/streusle.ud_dev.conllulex",
    "test_data_path": "./data/auxiliary_tasks/en/streusle.ud_test.conllulex",
    "evaluate_on_test": true,
    "model": {
      "type": "shared_crf_tagger",
      "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
      "label_namespace": "u_pos",
      "crf": true,
      "include_start_end_transitions": false,
      "constrain_crf_decoding": false,
      "verbose_metrics": false,
      "calculate_span_f1": false,
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
      "validation_metric": "+accuracy",
      "num_epochs": 150,
      "num_serialized_models_to_keep": 1,
      "grad_norm": 5.0,
      "patience": 10,
      "cuda_device": 0
    }
}