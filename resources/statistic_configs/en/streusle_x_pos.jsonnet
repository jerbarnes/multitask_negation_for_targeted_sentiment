{
    "dataset_reader": {
      "type": "streusle",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        }
      },
      "tag_name": "XPOS",
      "label_namespace": "x_pos"
    },
    "train_data_path": "./data/auxiliary_tasks/en/streusle.ud_train.conllulex", 
    "validation_data_path": "./data/auxiliary_tasks/en/streusle.ud_dev.conllulex",
    "test_data_path": "./data/auxiliary_tasks/en/streusle.ud_test.conllulex",
    "evaluate_on_test": true,
    "model": {
      "type": "simple_tagger",
      "regularizer": [[".*", {"type": "l2", "alpha": 0.0001}]],
      "label_namespace": "x_pos",
      "verbose_metrics": false,
      "text_field_embedder": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "trainable": true
        }
      },
      "encoder": {
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
      "cuda_device": -1
    }
}