{
  "dataset_reader": {
    "type": "negation_speculation",
    "label_namespace": "negation_labels",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 1
      }
    }
  },
  "train_data_path": "multitask_negation_target/tests/fixtures/allen/dataset_readers/negation_speculation/conan_doyle_data.conllu",
  "validation_data_path": "multitask_negation_target/tests/fixtures/allen/dataset_readers/negation_speculation/conan_doyle_data.conllu",
  "model": {
    "type": "shared_crf_tagger",
    "skip_connections": true,
    "label_namespace": "negation_labels",
    "crf": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 50
        },
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 25
          },
          "encoder": {
            "type": "gru",
            "input_size": 25,
            "hidden_size": 80,
            "num_layers": 2,
            "dropout": 0.0,
            "bidirectional": true
          }
        }
      }
    },
    "shared_encoder": {
      "type": "gru",
      "input_size": 210,
      "hidden_size": 300,
      "num_layers": 2,
      "dropout": 0.0,
      "bidirectional": true
    },
    "task_encoder": {
      "type": "gru",
      "input_size": 810,
      "hidden_size": 300,
      "num_layers": 2,
      "dropout": 0.0,
      "bidirectional": true
    }
  },
  "iterator": {"type": "basic", "batch_size": 32},
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 5,
    "cuda_device": -1
  }
}