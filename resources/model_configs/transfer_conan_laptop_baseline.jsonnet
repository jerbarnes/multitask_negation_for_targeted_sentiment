{
    "target_sentiment_dataset_reader": {
      "type": "targeted_sentiment",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        }
      },
      "label_namespace": "sentiment_labels"
    },
    "target_sentiment_train_data_path": "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data_conll/laptop14_train.txt", 
    "target_sentiment_validation_data_path": "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data_conll/laptop14_dev.txt",
    "target_sentiment_test_data_path": "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data_conll/laptop14_test.txt",
    "negation_dataset_reader": {
      "type": "negation_speculation",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
        }
      },
      "label_namespace": "negation_labels"
    },
    "negation_train_data_path": "./data/auxiliary_tasks/en/conandoyle_train.conllu", 
    "negation_validation_data_path": "./data/auxiliary_tasks/en/conandoyle_dev.conllu",
    "negation_test_data_path": "./data/auxiliary_tasks/en/conandoyle_test.conllu",
    "negation_model": {
      "label_namespace": "negation_labels",
      "label_encoding": "BIO"
    },
    "sentiment_model": {
      "label_namespace": "sentiment_labels",
      "label_encoding": "BIOUL"
    },
    "shared_model":{
      "constrain_crf_decoding": true,
      "calculate_span_f1": true,
      "dropout": 0.5,
      "include_start_end_transitions": false,
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
        "num_layers": 2
      }
    },
    "iterator": {
      "type": "basic",
      "batch_size": 32
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
}