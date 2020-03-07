import os

main_tasks = {
       "laptop": ["targeted_sentiment", "./data/main_task/en/laptop/train.conll", "./data/main_task/en/laptop/dev.conll", "./data/main_task/en/laptop/test.conll", "./resources/embeddings/en/laptop_model.tar.gz"],
       "restaurant": ["targeted_sentiment", "./data/main_task/en/restaurant/train.conll", "./data/main_task/en/restaurant/dev.conll", "./data/main_task/en/restaurant/test.conll", "./resources/embeddings/en/restaurant_model.tar.gz"],
       "mams": ["targeted_sentiment", "./data/main_task/en/MAMS/train.conll", "./data/main_task/en/MAMS/dev.conll", "./data/main_task/en/MAMS/test.conll", "./resources/embeddings/en/restaurant_model.tar.gz"],
       "mpqa": ["mpqa", "./data/main_task/en/mpqa/train.conll", "./data/main_task/en/mpqa/dev.conll", "./data/main_task/en/mpqa/test.conll", "./resources/embeddings/en/transformer-elmo-2019.01.10.tar.gz"]

}

base_stl_config = """
{{
    "dataset_reader": {{
      "type": "{0}",
      "token_indexers": {{
        "elmo": {{
          "type": "elmo_characters",
          "token_min_padding_length": 1
        }}
      }},
      "label_namespace": "sentiment_labels"
    }},
    "train_data_path": "{1}",
    "validation_data_path": "{2}",
    "test_data_path": "{3}",
    "evaluate_on_test": true,
    "model": {{
      "type": "shared_crf_tagger",
      "constrain_crf_decoding": true,
      "calculate_span_f1": true,
      "dropout": 0.5,
      "regularizer": [[".*", {{"type": "l2", "alpha": 0.0001}}]],
      "include_start_end_transitions": false,
      "label_namespace": "sentiment_labels",
      "label_encoding": "BIOUL",
      "skip_connections": true,
      "verbose_metrics": false,
      "text_field_embedder": {{
        "elmo": {{
          "type": "bidirectional_lm_token_embedder",
          "archive_file": "{4}",
          "bos_eos_tokens": ["<S>", "</S>"],
          "remove_bos_eos": true,
          "requires_grad": false
        }}
      }},
      "task_encoder": {{
        "type": "lstm",
        "input_size": 1144,
        "hidden_size": 50,
        "bidirectional": true,
        "num_layers": 1
      }},
      "shared_encoder": {{
        "type": "lstm",
        "input_size": 1024,
        "hidden_size": 60,
        "bidirectional": true,
        "num_layers": 1
      }}
    }},
    "iterator": {{
      "type": "basic",
      "batch_size": 32
    }},
    "trainer": {{
      "optimizer": {{
        "type": "adam",
        "lr": 0.0015
      }},
      "validation_metric": "+f1-measure-overall",
      "num_epochs": 150,
      "num_serialized_models_to_keep": 1,
      "grad_norm": 5.0,
      "patience": 10,
      "cuda_device": 0
    }}
}}
"""

if __name__ == "__main__":


    for main_task in main_tasks.keys():
        outfile = os.path.join("stl", "en", main_task + "_contextualized.jsonnet")
        print(outfile)
        config = main_tasks[main_task]
        to_write = base_stl_config.format(*config)
        with open(outfile, "w") as o:
            o.write(to_write)
