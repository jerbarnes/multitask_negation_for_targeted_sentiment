import os

aux_tasks = {
       "conan_doyle": ["task_negation", "negation_speculation", "", "negation_labels", "./data/auxiliary_tasks/en/conandoyle_train.conllu", "./data/auxiliary_tasks/en/conandoyle_dev.conllu", "./data/auxiliary_tasks/en/conandoyle_test.conllu", "negation_labels"],
       "sfu": ["task_negation", "negation_speculation", "", "negation_labels", "./data/auxiliary_tasks/en/SFU_train.conll", "./data/auxiliary_tasks/en/SFU_dev.conll", "./data/auxiliary_tasks/en/SFU_test.conll", "negation_labels"],
       "sfu_spec": ["task_speculation", "negation_speculation", '''"tag_label": "speculation",''', "speculation_labels", "./data/auxiliary_tasks/en/SFU_train.conll", "./data/auxiliary_tasks/en/SFU_dev.conll", "./data/auxiliary_tasks/en/SFU_test.conll", "speculation_labels"],
       "lextag": ["task_lextag", "streusle", '''"tag_name": "LEXTAG",''', "lextag", "./data/auxiliary_tasks/en/streusle.ud_train.conllulex", "./data/auxiliary_tasks/en/streusle.ud_dev.conllulex", "./data/auxiliary_tasks/en/streusle.ud_test.conllulex", "lextag"],
       "dr": ["task_dr", "streusle", '''"tag_name": "DR",''', "dr", "./data/auxiliary_tasks/en/streusle.ud_train.conllulex", "./data/auxiliary_tasks/en/streusle.ud_dev.conllulex", "./data/auxiliary_tasks/en/streusle.ud_test.conllulex", "dr"],
       "u_pos": ["task_u_pos", "streusle", '''"tag_name": "UPOS",''', "u_pos", "./data/auxiliary_tasks/en/streusle.ud_train.conllulex", "./data/auxiliary_tasks/en/streusle.ud_dev.conllulex", "./data/auxiliary_tasks/en/streusle.ud_test.conllulex", "u_pos"]
}

main_tasks = {
       "laptop": ["targeted_sentiment", "./data/main_task/en/laptop/train.conll", "./data/main_task/en/laptop/dev.conll", "./data/main_task/en/laptop/test.conll", "./resources/embeddings/en/laptop_model.tar.gz"],
       "restaurant": ["targeted_sentiment", "./data/main_task/en/restaurant/train.conll", "./data/main_task/en/restaurant/dev.conll", "./data/main_task/en/restaurant/test.conll", "./resources/embeddings/en/restaurant_model.tar.gz"],
       "mams": ["targeted_sentiment", "./data/main_task/en/MAMS/train.conll", "./data/main_task/en/MAMS/dev.conll", "./data/main_task/en/MAMS/test.conll", "./resources/embeddings/en/restaurant_model.tar.gz"],
       "mpqa": ["mpqa", "./data/main_task/en/mpqa/train.conll", "./data/main_task/en/mpqa/dev.conll", "./data/main_task/en/mpqa/test.conll", "./resources/embeddings/en/transformer-elmo-2019.01.10.tar.gz"]

}

base_multitask_config = """
{{
    "{0}": {{
        "dataset_reader": {{
            "type": "{1}",
            "token_indexers": {{
                "elmo": {{
                "type": "elmo_characters",
                "token_min_padding_length": 1
                }}
            }},
            {2}
            "label_namespace": "{3}"
        }},
        "train_data_path": "{4}",
        "validation_data_path": "{5}",
        "test_data_path": "{6}",
        "model": {{
            "type": "shared_crf_tagger",
            "constrain_crf_decoding": true,
            "calculate_span_f1": true,
            "dropout": 0.27,
            "regularizer": [[".*", {{"type": "l2", "alpha": 0.0001}}]],
            "include_start_end_transitions": false,
            "label_namespace": "{7}",
            "label_encoding": "BIO",
            "skip_connections": false,
            "verbose_metrics": false
        }},
        "trainer": {{
            "optimizer": {{
                "type": "adam",
                "lr": 0.0019
            }},
            "validation_metric": "+f1-measure-overall",
            "num_epochs": 150,
            "grad_norm": 5.0,
            "patience": 10,
            "num_serialized_models_to_keep": 1,
            "cuda_device": 0
        }},
        "evaluate": {{"cuda_device": 0}}
    }},
    "task_sentiment": {{
        "dataset_reader": {{
            "type": "{8}",
            "token_indexers": {{
                "elmo": {{
                "type": "elmo_characters",
                "token_min_padding_length": 1
                }}
            }},
            "label_namespace": "sentiment_labels"
        }},
        "train_data_path": "{9}",
        "validation_data_path": "{10}",
        "test_data_path": "{11}",
        "model": {{
            "type": "shared_crf_tagger",
            "constrain_crf_decoding": true,
            "calculate_span_f1": true,
            "dropout": 0.27,
            "regularizer": [[".*", {{"type": "l2", "alpha": 0.0001}}]],
            "include_start_end_transitions": false,
            "label_namespace": "sentiment_labels",
            "label_encoding": "BIOUL",
            "skip_connections": true,
            "verbose_metrics": false,
            "task_encoder": {{
                "type": "lstm",
                "input_size": 1154,
                "hidden_size": 50,
                "bidirectional": true,
                "num_layers": 1
            }}
        }},
        "trainer": {{
            "optimizer": {{
                "type": "adam",
                "lr": 0.0019
            }},
            "validation_metric": "+f1-measure-overall",
            "num_epochs": 150,
            "grad_norm": 5.0,
            "patience": 10,
            "num_serialized_models_to_keep": 1,
            "cuda_device": 0
        }},
        "evaluate": {{"cuda_device": 0}}
    }},
    "shared_values": {{
        "text_field_embedder": {{
            "elmo": {{
            "type": "bidirectional_lm_token_embedder",
            "archive_file": "{12}",
            "bos_eos_tokens": ["<S>", "</S>"],
            "remove_bos_eos": true,
            "requires_grad": false
        }}
        }},
        "shared_encoder": {{
            "type": "lstm",
            "input_size": 1024,
            "hidden_size": 65,
            "bidirectional": true,
            "num_layers": 1
        }},
        "iterator": {{
            "type": "basic",
            "batch_size": 32
        }}
    }},
    "trainer": {{
        "type": "multi_task_trainer",
        "task_order": ["{0}", "task_sentiment"],
        "main_task": "task_sentiment"
    }}
}}
"""

if __name__ == "__main__":

    for aux_task in aux_tasks.keys():
        for main_task in main_tasks.keys():
            print("{0}-{1}".format(aux_task, main_task))
            outfile = os.path.join("mtl", "en", aux_task, main_task + "_contextualized.jsonnet")
            print(outfile)
            config = aux_tasks[aux_task] + main_tasks[main_task]
            to_write = base_multitask_config.format(*config)
            with open(outfile, "w") as o:
                o.write(to_write)
