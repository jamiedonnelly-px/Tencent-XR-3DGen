import csv
import json
import os
from pathlib import Path

import datasets

DATA_TYPES = ['text', 'image', 'conditioning_image']

_DESCRIPTION = "This dataset consists of uv datas, normal as condition."
_HOMEPAGE = "tencent"
_LICENSE = ""

class UVDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.0.1")

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        feas = dict()
        for d_type in DATA_TYPES:
            feas[d_type] = datasets.Value("string")
        features = datasets.Features(feas)
        dataset = datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )
        return dataset

    def _split_generators(self, dl_manager):
        dataset_root = self.config.data_dir
        generators = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset_root": dataset_root,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dataset_root": dataset_root,
                    "split": "test",
                },
            ),
        ]
        return generators

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, dataset_root, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        input_json = os.path.join(dataset_root, f"{split}.json")
        assert os.path.exists(input_json), input_json

        key = 0
        with open(input_json, encoding='utf-8') as f:
            metas = json.load(f)
            for meta in metas:
                key += 1
                result = dict()

                for d_type in DATA_TYPES:
                    result[d_type] = meta.get(d_type, "")

                if len(result['text'])==0:
                    continue
                if split =='train' and ( len(result['image'])==0 or (not os.path.exists(result['image'])) ):
                        continue

                yield key, result
        pass
