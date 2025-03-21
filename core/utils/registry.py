"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# Borrowed from https://github.com/facebookresearch/pythia/blob/master/pythia/common/registry.py.

class Registry:
    r"""Registry is a class that stores class types across the entire dataset.
    It makes coding very convenient and flexible. Should not be manually
    instantiated.
    """
    mapping = {
        "dataset_name_mapping": {},
        "model_name_mapping": {},
        "trainer_name_mapping": {},
        "loss_name_mapping": {}
    }
    loaded = False

    @classmethod
    def register_dataset(cls, name):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the dataset will be registered.

        Usage:

            from core.utils.registry import registry

            @registry.register_dataset("opfdata")
            class OPFData:
                ...
        """

        def wrap(func):
            cls.mapping["dataset_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the model will be registered.

        Usage:

            from core.utils.registry import registry

            @registry.register_model("powerflownet")
            class PowerFlowNet:
                ...
        """

        def wrap(func):
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_trainer(cls, name):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the trainer will be registered.

        Usage:

            from core.utils.registry import registry

            @registry.register_trainer("base_trainer")
            class BaseTrainer:
                ...
        """

        def wrap(func):
            cls.mapping["trainer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_loss(cls, name):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the loss will be registered.

        Usage:

            from core.utils.registry import registry

            @registry.register_loss("obj_n_penalty")
            class Objective_n_Penalty:
                ...
        """

        def wrap(func):
            cls.mapping["loss_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def get_dataset_class(cls, name):
        r"""Obtain registry dataset class with key 'name'

        Args:
            name: Key with which the dataset is registered.

        Usage:

            from core.utils.registry import registry

            if __name__ == "__main__":
                ...
                OPFData = registry.get_dataset_class("opfdata")
                ...
        """

        return cls.mapping["dataset_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        r"""Obtain registry model class with key 'name'

        Args:
            name: Key with which the model is registered.

        Usage:

            from core.utils.registry import registry

            if __name__ == "__main__":
                ...
                PowerFlowNet = registry.get_model_class("powerflownet")
                ...
        """

        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_trainer_class(cls, name):
        r"""Obtain registry trainer class with key 'name'

        Args:
            name: Key with which the model is registered.

        Usage:

            from core.utils.registry import registry

            if __name__ == "__main__":
                ...
                BaseTrainer = registry.get_trainer_class("base_trainer")
                ...
        """

        return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get_loss_class(cls, name):
        r"""Obtain registry loss class with key 'name'

        Args:
            name: Key with which the loss is registered.

        Usage:

            from core.utils.registry import registry

            if __name__ == "__main__":
                ...
                BaseTrainer = registry.get_loss_class("obj_n_penalty")
                ...
        """

        return cls.mapping["loss_name_mapping"].get(name, None)

registry = Registry()
