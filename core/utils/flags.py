# File with all of main's flags.

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# From an old codebase.

import argparse


class Flags:
    r"""This class parses the arguments passed when main.py is called. Should
    not be manually instantiated. Inspired on"""

    def __init__(
        self,
    ):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def add_core_args(
        self,
    ):
        self.parser.add_argument(
            "--config",
            type=str,
            default="none",
            help="Name and location of config file.",
        )
        self.parser.add_argument(
            "--is_debug",
            action="store_true",
            default=False,
            help="When true, no results are saved.",
        )
        self.parser.add_argument(
            "--seed",
            default=0,
            type=int,
            help="Sets the randomized seed for all libraries that use one.",
        )
        self.parser.add_argument(
            "--cpu",
            action="store_true",
            default=False,
            help="Forces training to happen in CPU, even if GPU is available.",
        )


flags = Flags()
