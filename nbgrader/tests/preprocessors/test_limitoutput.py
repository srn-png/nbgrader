import pytest
import os

from textwrap import dedent
from ...preprocessors import LimitOutput
from nbformat.v4 import new_output
from .base import BaseTestPreprocessor
from .. import create_code_cell, create_text_cell


@pytest.fixture
def preprocessor():
    return LimitOutput()


class TestLimitOutput(BaseTestPreprocessor):

    def test_long_output(self):
        nb = self._read_nb(os.path.join("files", "long-output.ipynb"))
        cell, = nb.cells
        output, = cell.outputs
        assert len(output.text.split("\n")) > 1000

        pp = LimitOutput()
        nb, resources = pp.preprocess(nb, {})

        cell, = nb.cells
        output, = cell.outputs
        assert len(output.text.split("\n")) == 1000

    def test_infinite_recursion(self):
        nb = self._read_nb(os.path.join("files", "infinite-recursion.ipynb"))

        pp = LimitOutput()
        nb, resources = pp.preprocess(nb, {})

        cell, = nb.cells
        output, = cell.outputs
        assert len(output.traceback) == 100

    def test_stderr_after_long_stdout(self):
        nb = self._read_nb(os.path.join("files", "long-output.ipynb"))
        cell, = nb.cells
        stderr_output = new_output("stream", name="stderr", text="Error text")
        cell.outputs.append(stderr_output)

        pp = LimitOutput()
        nb, resources = pp.preprocess(nb, {})

        cell, = nb.cells
        assert len(cell.outputs) == 2
        stdout_out, stderr_out = cell.outputs
        assert stdout_out.name == "stdout"
        assert len(stdout_out.text.split("\n")) == 1000
        assert stderr_out.name == "stderr"
        assert stderr_out.text == "Error text"
