import sys
import pathlib
import pytest

# Make the src/ folder importable so that tests can execute without installing the package
root = pathlib.Path(__file__).parent.parent
src_path = root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from process_execution.process_execution import ProcessExecution
from tree_search.feature import (
    NodeAttributeNumeric,
    NodeAttributeCategorical,
)


@pytest.fixture
def simple_process_execution():
    """Minimal ProcessExecution graph with a single node and numeric attribute."""
    p = ProcessExecution()
    p.add_node("n1", attr={"x": 0, "color": "red"})
    return p


@pytest.fixture
def numeric_feature():
    """A numeric feature whose attribute "x" can be changed between -2 and 2."""
    return NodeAttributeNumeric(
        node_id="n1",
        attribute_name="x",
        value_original=0,
        value_step=1,
        value_min=-2,
        value_max=2,
    )


@pytest.fixture
def categorical_feature():
    """A categorical feature representing a color attribute."""
    return NodeAttributeCategorical(
        node_id="n1",
        attribute_name="color",
        value_original="red",
        category_values=["red", "blue", "green"],
    )
